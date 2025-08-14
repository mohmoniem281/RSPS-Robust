# optimizer.py
import json, copy, itertools, io, re, argparse, traceback, shutil
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from multiprocessing import Pool

# try tqdm; fall back silently
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# === Import your strategy entry ===
# strategy.py must expose: def test(config: dict) -> None
import strategy

# robust: handles commas, decimals, sci-notation, extra punctuation
CAPITAL_LINE_REGEX = re.compile(
    r"capital\s*to\s*check\s*non\s*filtered.*?([0-9]+(?:[.,][0-9]+)?(?:e[+-]?\d+)?)",
    re.IGNORECASE | re.DOTALL
)

# -------- helpers --------
def deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def build_override(path: str, value):
    cur = {}
    node = cur
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return cur

def frange(start: float, stop: float, step: float, *, inclusive=True, ndigits=10):
    if step == 0:
        raise ValueError("step must be non-zero")
    vals, n, x = [], 0, start
    cmp = (lambda a, b: a <= b + 1e-12) if step > 0 else (lambda a, b: a >= b - 1e-12)
    while cmp(x, stop) if inclusive else cmp(x, stop - step):
        vals.append(round(x, ndigits))
        n += 1
        x = start + n * step
    return vals

def unique_list(vals, tol=1e-12):
    out = []
    for v in vals:
        if not out:
            out.append(v); continue
        if isinstance(v, float):
            j = len(out) - 1
            while j >= 0 and not isinstance(out[j], float):
                j -= 1
            if j >= 0 and abs(v - out[j]) <= tol:
                continue
        if v not in out:
            out.append(v)
    return out

def expand_from_ranges(range_space: dict):
    values = {}
    for path, spec in range_space.items():
        start = spec["start"]; stop = spec["stop"]; step = spec["step"]
        nd = spec.get("round", 10)
        seq = frange(start, stop, step, inclusive=True, ndigits=nd)
        values[path] = unique_list(seq)
    return values

def expand_grid(space: dict):
    if not space:
        yield {}
        return
    keys, values_list = zip(*space.items())
    values_list = [unique_list(v) for v in values_list]
    seen = set()
    for combo in itertools.product(*values_list):
        ov = {}
        for path, val in zip(keys, combo):
            deep_update(ov, build_override(path, val))
        sig = json.dumps(ov, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        yield ov

def run_and_capture_capital(cfg: dict):
    """
    Run strategy.test(config), capture stdout+stderr, return (capital, stdout_tail).
    Raises on parse failure.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        strategy.test(cfg)

    out = buf.getvalue()
    m = CAPITAL_LINE_REGEX.search(out)
    if not m:
        raise RuntimeError("Could not find capital line in output.")
    raw = m.group(1).replace(",", "")
    capital = float(raw)
    tail = "\n".join(out.splitlines()[-200:])
    return capital, tail

def _rewrite_path(base_dir: Path, original: str, *sub):
    """Place the file under base_dir, preserving the filename."""
    name = Path(original).name
    p = base_dir.joinpath(*sub, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)

def isolate_output_paths(cfg: dict, run_dir: Path) -> dict:
    """
    Rewrite ALL output file paths to a per-run directory to avoid collisions.
    """
    cfg = copy.deepcopy(cfg)

    # top-level outputs
    for key in [
        "identifiers_file_path",
        "tournament_data_path",
        "equity_curve_file",
        "equity_curve_visualization",
        "equity_curve_file_filtered",
        "equity_curve_visualization_filtered",
        "index_file_path",
        "ratios_file_path",
    ]:
        if key in cfg:
            cfg[key] = _rewrite_path(run_dir, cfg[key])

    # assets outputs (price_history, normalized_history)
    if "assets" in cfg:
        assets = cfg["assets"]
        new_assets = {}
        for a, paths in assets.items():
            new_paths = {}
            for k, v in paths.items():
                new_paths[k] = _rewrite_path(run_dir / "assets" / a, v)
            new_assets[a] = new_paths
        cfg["assets"] = new_assets

    return cfg

# -------- worker --------
def evaluate_override(args):
    """
    Args: (index, base_cfg, override, runs_root, failures_dir)
    Returns: (index, capital, override, cfg_or_None)
    """
    i, base_cfg, override, runs_root, failures_dir = args
    run_dir = runs_root / f"run_{i:06d}"
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = isolate_output_paths(base_cfg, run_dir)
        deep_update(cfg, override)
        capital, tail = run_and_capture_capital(cfg)

        # ✅ success: delete the per-run directory to save disk
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        return i, capital, override, cfg

    except Exception as e:
        # ❌ failure: move the whole run_dir under failures and write fail.txt
        try:
            failures_dir.mkdir(parents=True, exist_ok=True)
            failed_run_dir = failures_dir / f"run_{i:06d}"
            # if exists from a prior attempt, clear it
            if failed_run_dir.exists():
                shutil.rmtree(failed_run_dir, ignore_errors=True)
            if run_dir.exists():
                shutil.move(str(run_dir), str(failed_run_dir))
            (failed_run_dir / "fail.txt").write_text(
                json.dumps({"override": override, "error": f"{e}\n{traceback.format_exc()}"}, indent=2)
                + "\n\n---- stdout/stderr tail ----\n"
                + (tail if 'tail' in locals() else "")
            )
        except Exception:
            pass
        return i, float("-inf"), override, None

# -------- optimizer --------
def optimize(
    explicit_space: dict | None = None,
    range_space: dict | None = None,
    outdir: Path | None = None,
    processes: int = 8,
):
    base_cfg_path = Path(strategy.__file__).parent / "config" / "config.json"
    base_cfg = json.loads(Path(base_cfg_path).read_text())

    unified_space = {}
    if range_space:
        unified_space.update(expand_from_ranges(range_space))
    if explicit_space:
        for k, v in explicit_space.items():
            unified_space[k] = unique_list(list(v))
    if not unified_space:
        raise ValueError("No search space provided (explicit or ranges).")

    outdir = outdir or (Path(__file__).parent / "tuning_results")
    outdir.mkdir(parents=True, exist_ok=True)
    failures_dir = outdir / "failures"
    runs_root = outdir / "runs"

    overrides = list(expand_grid(unified_space))
    total = len(overrides)
    print(f"Total unique combinations: {total}")

    worker_args = [(idx, base_cfg, ov, runs_root, failures_dir) for idx, ov in enumerate(overrides, 1)]

    leaderboard = []
    best = {"capital": float("-inf"), "override": None, "config": None}

    with Pool(processes=processes) as pool:
        iterator = pool.imap_unordered(evaluate_override, worker_args, 1)  # stream asap
        if tqdm:
            iterator = tqdm(iterator, total=total)

        for i, capital, ov, cfg in iterator:
            leaderboard.append({"i": i, "capital": capital, "override": ov})
            if capital > best["capital"]:
                best = {"capital": capital, "override": ov, "config": cfg or base_cfg}
            print(f"[{i}] capital={capital:.6f} override={ov}", flush=True)

    leaderboard.sort(key=lambda x: x["capital"], reverse=True)

    (outdir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    (outdir / "best_override.json").write_text(json.dumps(best["override"] or {}, indent=2))
    (outdir / "best_config.json").write_text(json.dumps(best["config"] or base_cfg, indent=2))
    (outdir / "best_capital.json").write_text(json.dumps({"capital": best["capital"]}, indent=2))

    print("\n===== BEST RESULT =====")
    print(f"Capital: {best['capital']:.6f}")
    print(json.dumps(best["override"], indent=2))
    print(f"Saved results to {outdir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    RANGE_SPACE = {
        #layer 1
        "trend_filters_settings.layer_1.slope_and_r2.r2_threshold":   {"start": 0.00, "stop": 0.90, "step": 0.10, "round": 2},
        "trend_filters_settings.layer_1.slope_and_r2.slope_threshold": {"start": 0.0000, "stop": 0.0010, "step": 0.0002, "round": 6},
        #layer 2
        "trend_filters_settings.layer_2.slope_and_r2.r2_threshold":   {"start": 0.00, "stop": 0.90, "step": 0.10, "round": 2},
        "trend_filters_settings.layer_2.slope_and_r2.slope_threshold": {"start": 0.00000, "stop": 0.00030, "step": 0.00005, "round": 8},
    }

    EXPLICIT_SPACE = {
        # optional hand-picked values to merge
        # "trend_filters_settings.layer_2.slope_and_r2.r2_threshold": [0.62, 0.66, 0.70],
    }

    optimize(EXPLICIT_SPACE, RANGE_SPACE, processes=args.procs)