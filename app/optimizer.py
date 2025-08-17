# optimizer.py
import json, copy, itertools, io, re, argparse, traceback, shutil, random, time, os
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
    seen_set = set()
    for v in vals:
        key = (round(v, 12) if isinstance(v, float) else v)
        if key in seen_set:
            continue
        seen_set.add(key)
        out.append(v)
    return out

def expand_from_ranges(range_space: dict):
    """Grid expansion from min/step/max ranges into discrete value lists."""
    values = {}
    for path, spec in range_space.items():
        start = spec["start"]; stop = spec["stop"]; step = spec["step"]
        nd = spec.get("round", 10)
        seq = frange(start, stop, step, inclusive=True, ndigits=nd)
        values[path] = unique_list(seq)
    return values

def expand_grid(space: dict):
    """Cartesian product over dot-path -> [values...] into nested override dicts."""
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

def random_overrides_from_ranges(range_space: dict, n: int, seed: int | None = None):
    """Uniform random sampling within continuous ranges specified in range_space."""
    if seed is not None:
        random.seed(seed)
    keys = list(range_space.keys())
    specs = [range_space[k] for k in keys]
    overrides = []
    for _ in range(n):
        ov = {}
        for path, spec in zip(keys, specs):
            start = spec["start"]; stop = spec["stop"]; nd = spec.get("round", 6)
            val = round(random.uniform(start, stop), nd)
            deep_update(ov, build_override(path, val))
        overrides.append(ov)
    return overrides

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

def _set_num_threads():
    """Prevent BLAS oversubscription—1 thread per worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def print_top_k(leaderboard, k=10):
    print("\n===== TOP {} =====".format(k))
    for rank, row in enumerate(leaderboard[:k], 1):
        print(f"#{rank:02d}  capital={row['capital']:.6f}  override={row['override']}")

# -------- optimizer core --------
def run_overrides(overrides, processes: int, outdir: Path, base_cfg: dict):
    """Common runner for any list of overrides."""
    failures_dir = outdir / "failures"
    runs_root = outdir / "runs"

    total = len(overrides)
    print(f"Total unique combinations: {total}")

    worker_args = [(idx, base_cfg, ov, runs_root, failures_dir) for idx, ov in enumerate(overrides, 1)]

    leaderboard = []
    best = {"capital": float("-inf"), "override": None, "config": None}

    # Slightly recycle workers to avoid RAM creep in long runs
    with Pool(processes=processes, initializer=_set_num_threads, maxtasksperchild=50) as pool:
        # Larger chunksize reduces IPC overhead when tasks are long (~2 min)
        chunksize = max(1, total // (processes * 8)) or 1
        iterator = pool.imap_unordered(evaluate_override, worker_args, chunksize)
        if tqdm:
            iterator = tqdm(iterator, total=total)

        start = time.time()
        for i, capital, ov, cfg in iterator:
            leaderboard.append({"i": i, "capital": capital, "override": ov})
            if capital > best["capital"]:
                best = {"capital": capital, "override": ov, "config": cfg or base_cfg}
            if len(leaderboard) % processes == 0:
                elapsed = time.time() - start
                rps = len(leaderboard) / max(elapsed, 1e-6)
                print(f"[progress] {len(leaderboard)}/{total}  "
                      f"{elapsed/60:.1f} min  {rps:.3f} runs/s", flush=True)

    leaderboard.sort(key=lambda x: x["capital"], reverse=True)

    # save results at the end
    (outdir / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
    (outdir / "best_override.json").write_text(json.dumps(best["override"] or {}, indent=2))
    (outdir / "best_config.json").write_text(json.dumps(best["config"] or base_cfg, indent=2))
    (outdir / "best_capital.json").write_text(json.dumps({"capital": best["capital"]}, indent=2))

    print_top_k(leaderboard, k=10)
    print("\n===== BEST RESULT =====")
    print(f"Capital: {best['capital']:.6f}")
    print(json.dumps(best["override"], indent=2))
    print(f"Saved results to {outdir.resolve()}")

    return leaderboard, best

# -------- refine helpers --------
def load_best_config(outdir: Path) -> dict:
    p = outdir / "best_config.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run a prior stage first.")
    return json.loads(p.read_text())

def build_local_refine_space(best_cfg: dict,
                             d_r2_l1: float, d_slope_l1: float,
                             d_r2_l2: float, d_slope_l2: float,
                             step_r2: float, step_slope_l1: float, step_slope_l2: float):
    b1 = best_cfg["trend_filters_settings"]["layer_1"]["slope_and_r2"]
    b2 = best_cfg["trend_filters_settings"]["layer_2"]["slope_and_r2"]
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    return {
        # L1
        "trend_filters_settings.layer_1.slope_and_r2.r2_threshold": {
            "start": clamp(b1["r2_threshold"] - d_r2_l1, 0.0, 0.90),
            "stop":  clamp(b1["r2_threshold"] + d_r2_l1, 0.0, 0.90),
            "step":  step_r2, "round": 2
        },
        "trend_filters_settings.layer_1.slope_and_r2.slope_threshold": {
            "start": clamp(b1["slope_threshold"] - d_slope_l1, 0.0, 1.0),
            "stop":  clamp(b1["slope_threshold"] + d_slope_l1, 0.0, 1.0),
            "step":  step_slope_l1, "round": 8
        },
        # L2
        "trend_filters_settings.layer_2.slope_and_r2.r2_threshold": {
            "start": clamp(b2["r2_threshold"] - d_r2_l2, 0.0, 0.90),
            "stop":  clamp(b2["r2_threshold"] + d_r2_l2, 0.0, 0.90),
            "step":  step_r2, "round": 2
        },
        "trend_filters_settings.layer_2.slope_and_r2.slope_threshold": {
            "start": clamp(b2["slope_threshold"] - d_slope_l2, 0.0, 1.0),
            "stop":  clamp(b2["slope_threshold"] + d_slope_l2, 0.0, 1.0),
            "step":  step_slope_l2, "round": 8
        },
    }

# -------- main modes --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--procs", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--mode", type=str, default="grid",
                        choices=["grid", "coarse_random", "refine_from_best"],
                        help="Search mode")
    parser.add_argument("--budget", type=int, default=400,
                        help="For coarse_random: number of random samples")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: ./tuning_results)")

    # local refine knobs
    parser.add_argument("--k", type=int, default=1,
                        help="Not used in this minimal refine (kept for future top-K refine)")
    parser.add_argument("--d_r2_l1", type=float, default=0.10)
    parser.add_argument("--d_slope_l1", type=float, default=0.00020)
    parser.add_argument("--d_r2_l2", type=float, default=0.10)
    parser.add_argument("--d_slope_l2", type=float, default=0.00008)
    parser.add_argument("--step_r2", type=float, default=0.02)
    parser.add_argument("--step_slope_l1", type=float, default=0.00002)
    parser.add_argument("--step_slope_l2", type=float, default=0.00001)

    args = parser.parse_args()

    base_cfg_path = Path(strategy.__file__).parent / "config" / "config.json"
    base_cfg = json.loads(Path(base_cfg_path).read_text())
    outdir = Path(args.outdir) if args.outdir else (Path(__file__).parent / "tuning_results")
    outdir.mkdir(parents=True, exist_ok=True)

    # Default global ranges (used by grid & coarse_random)
    RANGE_SPACE = {
        # # layer 1
        "trend_filters_settings.layer_1.slope_and_r2.r2_threshold": {
            "start": 0.00, "stop": 0.90, "step": 0.10, "round": 2
        },
        "trend_filters_settings.layer_1.slope_and_r2.slope_threshold": {
            "start": 0.0000, "stop": 0.0020, "step": 0.0002, "round": 6
        },
        "trend_filters_settings.layer_1.slope_and_r2.trend_slope_window": {
            "start": 3, "stop": 30, "step": 3, "round": 0
        },
        # # layer 2
        # "trend_filters_settings.layer_2.slope_and_r2.trend_slope_window":{"start": 3,"stop": 30,"step": 2,"round": 0},
        # "trend_filters_settings.layer_2.slope_and_r2.r2_threshold":
        #     {"start": 0.684, "stop": 0.691, "step": 0.001, "round": 3},
        # "trend_filters_settings.layer_2.slope_and_r2.slope_threshold":
        #     {"start": 0.00016, "stop": 0.00018, "step": 0.000005, "round": 8}
    }

    if args.mode == "grid":
        space = expand_from_ranges(RANGE_SPACE)
        overrides = list(expand_grid(space))
        run_overrides(overrides, args.procs, outdir, base_cfg)

    elif args.mode == "coarse_random":
        overrides = random_overrides_from_ranges(RANGE_SPACE, n=args.budget, seed=args.seed)
        run_overrides(overrides, args.procs, outdir, base_cfg)

    elif args.mode == "refine_from_best":
        best_cfg = load_best_config(outdir)
        local_space = build_local_refine_space(
            best_cfg,
            d_r2_l1=args.d_r2_l1, d_slope_l1=args.d_slope_l1,
            d_r2_l2=args.d_r2_l2, d_slope_l2=args.d_slope_l2,
            step_r2=args.step_r2,
            step_slope_l1=args.step_slope_l1,
            step_slope_l2=args.step_slope_l2,
        )
        space = expand_from_ranges(local_space)
        overrides = list(expand_grid(space))
        run_overrides(overrides, args.procs, outdir, base_cfg)

if __name__ == "__main__":
    main()