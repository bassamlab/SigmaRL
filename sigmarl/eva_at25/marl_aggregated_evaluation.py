# Usage example:
"""Run the following in the terminal
  python sigmarl/eva_at25/marl_aggregated_evaluation.py \
    --base_dir checkpoints/at25 \
    --mapping_csv checkpoints/at25/sigmarl/ref_paths.csv \
    --envs sigmarl \
    --seeds 1 2 3 --inits 1 2 3 --reps 1 2 3 \
    --veh_width 0.107 --veh_length 0.220 \
    --per_unit_m 100 \
    --out_dir metrics_out \
    --single_module sigmarl/eva_at25/marl_evaluation.py
"""

import argparse
import os
import importlib.util
from typing import List
import numpy as np
import pandas as pd

# Import evaluate_single_run from single_run_eval.py (local file import)
def load_single_run_eval(module_path: str = "marl_evaluation.py"):
    spec = importlib.util.spec_from_file_location("marl_evaluation", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def ci95_mean(x: np.ndarray):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    se = sd / max(np.sqrt(x.size), 1.0)
    margin = 1.96 * se
    return (m, sd, m - margin, m + margin)


def iqm(x: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    x.sort()
    n = x.size
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    if hi <= lo:
        return np.nan
    return float(np.mean(x[lo:hi]))


def parse_ids(s: str):
    s = str(s).replace(" ", "").replace("|", ",")
    return [int(x) for x in s.split(",") if x != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="results")
    ap.add_argument("--mapping_csv", type=str, required=True)
    ap.add_argument(
        "--envs", type=str, nargs="+", default=["digital_twin", "physical_lab"]
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--reps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--veh_width", type=float, default=0.107)
    ap.add_argument("--veh_length", type=float, default=0.220)
    ap.add_argument("--per_unit_m", type=float, default=10.0)
    ap.add_argument(
        "--scenario", type=str, default="CPM_entire"
    )  # fallback if missing in CSV
    ap.add_argument("--n1", type=int, default=3)
    ap.add_argument("--n2", type=int, default=5)
    ap.add_argument(
        "--single_module",
        type=str,
        default="marl_evaluation.py",
        help="path to single_run_eval.py",
    )
    ap.add_argument("--out_dir", type=str, default="metrics_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    single = load_single_run_eval(args.single_module)

    # Load mapping CSV
    df_map = pd.read_csv(args.mapping_csv)
    df_map.columns = [c.strip().lower() for c in df_map.columns]
    if "scenario" not in df_map.columns:
        df_map["scenario"] = args.scenario

    rows = []
    for env in args.envs:
        for seed in args.seeds:
            for init in args.inits:
                for rep in args.reps:
                    mrow = df_map[
                        (df_map["env"] == env)
                        & (df_map["seed"] == seed)
                        & (df_map["init"] == init)
                        & (df_map["rep"] == rep)
                    ]
                    if mrow.empty:
                        print(
                            f"[SKIP] missing mapping for {env} seed{seed} init{init}-{rep}"
                        )
                        continue
                    ref_ids = parse_ids(str(mrow.iloc[0]["ref_path_ids"]))
                    scenario = str(mrow.iloc[0]["scenario"])

                    td_path = os.path.join(
                        args.base_dir, env, f"seed{seed}", f"init{init}-{rep}.td"
                    )
                    if not os.path.exists(td_path):
                        print(f"[SKIP] missing file {td_path}")
                        continue

                    try:
                        metrics = single.evaluate_single_run(
                            td_path=td_path,
                            ref_path_ids=ref_ids,
                            scenario=scenario,
                            veh_width=args.veh_width,
                            veh_length=args.veh_length,
                            per_unit_m=args.per_unit_m,
                            max_steps=1800,
                            n1=args.n1,
                            n2=args.n2,
                            make_plots=False,
                        )
                        metrics.update(env=env, seed=seed, init=init, rep=rep)
                        rows.append(metrics)
                    except Exception as e:
                        print(f"[WARN] failed on {td_path}: {e}")

    if not rows:
        raise SystemExit("No results collected.")

    df = pd.DataFrame(rows)
    per_run_csv = os.path.join(args.out_dir, "per_run.csv")
    df.to_csv(per_run_csv, index=False)
    print(f"Wrote {per_run_csv} ({len(df)} rows)")

    # Aggregations
    def summarize(group: pd.DataFrame, labels: List[str]):
        metrics = [
            "CRA_A_per_unit",
            "LaneViolation_per_unit",
            "CD_mean_m",
            "AS_mean_mps",
        ]
        out = []
        for m in metrics:
            vals = group[m].to_numpy(dtype=float)
            mean, sd, lo, hi = ci95_mean(vals)
            row = {k: group.iloc[0][k] for k in labels}
            row.update(
                metric=m,
                mean=mean,
                sd=sd,
                ci95_lo=lo,
                ci95_hi=hi,
                iqm=iqm(vals),
                n=int(np.sum(~np.isnan(vals))),
            )
            out.append(row)
        return pd.DataFrame(out)

    env_summary = (
        df.groupby(["env"])
        .apply(lambda g: summarize(g, ["env"]))
        .reset_index(drop=True)
    )
    env_csv = os.path.join(args.out_dir, "env_summary.csv")
    env_summary.to_csv(env_csv, index=False)
    print(f"Wrote {env_csv}")

    seed_summary = (
        df.groupby(["env", "seed"])
        .apply(lambda g: summarize(g, ["env", "seed"]))
        .reset_index(drop=True)
    )
    seed_csv = os.path.join(args.out_dir, "seed_summary.csv")
    seed_summary.to_csv(seed_csv, index=False)
    print(f"Wrote {seed_csv}")

    init_summary = (
        df.groupby(["env", "init"])
        .apply(lambda g: summarize(g, ["env", "init"]))
        .reset_index(drop=True)
    )
    init_csv = os.path.join(args.out_dir, "init_summary.csv")
    init_summary.to_csv(init_csv, index=False)
    print(f"Wrote {init_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
