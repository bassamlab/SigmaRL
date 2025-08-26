import os
import re
import gc
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch

from sigmarl.helper_training import SaveData, reduce_out_td
from sigmarl.mappo_cavs import mappo_cavs
from vmas.simulator.utils import save_video


# ─────────────── Tunables (start conservative) ───────────────
DEFAULT_MAX_WORKERS = 3  # Increase to use more workers
BATCH_SIZE = 4  # how many jobs to run in one wave
SAVE_VIDEO = True
MAX_STEPS = 1800


def cap_threads():
    """Cap BLAS / Torch threads to avoid oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def safe_mkdirs(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def atomic_save_torch(obj, dst_path: str) -> None:
    tmp_path = str(Path(dst_path).with_suffix(Path(dst_path).suffix + ".tmp"))
    torch.save(obj, tmp_path)
    os.replace(tmp_path, dst_path)


def find_first_json(path_dir: str) -> str:
    for file in os.listdir(path_dir):
        if file.endswith(".json"):
            return os.path.join(path_dir, file)
    raise FileNotFoundError(f"No json file found in {path_dir!r}.")


def parse_row(row: pd.Series):
    seed_str = [p for p in str(row["file"]).split("/") if p.startswith("seed")][0]
    seed_num = seed_str.replace("seed", "")
    base_path = f"checkpoints/at25/sigmarl/seed{seed_num}/"

    td_name = [p for p in str(row["file"]).split("/") if p.startswith("init")][0]
    out_td_path = os.path.join(base_path, td_name)
    video_path = out_td_path.replace(".td", "")

    match = re.search(r"(\d+)(?!.*\d)", td_name)
    if not match:
        raise ValueError(
            f"No trailing number in td_name={td_name!r} (file={row['file']!r})"
        )
    random_seed = int(match.group(1))

    predefined_ref_path_idx = [int(x) for x in str(row["ref_path_ids"]).split("|")]
    init_state = [
        [row["v1_x"], row["v1_y"], row["v1_theta"]],
        [row["v2_x"], row["v2_y"], row["v2_theta"]],
        [row["v3_x"], row["v3_y"], row["v3_theta"]],
    ]

    return dict(
        base_path=base_path,
        out_td_path=out_td_path,
        video_path=video_path,
        random_seed=random_seed,
        predefined_ref_path_idx=predefined_ref_path_idx,
        init_state=init_state,
    )


def run_one(job: dict, save_video_flag: bool) -> tuple:
    """
    Returns (out_td_path, video_path, ok, message)
    """
    cap_threads()

    base_path = job["base_path"]
    out_td_path = job["out_td_path"]
    video_path = job["video_path"]

    # Check if out_td_path exist (enable if you want to skip the run whose results are already available)
    # if os.path.exists(out_td_path):
    #     print(f"File {out_td_path} exists. Return.")
    #     return (out_td_path, video_path + ".mp4", True, "ok")

    try:
        safe_mkdirs(out_td_path)
        if save_video_flag:
            safe_mkdirs(video_path + ".mp4")

        # Load parameters
        with open(find_first_json(base_path), "r") as f:
            data = json.load(f)
        saved_data = SaveData.from_dict(data)
        parameters = saved_data.parameters

        # Per-job overrides
        parameters.predefined_ref_path_idx = job["predefined_ref_path_idx"]
        parameters.init_state = job["init_state"]
        parameters.random_seed = job["random_seed"]

        # Runtime settings (lighter defaults)
        parameters.is_testing_mode = True
        parameters.is_real_time_rendering = False
        parameters.is_save_eval_results = False
        parameters.is_load_model = True
        parameters.is_load_final_model = False
        parameters.is_load_out_td = False
        parameters.max_steps = MAX_STEPS

        parameters.num_vmas_envs = 1

        parameters.scenario_type = "CPM_entire"
        parameters.n_agents = 3

        parameters.is_save_simulation_video = bool(save_video_flag)
        parameters.is_visualize_short_term_path = True
        parameters.is_visualize_lane_boundary = False
        parameters.is_visualize_extra_info = True

        # Build env and modules
        (
            env,
            decision_making_module,
            optimization_module,
            priority_module,
            cbf_controllers,
            parameters,
        ) = mappo_cavs(parameters=parameters)

        # Rollout
        out_td, frame_list = env.rollout(
            max_steps=parameters.max_steps - 1,
            policy=decision_making_module.policy,
            priority_module=priority_module,
            callback=lambda env, _: env.render(
                mode="rgb_array", visualize_when_rgb=False
            ),
            auto_cast_to_device=True,
            break_when_any_done=False,
            is_save_simulation_video=parameters.is_save_simulation_video,
        )

        # Save outputs
        out_td = reduce_out_td(out_td)
        atomic_save_torch(out_td, out_td_path)

        if parameters.is_save_simulation_video:
            save_video(video_path, frame_list, fps=1 / parameters.dt)

        # Cleanup to help the OS
        del (
            env,
            decision_making_module,
            optimization_module,
            priority_module,
            cbf_controllers,
        )
        del frame_list, out_td
        gc.collect()

        return (out_td_path, video_path + ".mp4", True, "ok")

    except Exception as e:
        return (out_td_path, video_path + ".mp4", False, repr(e))


def run_in_batches(
    jobs,
    max_workers=DEFAULT_MAX_WORKERS,
    batch_size=BATCH_SIZE,
    save_video_flag=SAVE_VIDEO,
):
    total = len(jobs)
    done = 0
    for start in range(0, total, batch_size):
        batch = jobs[start : start + batch_size]
        print(
            f"[INFO] Batch {start//batch_size + 1}: {len(batch)} jobs, max_workers={max_workers}"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_one, job, save_video_flag) for job in batch]
            for fut in as_completed(futs):
                out_td_path, video_path, ok, msg = fut.result()
                done += 1
                if ok:
                    print(f"[{done}/{total}] ✔ td: {out_td_path} | video: {video_path}")
                else:
                    print(
                        f"[{done}/{total}] ✘ FAILED for {out_td_path} | {video_path} — {msg}"
                    )


def main(csv_path: str, max_workers: int = DEFAULT_MAX_WORKERS):
    cap_threads()
    df = pd.read_csv(csv_path)
    jobs = [parse_row(row) for _, row in df.iterrows()]
    print(f"[INFO] Total jobs: {len(jobs)}")
    run_in_batches(
        jobs, max_workers=max_workers, batch_size=BATCH_SIZE, save_video_flag=SAVE_VIDEO
    )


if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    main("checkpoints/at25/sigmarl/poses.csv", max_workers=DEFAULT_MAX_WORKERS)
