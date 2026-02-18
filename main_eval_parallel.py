import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
from sigmarl.helper_common import get_name_suffix
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import os

os.environ["PYGLET_HEADLESS"] = "true"


PYTHON_EXEC = "python"
SCRIPT = "main_eval.py"
OUTPUT_DIR = "outputs/marl_cbf_new/cbf_informed_rl/seed1"

LOG_DIR = Path(OUTPUT_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"See logs at {LOG_DIR}")


print("Launcher started", flush=True)


def build_configs(is_skip_if_exist: bool = True):
    configs = []
    skipped = 0

    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    grouping_flags = [True, False]
    nom_controllers = ["rl"]
    # ["cpm_entire", "intersection_6", "intersection_4", "intersection_6", "interchange_3"]
    scenario_types = [
        "intersection_6",
        "intersection_4",
        "interchange_3",
        "cpm_entire",
        "cpm_mixed",
    ]
    # scenario_types = ["cpm_entire", "interchange_3"]
    is_using_cbf_testing_list = [True, False]

    for scenario in scenario_types:
        if scenario == "cpm_entire":
            n_agents = [10]
            max_group_sizes = [1, 2, 5, 10]
        elif scenario == "intersection_6":
            n_agents = [10]
            max_group_sizes = [1, 2, 5, 10]
        elif scenario == "intersection_4":
            n_agents = [8]
            max_group_sizes = [1, 2, 4, 8]
        elif scenario == "interchange_3":
            n_agents = [8]
            max_group_sizes = [1, 2, 4, 8]
        elif scenario == "cpm_mixed":
            n_agents = [4]
            max_group_sizes = [1, 2, 3, 4]
        else:
            raise ValueError(
                f"Please define n_agents and max_group_sizes for scenario type: {scenario}"
            )

        # n_agents = [SCENARIOS[scenario]["n_agents"]]

        print(scenario, n_agents)
        for n_agent in n_agents:
            for seed in seeds:

                for grouping in grouping_flags:
                    if grouping:
                        group_sizes_iter = max_group_sizes
                    else:
                        group_sizes_iter = [None]

                    for max_group in group_sizes_iter:
                        for nom in nom_controllers:
                            for is_using_cbf_testing in is_using_cbf_testing_list:
                                name_suffix = get_name_suffix(
                                    grouping,
                                    is_using_cbf_testing,
                                    n_agent,
                                    seed,
                                    max_group,
                                    nom,
                                    scenario,
                                )
                                td_name = f"out_td_{name_suffix}.td"

                                td_path = os.path.join(OUTPUT_DIR, td_name)

                                if is_skip_if_exist and os.path.exists(td_path):
                                    skipped += 1
                                    continue
                                cmd = [
                                    PYTHON_EXEC,
                                    SCRIPT,
                                    "--output_dir",
                                    OUTPUT_DIR,
                                    "--n_agents",
                                    str(n_agent),
                                    "--random_seed",
                                    str(seed),
                                    "--nom_controller_type",
                                    nom,
                                    "--scenario_type",
                                    scenario,
                                ]

                                if is_using_cbf_testing:
                                    cmd.append("--is_using_cbf_testing")

                                if grouping:
                                    cmd += [
                                        "--is_grouping_agents",
                                        "--max_group_size",
                                        str(max_group),
                                    ]

                                configs.append(cmd)

    return configs, skipped


def run_command(cmd):
    start_time = time.time()
    print("\r[RUN]", " ".join(cmd), flush=True)

    suffix = get_name_suffix(
        "--is_grouping_agents" in cmd,
        "--is_using_cbf_testing" in cmd,
        int(cmd[cmd.index("--n_agents") + 1]),
        int(cmd[cmd.index("--random_seed") + 1]),
        int(cmd[cmd.index("--max_group_size") + 1])
        if "--max_group_size" in cmd
        else None,
        cmd[cmd.index("--nom_controller_type") + 1],
        cmd[cmd.index("--scenario_type") + 1],
    )
    log_path = LOG_DIR / f"{suffix}.log"

    with open(log_path, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=f, text=True)

    return time.time() - start_time


if __name__ == "__main__":
    t_0 = time.time()

    is_skip_if_exist = True
    num_workers = 4

    configs, skipped = build_configs(is_skip_if_exist)
    total_jobs = len(configs)

    print(f"[INFO] Jobs to run: {total_jobs}", flush=True)
    if skipped > 0:
        print(f"[INFO] Jobs skipped (existing results): {skipped}", flush=True)

    if total_jobs == 0:
        print("[INFO] Nothing to do.")
        exit(0)

    bar_width = 30
    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in configs]

        for f in as_completed(futures):
            completed += 1
            elapsed = f.result()  # seconds for this job

            progress = completed / total_jobs
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)

            print(
                f"\r  [{bar}] {completed}/{total_jobs} jobs completed "
                f"(last job: {elapsed:.2f} s)",
                end="",
                flush=True,
            )

    print("\n" + "=" * 60)
    print(f"Launcher finished all jobs in {time.time() - t_0:.2f} s")
