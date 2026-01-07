import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os

PYTHON_EXEC = "python"
SCRIPT = "main_eval.py"

OUTPUT_DIR = "outputs/marl_cbf_0"

print("Launcher started", flush=True)


def build_configs(is_skip_if_exist: bool):
    configs = []
    skipped = 0

    n_agents = [8, 10, 12, 14, 16]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    grouping_flags = [False, True]
    max_group_sizes = [1, 2, 3, 4]
    nom_controllers = ["clf"]
    scenario_types = ["CPM_entire"]

    for n_agent in n_agents:
        for seed in seeds:
            for grouping in grouping_flags:
                for max_group in max_group_sizes:
                    for nom in nom_controllers:
                        for scenario in scenario_types:

                            grouping_tag = "on" if grouping else "off"
                            td_name = (
                                f"out_td_agents_{n_agent}_seed_{seed}_"
                                f"grouping_{grouping_tag}_maxgroup_{max_group}_"
                                f"nom_{nom}_scenario_{scenario.lower()}.td"
                            )
                            td_path = os.path.join(OUTPUT_DIR, td_name)

                            if is_skip_if_exist and os.path.exists(td_path):
                                skipped += 1
                                continue

                            cmd = [
                                PYTHON_EXEC,
                                SCRIPT,
                                "--n_agent",
                                str(n_agent),
                                "--random_seed",
                                str(seed),
                                "--max_group_size",
                                str(max_group),
                                "--nom_controller_type",
                                nom,
                                "--scenario_type",
                                scenario,
                            ]
                            if grouping:
                                cmd.append("--is_grouping_agents")

                            configs.append(cmd)

    return configs, skipped


def run_command(cmd):
    start_time = time.time()
    print("[RUN]", " ".join(cmd), flush=True)
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("\n[SUBPROCESS FAILED]")
        print("Command:", " ".join(e.cmd))
        print("Return code:", e.returncode)
        print("\n--- STDOUT ---")
        print(e.stdout)
        print("\n--- STDERR ---")
        print(e.stderr)
        print("----------------\n")
        raise
    finally:
        elapsed = time.time() - start_time

    return elapsed


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
