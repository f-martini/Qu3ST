import gc
import math
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn import cluster
from .iterations_dataframe import get_classical_baseline
from .ratio_dataframe import get_hammered


def sampler_decorator(row: pd.Series):
    return f"{row['Optimizer']} ({row['Shots']})" \
        if row['Optimizer'] == "SAMPLER" else row['Optimizer']


def count_ones(s: str) -> int:
    return sum([int(c) for c in s])


def str2list(s: str) -> list:
    to_remove = ["[", "]", "(", ")", "\n", ]
    for c in to_remove:
        s = s.replace(c, "")
    return [int(i) for i in s.split(" ") if i != ""]


def load_optimal_solution(res_dir: Path) -> dict:
    with open(res_dir / "results.json") as file:
        res = json.load(file)
    return {
        "solver": res["solver"],
        "transactions": str2list(res["transactions"]),
        "collateral": str2list(res["collateral"]),
        "cb_indicators": str2list(res["cb_indicators"]),
        "spl_indicators": str2list(res["spl_indicators"]),
        "evaluation": float(res["evaluation"]),
        "evaluation_call": int(res["evaluation_call"]),
        "runtime": float(res["runtime"]),
    }


def load_iter_result(res_dir: Path) -> dict:
    with open(res_dir / "iter_results.json") as file:
        res = json.load(file)
    return res


def get_line(df, pivot, values):
    grouped_stats = df.groupby(pivot)[values].agg(['mean', 'std'])
    x = grouped_stats.index.to_series()
    y = grouped_stats['mean']
    sd = grouped_stats['std']
    return x, y, sd


def plot_loss(base_dir: Path, config: dict, opt: dict, size: int, rep: int):
    optimizer = config["opt_mode"]
    h = "_HAMMER" if get_hammered(config) else ""
    if (base_dir / f"optimization_loss_{size}_{rep}_{optimizer}{h}.pdf").exists():
        return

    iterations_dir = sorted([
        (item, int(item.name.split("_")[1]))
        for item in base_dir.iterdir() if ("iteration_" in item.name)
    ], key=lambda x: x[1])
    evs = []
    bests = []
    for item, n in iterations_dir:
        try:
            iter_res = load_iter_result(item)
            evs.append(-iter_res["ev"])
            bests.append(-iter_res["best"]["of"])
        except (FileNotFoundError, KeyError):
            break
    if len(evs) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.hlines(0, 0, len(evs),
               linestyles=":", color="green", label="Validity Threshold")
    plt.hlines(opt["evaluation"], 0, len(evs),
               linestyles=":", color="orange", label="Optimal Payoff")
    plt.ylim(min(evs), 10)
    plt.yscale("symlog", linthresh=10 ** 0)
    plt.plot(range(len(evs)), evs, label="Point Evaluation")
    plt.plot(range(len(evs)), bests, linestyle="--", label="Best Solution")
    plt.ylabel("Payoff")
    plt.xlabel("Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base_dir / f"optimization_loss_{size}_{rep}_{optimizer}{h}.pdf")
    plt.close()


def plot_hamming_dist(base_dir: Path,
                      config: dict,
                      res: dict,
                      opt: dict,
                      size: int,
                      rep: int):

    optimizer = config["opt_mode"]
    if optimizer not in ["MBD", "QUEEN"]:
        return

    h = "_HAMMER" if get_hammered(config) else ""
    if (base_dir / f"hamming_weight_{size}_{rep}_{optimizer}{h}.pdf").exists():
        return

    hd_opt = sum([1 for c in opt["transactions"] if c == 1])
    hd_best = sum([1 for c in res["transactions"] if c == "1"])
    iterations_dir = sorted([
        (item, int(item.name.split("_")[1]))
        for item in base_dir.iterdir() if ("iteration_" in item.name)
    ], key=lambda x: x[1])
    zero_hamming_dict = {v: {} for v in range(size + 1)}

    for item, n in iterations_dir:
        if optimizer == "MBD":
            try:
                with open(item / "physical_count_res.json", "r") as file:
                    tmp_dict = json.load(file)
                for k, v in tmp_dict.items():
                    hw = count_ones(k)
                    zero_hamming_dict[hw][k] = 1
            except FileNotFoundError as e:
                break
        elif optimizer == "QUEEN":
            try:
                with open(item / "physical_probs_res.json", "r") as file:
                    tmp_dict = json.load(file)
                for k in tmp_dict:
                    hw = sum([1 for c in k if c == 1])
                    zero_hamming_dict[hw][str(k)] = 1
            except FileNotFoundError as e:
                break

    # Data for the overlapping bar plot
    hds = sorted(zero_hamming_dict.keys())
    hw_measured = [sum(zero_hamming_dict[k].values()) for k in hds]
    max_elements = [math.comb(size, k) for k in hds]

    # Create overlapping bar plot
    bar_width = 0.4  # Controls the width of the bars

    colors = ['gray'] * len(hds)
    colors[hd_opt] = 'blue'
    plt.bar(hds, max_elements, width=bar_width, label='Maximum Values',
            alpha=0.5, color=colors)  # Slight transparency
    colors = ['red'] * len(hds)
    colors[hd_best] = 'green'
    plt.bar(hds, hw_measured, width=bar_width, label='Measured States',
            alpha=1, color=colors)  # Overlapping with transparency

    # Add labels and title
    plt.xlabel('Hamming Weight')
    plt.ylabel('Count')
    plt.title('Measured Hamming Weights')
    plt.yscale("symlog", linthresh=10 ** 0)
    # Show legend
    plt.legend()
    plt.savefig(base_dir / f"hamming_weight_{size}_{rep}_{optimizer}{h}.pdf")
    plt.close()


def load_config(config_file: Path) -> dict:
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def plot_runs_results(res_path: Path, max_rep: int = 9):
    for experiment_path in res_path.iterdir():
        if not experiment_path.is_dir() or "gs_" not in experiment_path.name:
            continue
        print(f"Elaborating results in {experiment_path.name}")
        cplex_results, classical_res = get_classical_baseline(experiment_path)
        qubits = int(experiment_path.name.split("_")[1])
        repetition = int(experiment_path.name.split("_")[2])
        if repetition > max_rep:
            continue

        for base_dir in experiment_path.iterdir():
            if base_dir.name.startswith("classical"):
                continue

            res_file = base_dir / "results.json"
            if not res_file.exists():
                print(f"Res file missing in {experiment_path.name}: SKIPPING")
                continue
            config_file = base_dir / "config.json"
            if not config_file.exists():
                print(f"Conf file missing in {experiment_path.name}: SKIPPING")
                continue

            optimal_solution = load_optimal_solution(classical_res)
            config = load_config(config_file)
            results = load_config(res_file)


            plot_hamming_dist(base_dir, config, results,
                              optimal_solution, qubits, repetition)

            gc.collect()


def add_loss_line(selected_data: pd.DataFrame, label: str, mode: str):
    if len(selected_data) == 0:
        return
    avg = []
    std_dev = []
    x = range(300)
    for it in range(300):
        if mode != "payoff":
            values = -selected_data[selected_data['iteration'] == it][
                'payoff_best'] / \
                     selected_data[selected_data['iteration'] == it][
                         'cplex_payoff']
        else:
            values = selected_data[selected_data['iteration'] == it]["payoff"]
        avg.append(values.mean())
        std_dev.append(values.std())

    # Plot average
    plt.plot(x, avg, label=label)
    # Plot standard deviation as shadowed area
    # plt.fill_between(x,
    #                  [avg[i] - std_dev[i] for i in range(300)],
    #                  [avg[i] + std_dev[i] for i in range(300)],
    #                  alpha=0.2
    #                  )


def plot_avg_and_standard_dev(df: pd.DataFrame,
                              base_dir: Path,
                              n_qubits: int,
                              mode: str = "payoff"):
    """
    Plot the average and standard deviation of the objective function value over
    iterations for the specified optimizers and number of qubits.
    """
    for OPT in df['optimizer'].unique():
        for PARAMS in df["minimizer_params"].unique():
            selected_data = df[df['optimizer'] == OPT]
            selected_data = selected_data[selected_data['minimizer_params'] == PARAMS]
            selected_data = selected_data[selected_data['qubits'] == n_qubits]

            if OPT == "MBD":
                selected_data = selected_data[selected_data['raw_counts'] <= 0]
                add_loss_line(selected_data, f"Average {OPT} ({PARAMS})", mode)
                selected_data_h = selected_data[selected_data['raw_counts'] > 0]
                add_loss_line(selected_data_h,
                              f"Average {OPT} HAMMER ({PARAMS})", mode)
            else:
                add_loss_line(selected_data, f"Average {OPT} ({PARAMS})", mode)
    if mode == "payoff":
        plt.ylim(-1000, 1.1)
        plt.yscale("symlog", linthresh=1)
    else:
        plt.ylim(-0.1, 1.1)

    plt.legend()
    plt.title(
        f'Normalized best payoff with {n_qubits} transactions'
    )
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Payoff')

    # save fig
    plt.savefig(base_dir / f'avg_std_{mode}_{n_qubits}.pdf')
    plt.show()


def get_cluster(raw_x: pd.Series, nq: int) -> np.ndarray:
    X = np.array([[float(v) for v in params.split(" ")] for params in raw_x])
    # ms = MeanShift(bandwidth=None)
    # clusters = ms.fit_predict(X)
    # ms = KMeans(n_clusters=nq)
    # clusters = ms.fit_predict(X)
    ms = cluster.SpectralClustering()
    clusters = ms.fit_predict(X)
    return np.reshape(clusters, (1, len(clusters)))