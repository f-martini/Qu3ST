from qu3st.optimizers import ObjectiveFunction
from qu3st.ntsp.loader import Loader
from pathlib import Path
import json

import numpy as np
import pandas as pd
from typing import Any
from natsort import natsorted
import sys

root_path = Path(__file__).parent.parent.parent.parent
sys.path.append(str(root_path))


def get_minimizer_params(config: dict) -> str:
    minimizer = config["opt_options"]["minimizer"]
    if minimizer == "BAYESIAN":
        fun = "ucb"
        p = "2.576"
        if "fun" in config["opt_options"]["minimizer_options"].keys() \
                and "p" in config["opt_options"]["minimizer_options"].keys():
            fun = config["opt_options"]["minimizer_options"]["fun"]
            p = config["opt_options"]["minimizer_options"]["p"]
        return f"{fun} ({p})"
    else:
        return "-"


def get_params_dict(res_path: Path, files: pd.Series) -> dict:
    params_dict = {}
    for file in files:
        with open(res_path / file / "config.json", "r") as f:
            config = json.load(f)
        params_dict[file] = get_minimizer_params(config)
    return params_dict


def get_job_summary(iter_dir: Path) -> dict:
    job_summary = {
        "q_time": -1,
        "circuit_depth": -1
    }
    job_info_file = iter_dir / "_job_data.json"
    if not job_info_file.exists():
        return job_summary
    with open(job_info_file, "r") as file:
        job_info = json.load(file)

    try:
        if "device_time" in job_info.keys():
            job_summary["q_time"] = float(job_info["device_time"])
    except TypeError:
        pass
    try:
        if ("metrics" in job_info.keys() and
                "circuit_depths" in job_info["metrics"].keys()):
            job_summary["circuit_depth"] = int(
                job_info["metrics"]["circuit_depths"][0]
            )
    except TypeError:
        pass

    return job_summary


def get_classical_baseline(res_path: Path) -> tuple:
    classical_dir = [d for d in res_path.iterdir()
                     if d.name.startswith("classical")][0]
    classical_file = classical_dir / "results.json"
    if not classical_file.exists():
        raise FileNotFoundError(
            f"Missing classical baseline for {res_path.name}!")
    with open(classical_file, "r") as file:
        classical_results = json.load(file)
    return classical_results, classical_dir


def get_samples_summary(res_dir: Path,
                        optimizer: str,
                        hammered: bool,
                        explored_dict: dict,
                        threshold: float,
                        of: ObjectiveFunction) -> tuple:
    samples_summary = {
        "max_sample_probs": 0,
        "max_sample_payoff": None,
        "min_sample_payoff": None,
        "relevant_states": 0,
        "raw_counts": 0,
        "hammered_count": 0,
        "max_hammered_payoff": 0,
        "min_hammered_payoff": 0,
        "valid_hammered_probs": 0,
        "valid_hammered_counts": 0,
        "valid_states": 0,
        "valid_states_probs": 0
    }

    raw_dict_file = res_dir / "physical_count_res.json"
    hammered_probs_file = res_dir / "physical_probs_res.json"

    if optimizer == "QUEEN":
        if hammered_probs_file.exists():
            with open(hammered_probs_file, "r") as file:
                hammered_probs = json.load(file)
            hammered_dict = {}
            for k in hammered_probs:
                key = "".join([str(int(c)) for c in k])
                if str(k) not in hammered_dict.keys():
                    explored_dict[key] = 1
                    hammered_dict[key] = 1
                else:
                    hammered_dict[key] += 1
            normalizer = sum(hammered_dict.values())
            _, infos = of.evaluate(
                {k: v / normalizer for k, v in hammered_dict.items()},
                info=True,
            )
            samples_summary["relevant_states"] = len(hammered_dict.keys())
            samples_summary["hammered_count"] = len(hammered_dict.keys())
            samples_summary["max_sample_probs"] = infos[1][np.argmax(
                -infos[0])]
            samples_summary["max_sample_payoff"] = max(-infos[0])
            samples_summary["min_sample_payoff"] = min(-infos[0])
            samples_summary["valid_states"] = sum(
                [1 for v in infos[0] if -v >= 0]
            )
            samples_summary["valid_states_probs"] = sum(
                [infos[1][n] for n, v in enumerate(infos[0]) if -v >= 0]
            )
            # initialize with raw values
            samples_summary["max_hammered_payoff"] = samples_summary[
                "max_sample_payoff"]
            samples_summary["min_hammered_payoff"] = samples_summary[
                "min_sample_payoff"]
            samples_summary["valid_hammered_probs"] = samples_summary[
                "valid_states_probs"]
            samples_summary["valid_hammered_counts"] = samples_summary[
                "valid_states"]

    elif optimizer == "MBD":

        raw_dict = None
        if hammered and raw_dict_file.exists():
            with open(raw_dict_file, "r") as file:
                raw_dict = json.load(file)
            samples_summary["raw_counts"] = len(raw_dict)
        elif not hammered and hammered_probs_file.exists():
            with open(hammered_probs_file, "r") as file:
                raw_dict = json.load(file)
        if raw_dict is not None:
            normalizer = sum(raw_dict.values())
            _, infos = of.evaluate(
                {k: v / normalizer for k, v in raw_dict.items()},
                info=True,
            )
            for k in raw_dict.keys():
                explored_dict[k] = 1
            samples_summary["max_sample_probs"] = infos[1][np.argmax(
                -infos[0])]
            samples_summary["max_sample_payoff"] = max(-infos[0])
            samples_summary["min_sample_payoff"] = min(-infos[0])
            samples_summary["valid_states"] = sum(
                [1 for v in infos[0] if -v >= 0]
            )
            samples_summary["valid_states_probs"] = sum(
                [infos[1][n] for n, v in enumerate(infos[0]) if -v >= 0]
            )
            # initialize to raw values
            samples_summary["max_hammered_payoff"] = samples_summary[
                "max_sample_payoff"]
            samples_summary["min_hammered_payoff"] = samples_summary[
                "min_sample_payoff"]
            samples_summary["valid_hammered_probs"] = samples_summary[
                "valid_states_probs"]
            samples_summary["valid_hammered_counts"] = samples_summary[
                "valid_states"]

        if hammered_probs_file.exists():
            with open(hammered_probs_file, "r") as file:
                hammered_probs = json.load(file)
            samples_summary["hammered_count"] = len(hammered_probs)
            samples_summary["relevant_states"] = sum(
                [1 for i in hammered_probs.values() if i > threshold])
            # overwrite with tre hammered values
            if raw_dict is not None:
                payoff_dict = {k: -infos[0][n]
                               for n, k in enumerate(raw_dict.keys())
                               if k in hammered_probs.keys()}
                pvals = payoff_dict.values()
                samples_summary["max_hammered_payoff"] = max(pvals)
                samples_summary["min_hammered_payoff"] = min(pvals)
                samples_summary["valid_hammered_counts"] = sum(
                    [1 for v in pvals if v >= 0]
                )
                samples_summary["valid_hammered_probs"] = sum(
                    [
                        hammered_probs[k]
                        for k, v in payoff_dict.items() if v >= 0
                    ]
                )

    return samples_summary


def get_metrics_summary(res_dir: Path) -> dict:
    iter_res_file = res_dir / "iter_results.json"
    if not iter_res_file.exists():
        return None, None
    with open(iter_res_file, "r") as file:
        iter_results = json.load(file)
    best_solution = str(iter_results["best"]["solution"]).replace("\n", "")
    old_params = "".join(f"{v} " for v in iter_results["parameters"][0])[:-1]
    new_params = "".join(f"{v} " for v in iter_results["parameters"][1])[:-1]
    return {
        "best_solution": best_solution,
        "payoff_best": float(iter_results["best"]["of"]),
        "payoff": - float(
            iter_results["ev"]) if "ev" in iter_results.keys() else None,
        "old_params": old_params,
        "new_params": new_params
    }


def filter_column(fc_df: pd.DataFrame, col: str, vals: list) -> pd.DataFrame:
    if vals is None:
        return fc_df
    for v, op in vals:
        if v is None:
            fc_df = fc_df[fc_df[col].isna()]
        else:
            if op == "==":
                fc_df = fc_df[fc_df[col] == v]
            elif op == "!=":
                fc_df = fc_df[fc_df[col] != v]
            elif op == ">=":
                fc_df = fc_df[fc_df[col] >= v]
            elif op == "<=":
                fc_df = fc_df[fc_df[col] <= v]
            elif op == ">":
                fc_df = fc_df[fc_df[col] > v]
            elif op == "<":
                fc_df = fc_df[fc_df[col] < v]
            elif op == "in":
                fc_df = fc_df[fc_df[col].isin(v)]
    return fc_df


def filter_iterations_df(
        df: pd.DataFrame,
        filter_dict: dict[str, list[Any]] | None = None) -> pd.DataFrame:
    fdf = df.copy()
    if filter_dict is not None:
        for k in filter_dict.keys():
            fdf = filter_column(fdf, k, vals=filter_dict[k])
    return fdf


def get_backend(config: dict) -> str:
    if "primitive_options" in config["opt_options"].keys():
        prm_opts = config["opt_options"]["primitive_options"]
        if "backend_options" in prm_opts.keys():
            if "name" in prm_opts["backend_options"].keys():
                return prm_opts["backend_options"]["name"]
    return "-"


def is_hammered(config: dict) -> bool:
    if "opt_options" in config.keys():
        if "primitive_options" in config["opt_options"].keys():
            primitive_options = config["opt_options"]["primitive_options"]
            if "mitigation_options" in primitive_options.keys():
                if "rem" in primitive_options["mitigation_options"].keys():
                    rem = primitive_options["mitigation_options"]["rem"]
                    if "hammer" in rem.keys():
                        return len(rem["hammer"]) > 0
    return False


def check_row(row: pd.Series, reference_row: dict) -> bool:
    for k in reference_row.keys():
        if not pd.isna(row[k]) and row[k] != reference_row[k]:
            return False
        if pd.isna(row[k]) and reference_row[k] is not None:
            return False
    return True


def check_fields(row: pd.Series, fields: list[str]) -> bool:
    for field in fields:
        if row[field] is None or not pd.isna(row[field]):
            return False
    return True


def row_exists(computed: dict, file: str) -> bool:
    return file in computed.keys()


def get_df(res_path: Path) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "file",
        "backend_name",
        "optimizer",
        "shots",
        "circuit",
        "qubits",
        "repetition",
        "iteration",
        "explored_states",
        "relevant_states",
        "max_sample_probs",
        "max_sample_payoff",
        "min_sample_payoff",
        "valid_states",
        "valid_states_probs",
        "max_hammered_payoff",
        "min_hammered_payoff",
        "valid_hammered",
        "valid_hammered_probs",
        "best_solution",
        "payoff_best",
        "payoff",
        "raw_counts",
        "hammered_count",
        "q_time",
        "circuit_depth",
        "cplex_payoff",
        "cplex_solution",
        "minimizer",
        "minimizer_params",
        "old_params",
        "new_params"
    ]
    if (res_path / "iterations_df.csv").exists():
        df = pd.read_csv(res_path / "iterations_df.csv")
        for c in df.columns:
            if "Unnamed" in c:
                df = df.drop(c, axis=1)
    else:
        df = pd.DataFrame(columns=columns)
    return df, columns


def process_results(res_dir: Path) -> pd.DataFrame:
    df, columns = get_df(res_dir)
    data = df.values.tolist()
    computed = {f: True for f in df["file"]}
    exp_paths = [p for p in res_dir.iterdir()]
    for n_exp, experiment_path in enumerate(exp_paths):
        if not experiment_path.is_dir() or "gs_" not in experiment_path.name:
            continue
        print(f"Elaborating results in {experiment_path.name} "
              f"- {n_exp}/{len(exp_paths)}")

        cplex_results, classical_dir = get_classical_baseline(experiment_path)
        updated = 0
        for config_dir in experiment_path.iterdir():
            if config_dir.name.startswith("classical"):
                continue

            conf_file = config_dir / "config.json"
            if not conf_file.exists():
                print(
                    f"Config file missing for {experiment_path.name}: SKIPPING "
                    f"...")
                continue
            with open(conf_file, "r") as file:
                config = json.load(file)
            file = f"{experiment_path.name}/{config_dir.name}"
            if not row_exists(computed, file):
                explored_dict = {}
                of = ObjectiveFunction(
                    Loader("json").load_json(config["instance"]),
                    **config["opt_options"]["of_options"],
                )
                iter_dir_names = natsorted(
                    [d.name for d in config_dir.iterdir()])

                for itt, iter_dir_name in enumerate(iter_dir_names):
                    iter_dir = config_dir / iter_dir_name
                    if not iter_dir.is_dir():
                        continue
                    if not iter_dir.name.startswith("iteration_"):
                        continue
                    samples_summary = get_samples_summary(
                        iter_dir,
                        config["opt_mode"],
                        is_hammered(config),
                        explored_dict,
                        1 / (10 * config["opt_options"]["shots"]),
                        of
                    )
                    metrics_summary = get_metrics_summary(iter_dir)
                    job_summary = get_job_summary(iter_dir)

                    iteration = int(iter_dir.name.split("_")[1])
                    backend_name = get_backend(config)
                    optimizer = config["opt_mode"]
                    shots = config["opt_options"]["shots"]
                    circuit = config["opt_options"]["ansatz"]
                    qubits = int(experiment_path.name.split("_")[1])
                    repetition = int(experiment_path.name.split("_")[2])
                    explored_states = len(explored_dict)
                    relevant_states = samples_summary["relevant_states"]
                    max_sample_probs = samples_summary["max_sample_probs"]
                    max_sample_payoff = samples_summary["max_sample_payoff"]
                    min_sample_payoff = samples_summary["min_sample_payoff"]
                    valid_states = samples_summary["valid_states"]
                    valid_states_probs = samples_summary["valid_states_probs"]
                    max_hammered_payoff = samples_summary["max_hammered_payoff"]
                    min_hammered_payoff = samples_summary["min_hammered_payoff"]
                    valid_hammered = samples_summary["valid_hammered_counts"]
                    valid_hammered_probs = samples_summary["valid_hammered_probs"]
                    best_solution = metrics_summary["best_solution"]
                    payoff_best = metrics_summary["payoff_best"]
                    payoff = metrics_summary["payoff"]
                    raw_counts = samples_summary["raw_counts"]
                    hammered_counts = samples_summary["hammered_count"]
                    q_time = job_summary["q_time"]
                    circuit_depth = job_summary["circuit_depth"]
                    cplex_payoff = cplex_results["evaluation"]
                    cplex_solution = cplex_results["transactions"].replace("\n",
                                                                           "")
                    minimizer = config["opt_options"]["minimizer"]
                    minimizer_params = get_minimizer_params(config)
                    old_params = metrics_summary["old_params"]
                    new_params = metrics_summary["new_params"]

                    row = [
                        file,
                        backend_name,
                        optimizer,
                        shots,
                        circuit,
                        qubits,
                        repetition,
                        iteration,
                        explored_states,
                        relevant_states,
                        max_sample_probs,
                        max_sample_payoff,
                        min_sample_payoff,
                        valid_states,
                        valid_states_probs,
                        max_hammered_payoff,
                        min_hammered_payoff,
                        valid_hammered,
                        valid_hammered_probs,
                        best_solution,
                        payoff_best,
                        payoff,
                        raw_counts,
                        hammered_counts,
                        q_time,
                        circuit_depth,
                        cplex_payoff,
                        cplex_solution,
                        minimizer,
                        minimizer_params,
                        old_params,
                        new_params
                    ]

                    data.append(row)
                    print(f"\r{itt}\t\t", end="")
                    updated += 1
        if updated > 0:
            df = pd.DataFrame(data=data, columns=columns)
            computed = {f: True for f in df["file"]}
            df.to_csv(res_dir / "iterations_df.csv")
    return df


def load_iterations_df(res_path: Path, compute: bool = True) -> pd.DataFrame:
    if not (res_path / "iterations_df.csv").exists() or compute:
        return process_results(res_path)
    else:
        return pd.read_csv(res_path / "iterations_df.csv")


if __name__ == "__main__":
    res_path = Path(__file__).parent / "results"
    process_results(res_path)
