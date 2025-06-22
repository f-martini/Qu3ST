import json
from typing import Any

import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
RATIO_DATAFRAME_NAME = "ratio_dataframe.csv"


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


def filter_ratio_df(
        df: pd.DataFrame,
        filter_dict: dict[str, list[Any]] | None = None) -> pd.DataFrame:
    fdf = df.copy()
    if filter_dict is not None:
        for k in filter_dict.keys():
            fdf = filter_column(fdf, k, vals=filter_dict[k])
    return fdf


def get_maxiters(config: dict) -> int:
    maxiters = 0
    if "maxiters" in config.keys():
        maxiters = config["maxiters"]
    elif "opt_options" in config.keys() and "maxiters" in config[
        "opt_options"].keys():
        maxiters = config["opt_options"]["maxiters"]
    return maxiters


def get_sparsity(config: dict) -> float:
    sparsity = 0
    if "opt_options" in config.keys() and "sd" in config[
        "opt_options"].keys():
        sparsity = config["opt_options"]["sd"]
    return sparsity


def get_hammered(config: dict) -> bool:
    try:
        hammer = config["opt_options"]["primitive_options"]["mitigation_options"]["rem"]["hammer"]
    except KeyError:
        return False
    else:
        return len(hammer.keys()) != 0


def get_shots(config: dict) -> int:
    shots = 0
    if "shots" in config.keys():
        shots = config["shots"]
    elif "opt_options" in config.keys() and "shots" in config[
        "opt_options"].keys():
        shots = config["opt_options"]["shots"]
    return shots


def get_minimizer(config: dict) -> str:
    if "opt_options" in config.keys():
        if "minimizer" in config["opt_options"].keys():
            return config["opt_options"]["minimizer"]
    return "-"


def get_minimizer_params(config: dict) -> str:
    minimizer = get_minimizer(config)
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

def load_ratio_dataframe(res_path: Path, load: bool = True) -> pd.DataFrame:
    dataframe_file = res_path / RATIO_DATAFRAME_NAME
    if dataframe_file.exists() and load:
        df = pd.read_csv(dataframe_file)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
    else:
        df = pd.DataFrame(
            columns=[
                "Optimizer",
                "Qubits",
                "Payoff",
                "Ratio",
                "Shots",
                "Iterations",
                "Repetition",
                "Sparsity",
                "Hammered",
                "Minimizer",
                "Params"
            ]
        )

        for experiment_path in res_path.iterdir():
            if (not experiment_path.is_dir() or
                    "gs_" not in experiment_path.name):
                continue
            print(f"Elaborating results in {experiment_path.name}")
            num_qubits = int(
                experiment_path.name.replace(".", "_").split("_")[1])
            rep = int(experiment_path.name.replace(".", "_").split("_")[2])
            for config_dir in experiment_path.iterdir():
                if not config_dir.is_dir():
                    continue
                conf_file = config_dir / "config.json"
                res_file = config_dir / "results.json"
                if not conf_file.exists():
                    print(
                        f"Config file missing for "
                        f"{experiment_path.name}/{config_dir.name}: SKIPPING "
                        f"...")
                    continue
                if not res_file.exists():
                    print(
                        f"Res file missing for "
                        f"{experiment_path.name}/{config_dir.name}: SKIPPING "
                        f"...")
                    continue
                with open(conf_file, "r") as file:
                    config = json.load(file)

                with open(res_file, "r") as file:
                    results = json.load(file)

                new_row = {
                    "Optimizer": config["opt_mode"],
                    "Qubits": num_qubits,
                    "Payoff": float(results["evaluation"]),
                    "Ratio": 0,
                    "Shots": get_shots(config),
                    "Iterations": get_maxiters(config),
                    "Repetition": rep,
                    "Sparsity": get_sparsity(config),
                    "Hammered": get_hammered(config),
                    "Minimizer": get_minimizer(config),
                    "Params": get_minimizer_params(config),
                }
                df.loc[len(df)] = new_row
        df['Max_Payoff'] = df.groupby(['Qubits'])['Payoff'].transform('max')
        df['Ratio'] = df['Payoff'] / df['Max_Payoff']
        df.to_csv(dataframe_file)
    return df
