import json
from pathlib import Path
from typing import Any
import pandas as pd
import logging

from qu3st.experiments.qpu_time_diagnostic import get_qpu_time

logger = logging.getLogger(__name__)
DATAFRAME_NAME = "qpu_time_diagnostic_data.csv"


def check_row(row: pd.Series, reference_row: dict) -> bool:
    for k in reference_row.keys():
        if not pd.isna(row[k]) and row[k] != reference_row[k]:
            return False
        if pd.isna(row[k]) and reference_row[k] is not None:
            return False
    return True


def load_dataframe(res_path: Path) -> pd.DataFrame:
    res_file = res_path / DATAFRAME_NAME
    if res_file.exists():
        df = pd.read_csv(res_file)
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)
    else:
        empty_dict = {
            "backend_name": [],
            "n_states": [],
            "opt_level": [],
            "circuit": [],
            "layers": [],
            "num_qubits": [],
            "conf_id": [],
            "shots": [],
            "info_log": [],
            "res_file": [],
            "qpu_time": [],
        }
        df = pd.DataFrame(empty_dict)
        df.to_csv(res_file)
    return df


def get_row(
        df: pd.DataFrame,
        backend_name: str,
        opt_level: int,
        circuit: str,
        layers: int,
        num_qubits: int,
        conf_id: int,
        shots: int,
        info_log: str) -> tuple[pd.Series, int, bool]:
    partial_row = {
        "backend_name": backend_name,
        "opt_level": opt_level,
        "circuit": circuit,
        "layers": layers,
        "num_qubits": num_qubits,
        "conf_id": conf_id,
        "shots": shots,
    }

    matching_indices = df.loc[
        df.apply(lambda r: check_row(r, partial_row), axis=1)
    ].index

    if len(matching_indices) > 1:
        raise ValueError(f"More then one matching index: {matching_indices}")
    elif len(matching_indices) == 1:
        position = matching_indices[0]
        row = pd.Series(df.loc[position])
        if row["fidelity"] is not None and not pd.isna(row["fidelity"]):
            logger.info(
                f"Row already computed; the old log file can be found at:\n"
                f"\t{row['info_log']}"
            )
            return row, position, True
        else:
            logger.info(
                f"Updating log file; the old log file can be found at:\n"
                f"\t{info_log}"
            )
            row["info_log"] = info_log
            return row, position, False
    else:
        partial_row["info_log"] = info_log
        partial_row["res_file"] = None
        partial_row["qpu_time"] = None
        return pd.Series(partial_row), len(df), False


def filter_column(fc_df: pd.DataFrame, col: str, vals: list) -> pd.DataFrame:
    if vals is None:
        return fc_df
    for v, op in vals:
        if v is None:
            fc_df = fc_df[fc_df[col].isna()]
        else:
            if op == "==":
                fc_df = fc_df[fc_df[col] == v]
            elif op == ">=":
                fc_df = fc_df[fc_df[col] >= v]
            elif op == "<=":
                fc_df = fc_df[fc_df[col] <= v]
            elif op == ">":
                fc_df = fc_df[fc_df[col] > v]
            elif op == "<":
                fc_df = fc_df[fc_df[col] < v]
    return fc_df


def filter_df(
        df: pd.DataFrame,
        backend_name: list[Any] | None = None,
        n_states: list[Any] | None = None,
        opt_level: list[Any] | None = None,
        circuit: list[Any] | None = None,
        layers: list[Any] | None = None,
        num_qubits: list[Any] | None = None,
        conf_id: list[Any] | None = None,
        sample_mode: list[Any] | None = None,
        shots: list[Any] | None = None,
        info_log: list[Any] | None = None,
        res_file: list[Any] | None = None,
        qpu_time: list[Any] | None = None,
) -> pd.DataFrame:
    fdf = df.copy()
    fdf = filter_column(fdf, "backend_name", backend_name)
    fdf = filter_column(fdf, "n_states", n_states)
    fdf = filter_column(fdf, "opt_level", opt_level)
    fdf = filter_column(fdf, "circuit", circuit)
    fdf = filter_column(fdf, "layers", layers)
    fdf = filter_column(fdf, "num_qubits", num_qubits)
    fdf = filter_column(fdf, "conf_id", conf_id)
    fdf = filter_column(fdf, "sample_mode", sample_mode)
    fdf = filter_column(fdf, "shots", shots)
    fdf = filter_column(fdf, "info_log", info_log)
    fdf = filter_column(fdf, "res_file", res_file)
    fdf = filter_column(fdf, "qpu_time", qpu_time)
    return fdf


def update_row(df: pd.DataFrame,
               row: pd.Series,
               position: int,
               res_path: Path) -> None:
    row["qpu_time"] = get_qpu_time(res_path, row["res_file"])
    df.loc[position] = row
    df.to_csv(res_path / DATAFRAME_NAME)
