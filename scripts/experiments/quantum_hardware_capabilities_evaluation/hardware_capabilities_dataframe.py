from typing import Any
import pandas as pd


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
        mitigation_mode: list[Any] | None = None,
        # fidelity
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
    fdf = filter_column(fdf, "mitigation_mode", mitigation_mode)
    return fdf