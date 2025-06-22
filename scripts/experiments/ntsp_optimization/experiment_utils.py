import json
import os
from typing import Any
import pandas as pd
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from natsort import natsorted

from qu3st.ntsp.instance import Instance
from qu3st.optimizers import OptResult, Optimizer
from qu3st.optimizers.classical.cplex.objective_function import get_weights
import logging

from qu3st.optimizers.quantum.objective_function import ObjectiveFunction

logger = logging.getLogger(__name__)

IT_DATAFRAME_NAME = "iterations.csv"


def print_summary(result):
    logger.info(f'CANDIDATE SOLUTION:')
    logger.info(f"Settled transactions:\n"
                f"{result.transactions}")
    logger.info(f"Collateral Usage:\n"
                f"{result.collateral}")
    logger.info(f"Cash balance indicators:\n"
                f"{result.cb_indicators}")
    logger.info(f"CMB-security-position links indicators:\n"
                f"{result.spl_indicators}")
    logger.info(f"Payoff: {result.evaluation}")
    logger.info(f"Time elapsed: {result.runtime} sec.")


def check_validity(solution: OptResult, model: Any, instance: Instance):
    # retrieve solution values
    transactions = solution.transactions
    collateral = solution.collateral
    cb_indicators = solution.cb_indicators
    spl_indicators = solution.spl_indicators

    # generate new cplex solution according to the retrieved values
    possible_mp_solution = model.new_solution()
    indicators = np.concatenate((transactions, spl_indicators, cb_indicators))

    for i, v in enumerate(model.iter_binary_vars()):
        possible_mp_solution.add_var_value(v, indicators[i])
        # logger.info(f"{v}: {indicators[i]}", end=" ")

    for i, v in enumerate(model.iter_integer_vars()):
        possible_mp_solution.add_var_value(v, collateral[i])
        # logger.info(f"{v}: {collateral[i]}", end=" ")

    # logger.info validity-check summary
    logger.info("Unsatisfied constraints:")
    logger.info(possible_mp_solution.find_unsatisfied_constraints(model))
    logger.info("Is it a valid solution?")
    logger.info(possible_mp_solution.is_valid_solution())
    logger.info("Cplex payoff:")
    weights = get_weights(instance, lam=0.5)
    logger.info(np.sum(weights * solution.transactions))


def print_cplex_info(cplex_solution: OptResult):
    try:
        logger.info(
            f"Solution Status: {cplex_solution.custom_params['details'].status.upper()}"
            f" - Relative Gap:"
            f" {cplex_solution.custom_params['details'].mip_relative_gap}"
        )
    except TypeError:
        logger.info("Missing field...probably due partial results loading.")


def get_classical_config() -> dict:
    return {
        "ntsp": {
            "sanitize": True,
            "reload": True,
            "mode": "json"
        },
        "solver": {
            "lam": 0.5,
            "opt_mode": "CPLEX",
            "opt_options": {
                "verbose": True,
                "collateral": True
            },
        }
    }


def validate_configuration(config: dict, instance_file: str):
    load_dotenv()
    channel = os.getenv('QISKIT_IBM_RUNTIME_CHANNEL')
    instance = os.getenv('QISKIT_IBM_RUNTIME_INSTANCE')
    key = os.getenv('QISKIT_API_KEY')
    # critical data
    critical_channel = "ibm_cloud"
    critical_instance = os.getenv('CRITICAL_INSTANCE')
    critical_key = os.getenv('CRITICAL_KEY')

    using_critical = (channel == critical_channel or
                      instance == critical_instance or
                      key == critical_key
                      )

    bdi_config_file = Path(__file__).parent / "data/bdi_config.json"
    with open(bdi_config_file, "r") as file:
        bdi_config = json.load(file)
    isp_config_20_file = Path(__file__).parent / "data/isp_config_20.json"
    with open(isp_config_20_file, "r") as file:
        isp_config_20 = json.load(file)
    isp_config_20_h_file = Path(__file__).parent / \
        "data/isp_config_20_hammer.json"
    with open(isp_config_20_h_file, "r") as file:
        isp_config_20_h = json.load(file)

    exp_config = {k: v for k, v in config.items()}
    exp_config["instance"] = instance_file
    if not using_critical:
        return
    if using_critical and bdi_config == exp_config:
        return
    if using_critical and isp_config_20 == exp_config:
        return
    if using_critical and isp_config_20_h == exp_config:
        return
    raise ValueError("Critical account credentials detected while "
                     "executing on eagle QPU.")


def get_lambda(val: Any, greater: Any, smaller: Any) -> float:
    if greater - smaller == 0:
        return 0
    return (val - smaller) / (greater - smaller)


def interpolate(df: pd.DataFrame,
                num_qubits: int,
                nq_lambda: float,
                shots: int,
                shots_lambda: float,
                sparse: int,
                sparse_lambda: float) -> float:
    sd = 0
    for n, row in df.iterrows():
        prod_0 = nq_lambda if row[0] > shots else (1 - nq_lambda)
        prod_1 = shots_lambda if row[1] > sparse else (1 - shots_lambda)
        prod_2 = sparse_lambda if row[2] > num_qubits else (1 - sparse_lambda)
        sd += prod_0 * prod_1 * prod_2 * row[3]
    return sd


def get_close_values(df: pd.DataFrame, col: int, val: Any) -> tuple:
    sorted_vals = sorted(df[col].unique())
    for n, v in enumerate(sorted_vals):
        if val < v:
            if n == 0:
                raise ValueError(f"Value {val} in col {col} outside "
                                 f"boundaries.")
            else:
                return sorted_vals[n], sorted_vals[n - 1]
    if sorted_vals[-1] == val:
        return val, val
    raise ValueError(f"Value {val} in col {col} outside boundaries.")


def find_nearest_sd(num_qubits: int, shots: int, sparse: int = 128) -> float:
    fallback = 2
    try:
        csv_path = (Path(__file__).resolve().parent) / "data/sigmas.csv"
        df = pd.read_csv(csv_path, header=None)
        greater_shots, smaller_shots = get_close_values(df, 0, shots)
        df = df[(df[0] == greater_shots) | (df[0] == smaller_shots)]
        greater_counts, smaller_counts = get_close_values(df, 1, sparse)
        df = df[(df[1] == greater_counts) | (df[1] == smaller_counts)]
        greater_nq, smaller_nq = get_close_values(df, 2, num_qubits)
        df = df[(df[2] == greater_nq) | (df[2] == smaller_nq)]
        val = interpolate(
            df,
            num_qubits,
            get_lambda(num_qubits, greater_nq, smaller_nq),
            shots,
            get_lambda(shots, greater_shots, smaller_shots),
            sparse,
            get_lambda(sparse, greater_counts, smaller_counts)
        )
    except ValueError as e:
        print(f"ValueError occurred: {e}")
        print(
            f"Using fallback value: standard deviation will be set to {fallback}")
        return fallback
    return val


def get_quantum_config(
        file_name: str,
        shots: int,
        max_iters: int,
        sparse_count: int,
        hammer: bool,
        backend_name: str | None,
        p: float | None,
        fun: str | None
) -> dict:
    num_qubits = int(file_name.split(".")[0].split("_")[1])
    sd = find_nearest_sd(num_qubits, shots, sparse_count)

    minimizer_options = {
        "maxiter": max_iters,
        "sd": sd,
        "reload": True,
    }
    if p is not None and fun is not None:
        minimizer_options["p"] = p
        minimizer_options["fun"] = fun
    elif p is not None:
        logger.info("p is not None but fun is None: ignoring pair.")
    elif fun is not None:
        logger.info("fun is not None but p is None: ignoring pair.")

    if backend_name is None:
        backend_mode = "local"
    elif "fake" in backend_name:
        backend_mode = "fake"
    else:
        backend_mode = "real"
    if hammer:
        rem_dict = {
            "hammer": {
                "mit_sample_mode": "zero",
                "iterative": True,
                "iter_weights": False,
                "max_hd": None
            }
        }
    else:
        rem_dict = {}
    mit_opt = {
        "dd": {"sequence_type": "XpXm"},
        "rem": rem_dict
    }

    if sd == 0:
        raise ValueError("Standard deviation has been set to 0.")
    return {
        "ntsp": {
            "sanitize": False,
            "reload": True,
            "mode": "json"
        },
        "solver": {
            "opt_mode": "MBD",
            "opt_options": {
                "ansatz": "hardware_efficient",
                "ansatz_options": {
                    "rep": 2
                },
                "backend": backend_mode,
                "primitive": "sampler",
                "primitive_options": {
                    "backend_options": {
                        "name": backend_name
                    },
                    "mitigation_options": mit_opt
                    if backend_mode != "local" else None
                },
                "shots": shots,
                "verbose": False,
                "gamma": 1,
                "minimizer": "BAYESIAN",
                "minimizer_options": minimizer_options,
                "of_options": {
                    "lam": 0.5,
                    "gamma": 1,
                    "activation_function": "relogist",
                }
            }
        }
    }


def get_queen_config(file_name: str,
                     shots: int,
                     max_iters: int,
                     sparse_count: int,
                     p: float | None,
                     fun: str | None) -> dict:
    num_qubits = int(file_name.split(".")[0].split("_")[1])
    sd = find_nearest_sd(num_qubits, shots, sparse_count)
    minimizer_options = {
        "maxiter": max_iters,
        "sd": sd,
    }
    if p is not None and fun is not None:
        minimizer_options["p"] = p
        minimizer_options["fun"] = fun
    elif p is not None:
        logger.info("p is not None but fun is None: ignoring pair.")
    elif fun is not None:
        logger.info("fun is not None but p is None: ignoring pair.")
    return {
        "ntsp": {
            "sanitize": False,
            "mode": "json"
        },
        "solver": {
            "opt_mode": "QUEEN",
            "opt_options": {
                "ansatz": "rotations",
                "ansatz_options": {
                    "start": "random",
                },
                "shots": shots,
                "verbose": False,
                "minimizer": "BAYESIAN",
                "minimizer_options": minimizer_options,
                "of_options": {
                    "lam": 0.5,
                    "gamma": 1,
                    "activation_function": "relogist",
                }
            }
        }
    }


def get_sampler_config(shots: int,
                       max_iters: int) -> dict:
    return {
        "ntsp": {
            "sanitize": False,
            "mode": "json"
        },
        "solver": {
            "opt_mode": "SAMPLER",
            "opt_options": {
                "shots": shots,
                "maxiters": max_iters,
                "verbose": False,
                "of_options": {
                    "lam": 0.5,
                    "gamma": 1,
                    "activation_function": "relogist",
                }
            }
        }
    }
