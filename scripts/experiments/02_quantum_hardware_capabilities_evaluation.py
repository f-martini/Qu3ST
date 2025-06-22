import logging
from qu3st.experiments.quantum_hardware_capabilities_evaluation import *
import argparse
from pathlib import Path
import sys

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

experiment_path = root_path / ("modules/experiments/"
                               "quantum_hardware_capabilities_evaluation")

parser = argparse.ArgumentParser()
parser.add_argument('--shmin', type=int, default=10000)
parser.add_argument('--shmax', type=int, default=10000)
parser.add_argument('--sc', type=int, default=128)
parser.add_argument('--ibm', type=str, default="brisbane")
parser.add_argument('--circ', type=str, default="alternating")
parser.add_argument('--lmin', type=int, default=1)
parser.add_argument('--lmax', type=int, default=20)
parser.add_argument('--opt', type=int, default=3)
parser.add_argument('--qmin', type=int, default=20)
parser.add_argument('--qmax', type=int, default=50)
parser.add_argument('--rep', type=int, default=10)

args = parser.parse_args()

# set logger
log_path = experiment_path / "logs"
log_path.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

# initialize results dir
res_path = experiment_path / "results"
qcs_path = res_path / "qcs"
res_path.mkdir(parents=True, exist_ok=True)
qcs_path.mkdir(parents=True, exist_ok=True)

SHOTS = [
    nl for nl in [250, 500, 1000, 2500, 5000, 10000]
    if args.shmin <= nl <= args.shmax
]
N_STATES = args.sc
IBM_BACKENDS = [f"ibm_{args.ibm}"]
CIRCUITS = [
    (args.circ, nl) for nl in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    if args.lmin <= nl <= args.lmax
]
MITIGATION_MODES = ["Raw", "Hammer"]
OPTIMIZATION_LEVELS = [args.opt]
QUBITS = [qb for qb in [20, 25, 30, 35, 40, 45, 50]
          if args.qmin <= qb <= args.qmax]
REPETITION = args.rep
CLEAN_AFTER = False

df = load_dataframe(res_path)

for conf_id in range(REPETITION):
    for backend_name in IBM_BACKENDS:
        for opt in OPTIMIZATION_LEVELS:
            for circuit, layers in CIRCUITS:
                for num_qubits in QUBITS:
                    for shots in SHOTS:

                        SAMPLE = "zero"
                        # raw probs / mitigation run
                        print(
                            f"Computing: "
                            f"Raw_{backend_name}_{opt}_{circuit}_{layers}_"
                            f"{num_qubits}_{conf_id}_{SAMPLE}_{shots}"
                        )
                        info_log = set_logger(log_path)
                        row, position, filled = get_row(
                            df=df, mitigation_mode="Raw",
                            backend_name=backend_name,
                            opt_level=opt, circuit=circuit, layers=layers,
                            num_qubits=num_qubits, conf_id=conf_id,
                            sample_mode='zero', shots=shots, n_states=N_STATES,
                            info_log=info_log)
                        # common qc
                        raw_counts, ground_truth, file_name = get_counts(
                            row, res_path, qcs_path
                        )
                        row["res_file"] = file_name
                        # extract mit related count
                        mit_raw_counts = raw_counts[0]
                        mit_ground_truth = ground_truth[0]
                        mit_normalizer = sum(mit_raw_counts.values())
                        mit_raw_probs = {
                            k: v / mit_normalizer for k, v in
                            mit_raw_counts.items()
                        }
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=mit_raw_probs,
                                ground_truth=mit_ground_truth,
                                res_path=res_path)

                        # sparse probs
                        print(
                            f"Computing: "
                            f"Raw_{backend_name}_{opt}_{circuit}_{layers}_"
                            f"{num_qubits}_{conf_id}_{'sparse'}_{shots}"
                        )
                        info_log = set_logger(log_path)
                        row, position, filled = get_row(
                            df=df, mitigation_mode="Raw",
                            backend_name=backend_name,
                            opt_level=opt, circuit=circuit, layers=layers,
                            num_qubits=num_qubits, conf_id=conf_id,
                            sample_mode="sparse", shots=shots,
                            n_states=N_STATES,
                            info_log=info_log)
                        row["res_file"] = file_name
                        sparse_raw_counts = raw_counts[1]
                        sparse_ground_truth = ground_truth[1]
                        sparse_normalizer = sum(sparse_raw_counts.values())
                        sparse_raw_probs = {
                            k: v / sparse_normalizer
                            for k, v in sparse_raw_counts.items()
                        }
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=sparse_raw_probs,
                                ground_truth=sparse_ground_truth,
                                res_path=res_path)

                        # hammered probs
                        print(
                            f"Computing: "
                            f"Hammer_{backend_name}_{opt}_{circuit}_{layers}_"
                            f"{num_qubits}_{conf_id}_{'zero'}_{shots}"
                        )
                        row, position, filled = get_row(
                            df=df, mitigation_mode="Hammer",
                            backend_name=backend_name, opt_level=opt,
                            circuit=circuit, layers=layers,
                            num_qubits=num_qubits,
                            conf_id=conf_id, sample_mode="zero", shots=shots,
                            n_states=N_STATES, info_log=info_log)
                        mit_hammered_probs = apply_hammer(
                            row=row, res_path=res_path, raw_prob=mit_raw_probs,
                            mit_prob=mit_raw_probs,
                            num_qubits=num_qubits,
                            shots=shots,
                        )
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=mit_hammered_probs,
                                ground_truth=mit_ground_truth,
                                res_path=res_path
                            )

                        # hammered sparse probs
                        print(
                            f"Computing: "
                            f"Hammered_Sparse_{backend_name}_{opt}_{circuit}_"
                            f"{layers}_"
                            f"{num_qubits}_{conf_id}_{'sparse'}_{shots}"
                        )
                        row, position, filled = get_row(
                            df=df, mitigation_mode="Hammer",
                            backend_name=backend_name, opt_level=opt,
                            circuit=circuit, layers=layers,
                            num_qubits=num_qubits,
                            conf_id=conf_id, sample_mode="sparse", shots=shots,
                            n_states=N_STATES, info_log=info_log)
                        sps_hammered_probs = apply_hammer(
                            row=row, res_path=res_path,
                            raw_prob=sparse_raw_probs, mit_prob=mit_raw_probs,
                            num_qubits=num_qubits, shots=shots)
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=sps_hammered_probs,
                                ground_truth=sparse_ground_truth,
                                res_path=res_path
                            )

                        # hammered+Dep probs
                        mitimode = "Hammer+Dep"
                        print(
                            f"Computing: "
                            f"{mitimode}_{backend_name}_{opt}_{circuit}_{layers}_"
                            f"{num_qubits}_{conf_id}_{'zero'}_{shots}"
                        )
                        row, position, filled = get_row(
                            df=df, mitigation_mode=mitimode,
                            backend_name=backend_name, opt_level=opt,
                            circuit=circuit, layers=layers,
                            num_qubits=num_qubits,
                            conf_id=conf_id, sample_mode="zero", shots=shots,
                            n_states=N_STATES, info_log=info_log)
                        hdc_probs = apply_dep_channel_correction(
                            row=row,
                            res_path=res_path,
                            num_qubits=num_qubits,
                            mit_sample="0" * num_qubits,
                            mit_probs=mit_raw_probs,
                            physical_probs=mit_hammered_probs,
                        )
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=mit_hammered_probs,
                                ground_truth=mit_ground_truth,
                                res_path=res_path
                            )

                        # hammered sparse probs
                        print(
                            f"Computing: "
                            f"{mitimode}_Sparse_{backend_name}_{opt}_{circuit}_"
                            f"{layers}_"
                            f"{num_qubits}_{conf_id}_{'sparse'}_{shots}"
                        )
                        row, position, filled = get_row(
                            df=df, mitigation_mode=mitimode,
                            backend_name=backend_name, opt_level=opt,
                            circuit=circuit, layers=layers,
                            num_qubits=num_qubits,
                            conf_id=conf_id, sample_mode="sparse", shots=shots,
                            n_states=N_STATES, info_log=info_log)
                        hdc_probs = apply_dep_channel_correction(
                            row=row,
                            res_path=res_path,
                            num_qubits=num_qubits,
                            mit_sample="0" * num_qubits,
                            mit_probs=mit_raw_probs,
                            physical_probs=sps_hammered_probs,
                        )
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                probs=hdc_probs,
                                ground_truth=sparse_ground_truth,
                                res_path=res_path
                            )

if CLEAN_AFTER:
    # initialize results dir
    res_path = experiment_path / "results"
    tmp_df = load_dataframe(res_path)
    cleaned_df = tmp_df[tmp_df["mitigation_mode"] != "Hammer"]
    cleaned_df.to_csv(res_path / DATAFRAME_NAME)
    # Iterate over all files in the directory
    for file_path in res_path.iterdir():
        # Only process files (skip directories)
        if file_path.is_file() and "Hammer" in file_path.name:
            # Delete the file if the string is found
            file_path.unlink()
            print(f"Deleted: {file_path}")
