from qu3st.experiments.qpu_time_diagnostic import *
import sys
import pathlib
import argparse
import logging

# add project-root to PYTHONPATH
root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

experiment_path = root_path / ("modules/experiments/"
                               "qpu_time_diagnostic")
# set logger
log_path = experiment_path / "logs"
log_path.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)

# initialize results dir
res_path = experiment_path / "results"
res_path.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--shmin', type=int, default=250)
parser.add_argument('--shmax', type=int, default=10000)
parser.add_argument('--ibm', type=str, default="brisbane")
parser.add_argument('--circ', type=str, default="alternating")
parser.add_argument('--lmin', type=int, default=2)
parser.add_argument('--lmax', type=int, default=2)
parser.add_argument('--opt', type=int, default=3)
parser.add_argument('--qmin', type=int, default=20)
parser.add_argument('--qmax', type=int, default=20)
parser.add_argument('--rep', type=int, default=10)

args = parser.parse_args()
SHOTS = [
    nl for nl in [250, 500, 1000, 2000, 4000, 8000]
    if args.shmin <= nl <= args.shmax
]
IBM_BACKENDS = [f"ibm_{args.ibm}"]
CIRCUITS = [
    (args.circ, nl) for nl in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    if args.lmin <= nl <= args.lmax
]
OPTIMIZATION_LEVELS = [args.opt]
QUBITS = [qb for qb in [20, 25, 30, 35, 40, 45, 50]
          if args.qmin <= qb <= args.qmax]
REPETITION = args.rep

df = load_dataframe(res_path)

for shots in SHOTS:
    for conf_id in range(REPETITION):
        for backend_name in IBM_BACKENDS:
            for opt in OPTIMIZATION_LEVELS:
                for circuit, layers in CIRCUITS:
                    for num_qubits in QUBITS:
                        print(
                            f"Computing: "
                            f"Hammered_Sparse_{backend_name}_{opt}_{circuit}_"
                            f"{layers}_"
                            f"{num_qubits}_{conf_id}_{shots}"
                        )
                        info_log = set_logger(log_path)
                        row, position, filled = get_row(
                            df=df,
                            backend_name=backend_name,
                            opt_level=opt,
                            circuit=circuit,
                            layers=layers,
                            num_qubits=num_qubits,
                            conf_id=conf_id,
                            shots=shots,
                            info_log=info_log
                        )
                        # common qc
                        file_name = get_times(
                            row, res_path
                        )
                        row["res_file"] = file_name
                        if not filled:
                            update_row(
                                df=df, row=row, position=position,
                                res_path=res_path
                            )
