from .ntsp_optimization import *
from qu3st.nts_solver import NTSSolver
import sys
import pathlib
import argparse

# add project-root to PYTHONPATH
root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

parser = argparse.ArgumentParser()
parser.add_argument('--gs', type=str, default="20")
parser.add_argument('--ibm', type=str, default="brisbane")
parser.add_argument('--rep', type=int, default=3)
parser.add_argument('--it', type=int, default=300)
parser.add_argument('--sh', type=int, default=10000)
parser.add_argument('--sc', type=int, default=128)
parser.add_argument('--p', type=float, default=None)
parser.add_argument('--fun', type=str, default=None)
parser.add_argument('--cplex', type=str, default="y")
parser.add_argument('--queen', type=str, default="y")
parser.add_argument('--rand', type=str, default="y")
parser.add_argument('--mbd', type=str, default="y")
parser.add_argument('--hammer', type=str, default="y")

args = parser.parse_args()

if args.gs.isnumeric():
    instance_file_name = f"gs_{args.gs}.json"
else:
    instance_file_name = args.gs
INSTANCES = [
    instance_file_name
]
BACKEND = f"ibm_{args.ibm}"
SHOTS = args.sh
SPARSE_COUNT = args.sc
MAX_ITERS = args.it
REPETITION = args.rep
CPLEX = args.cplex.lower() in ["t", "y", "true"]
MBD = args.mbd.lower() in ["t", "y", "true"]
QUEEN = args.queen.lower() in ["t", "y", "true"]
RANDOM = args.rand.lower() in ["t", "y", "true"]
HAMMER = args.hammer.lower() in ["t", "y", "true"]
P = args.p
FUN = args.fun

for rep in range(REPETITION):
    for instance_file in INSTANCES:
        # EXPERIMENT DIRECTORY
        res_dir_name = instance_file.split('.')[0] + "_" + str(rep)
        res_path = (root_path /
                    f"modules/experiments/ntsp_optimization"
                    f"/results/{res_dir_name}")
        res_path.mkdir(parents=True, exist_ok=True)

        # CPLEX
        if CPLEX:
            cplex_config = get_classical_config()
            cplex_solver = NTSSolver(optimizer="classical", res_path=res_path,
                                     **cplex_config["ntsp"])
            cplex_solution = cplex_solver.optimize(
                data=instance_file, load=False, **cplex_config["solver"]
            )
            print_summary(cplex_solution)
            print_cplex_info(cplex_solution)
            cplex_model = cplex_solution.model.model
            instance = cplex_solution.model.instance
            check_validity(solution=cplex_solution, model=cplex_model,
                           instance=instance)
            cplex_solver.save(cplex_solution)

        # MBD
        if MBD:
            mbd_config = get_quantum_config(
                instance_file,
                SHOTS,
                MAX_ITERS,
                SPARSE_COUNT,
                HAMMER,
                BACKEND,
                P,
                FUN
            )
            validate_configuration(mbd_config["solver"], instance_file)
            mbd_solver = NTSSolver(optimizer="quantum", res_path=res_path,
                                   **mbd_config["ntsp"])
            mbd_result = mbd_solver.optimize(
                data=instance_file, **mbd_config["solver"])
            print_summary(mbd_result)
            if CPLEX:
                check_validity(solution=mbd_result, model=cplex_model,
                               instance=instance)
            mbd_solver.save(mbd_result)

        # QUEEN-SPIRED
        if QUEEN:
            queen_config = get_queen_config(
                instance_file,
                SHOTS,
                MAX_ITERS,
                SPARSE_COUNT,
                P,
                FUN
            )
            queen_solver = NTSSolver(optimizer="quantum", res_path=res_path,
                                     reload=True, **queen_config["ntsp"])
            queen_result = queen_solver.optimize(
                data=instance_file, **queen_config["solver"])
            print_summary(queen_result)
            if CPLEX:
                check_validity(solution=queen_result, model=cplex_model,
                               instance=instance)
            queen_solver.save(queen_result)

        # RANDOM SAMPLING
        if RANDOM:
            sampler_config = get_sampler_config(
                SHOTS,
                MAX_ITERS
            )
            sampler_solver = NTSSolver(
                optimizer="sampler", res_path=res_path,
                reload=True, **sampler_config["ntsp"])
            sampler_result = sampler_solver.optimize(
                data=instance_file, **sampler_config["solver"])
            sampler_solver.save(sampler_result)
            print_summary(sampler_result)
            if CPLEX:
                check_validity(solution=sampler_result, model=cplex_model,
                               instance=instance)
