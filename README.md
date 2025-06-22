# ⚠️ **IMPORTANT NOTICE** ⚠️

> **Due to the IBM Quantum platform migration, this codebase is currently not runnable as-is. The code must be ported to the new IBM platform APIs and infrastructure before experiments or scripts can be executed.**
>
> _If you are interested in using this project, please be aware that significant updates are required for compatibility with the latest IBM Quantum services._

# Qu3ST



## Installation

### Create the project root directory

- [ ] _**Unpack the .zip file**_
Copy the project archive `Qu3ST-main.zip` into a directory `[ROOT]` of choice.
```bash
mv  [ZIP-LOCATION]/Qu3ST-main.zip [ROOT]/Qu3ST-main.zip
```
Move in `[ROOT]` directory and unpack `Qu3ST-main.zip` content.
```bash
cd [ROOT]
# bash
sudo apt-get install unzip
unzip Qu3ST-main.zip
```
The final project directory structure should look like this:
```
[ROOT]
└── Qu3ST-main
    ├── docs
    ├── modules
    ├── scripts
    ├── tests
    ├── .github
    ├── config files ...
    ...
```

### Setting-up python environment


- [ ] _**Install Python >=3.10.10:**_
[Download](https://www.python.org/downloads/release/python-31010/)
and install Python >=3.10.10 for your OS.

- [ ] _**Set up virtual environment:**_

Initialize the virtual environment in the `[ROOT]/Qu3ST-main` directory,
and activate it.
```bash
cd [ROOT]/Qu3ST-main
python3.10 -m venv ./venv
# bash
source venv/bin/activate
# powershell
# source venv/Scripts/activate
```
Install the project dependencies:
```bash
pip install -r requirements.txt
```

### Install CPLEX _[Optional]_

A free license for IBM ILOG CPLEX Optimization Studio (and CPLEX optimization
algorithms) is provided to academics upon request
(follow [this link](https://www.ibm.com/products/ilog-cplex-optimization-studio#Educational+resources)).

Alternatively, it is possible to install a trial version of CPLEX by executing the
following commands.
```bash
cd [ROOT]/Qu3ST-main
source venv/bin/activate
pip install cplex==22.1.1.2
```
The trial version comes with limitations in the number of decision variables and
constraints that can be instantiated. Nonetheless, it can be used to solve
small probelm instances and test the installation.


### Secrets configuration

Secrets are stored in a `.env` file (not versioned). The `.env_scheme` file contains the
secrets used in the project (uppercase text). Each secret shall be properly assigned
for the main scripts to run.
```bash
QISKIT_API_KEY = [api-key]
QISKIT_IBM_RUNTIME_CHANNEL = ["ibm_cloud" | "ibm_quantum"]
QISKIT_IBM_RUNTIME_INSTANCE = [ibm-cloud-instance | "ibm-q/open/main"]
CRITICAL_KEY = [critical-api-key]
CRITICAL_INSTANCE = [critical-instance]
```

After filling it, it shall be renamed as `.env`.
```bash
cd [ROOT]/Qu3ST-main
cp ./.env_scheme ./.env
```


## Run the experiments

### NTSP optimization

This experiment allows to run the classical and quantum optimization algorithms.
Script parameters:
- _**gs**_: NTSP instance file name;
- _**ibm**_: ibm backend;
- _**rep**_: experiment repetitions;
- _**it**_: number of iterations of the VQA;
- _**sh**_: number of shots;
- _**sc**_: sparsity-control parameter (integer in $\geq 1$);
- _**fun**_: bayesian optimizer acquisition function;
- _**p**_: bayesian optimizer exploration/exploitation parameter;
- _**cplex**_: enable CPLEX solver;
- _**queen**_: enable QUEEN solver;
- _**rand**_: enable SAMPLER solver;
- _**mbd**_: enable QTSA solver;
- _**hammer**_: activate iHAMMER mitigation technique.

```bash
cd [ROOT]/Qu3ST-main
cp ./.env_scheme ./.env
python ./modules/experiments/01_ntsp_optimization.py
  --gs [file-name]
  --ibm [ibm-device]
  --rep [int]
  --it [int]
  --sh [int]
  --sc [int]
  --fun ["ei","ucb"]
  --p [float]
  --cplex ["t", "f"]
  --queen ["t", "f"]
  --rand ["t", "f"]
  --mbd ["t", "f"]
  --hammer ["t", "f"]
```


## References

<a name="1">__[1]__</a> Alekseeva, E., Ghariani, S., & Wolters, N. (2020).
__Securities and Cash Settlement Framework__. In Mathematical Optimization
Theory and Operations Research: 19th International Conference, MOTOR 2020,
Novosibirsk, Russia, July 6–10, 2020, Proceedings 19 (pp. 391-405).
Springer International Publishing.
[>](https://link.springer.com/chapter/10.1007/978-3-030-49988-4_27)

<a name="2">__[2]__</a> European Central Bank, __Target2-Securities User
Detailed Functional Specification__, v7.0, March 16, 2023, Official
T2S Documentation
[>](https://www.ecb.europa.eu/pub/pdf/annex/T2S_UDFS_V7.0_revised_20220316.en.pdf)

<a name="3">__[3]__</a> European Central Bank, __T2S-0763-SYS T2S Multi-Criteria
Settlement Optimisation__, May 11, 2021, T2S Change Request
[>](https://www.ecb.europa.eu/paym/target/t2s/governance/pdf/crg/ecb.targetseccrg210621_T2S-0763-SYS.en.pdf?d053cd180f9a2efecbdb9d94159e41dd)

<a name="4">__[4]__</a> European Central Bank, __T2S NTS Algorithms Objectives__
, v1.1, December 28, 2018, Official T2S Documentation
[>](https://www.ecb.europa.eu/paym/target/t2s/profuse/shared/pdf/T2S_NTS_algorithms_objectives_V1.1.pdf?3980f7a1a91f882bf5ce2672be0792d7)