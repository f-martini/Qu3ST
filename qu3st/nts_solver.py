import json
from datetime import datetime
from pathlib import Path
from typing import Any, Type
from .ntsp.loader import Loader
from .optimizers import *
import logging

logger = logging.getLogger(__name__)


class NTSSolver:

    def __init__(self,
                 mode: str,
                 optimizer: str,
                 res_path: str | Path | None = None,
                 reload: bool = False,
                 **kwargs
                 ):
        """
        Generates or loads a dataset of cash and securities transactions and
        initializes a Night-Time-Settlement problem instance.
        Assumption: each financial actor has only one security account with one
        security position and one currency account (EUR) with one cash balance.
        Args:
            mode: data-loading mode ["json"]
            optimizer: optimization-mode ["classical", "quantum"]
            kwargs: mode-specific parameters
        """
        # optimizer mode
        self.optimizer = optimizer
        # initialize instance field
        self.instance = None
        # initialize data loader
        self.loader = Loader(mode, **kwargs)
        # set reaload flag
        self.reload = reload
        # create results directory
        self.res_path_string = res_path
        self.res_path = None

    def _setup_logger(self):
        if self.res_path is not None:
            logs_path = self.res_path / "logs"
            logs_path.mkdir(exist_ok=True, parents=True)
            time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            info_log = f'{self.res_path.name}_info_{time_string}.log'
            frmt = '%(asctime)s : %(message)s'
            logging.basicConfig(filename=logs_path / info_log,
                                level=logging.INFO,
                                filemode="w",
                                format=frmt,
                                force=True)
            mes = (
                f"Results and logs for this run will be stored in:\n"
                f"\t{self.res_path}\n"
            )
            print(mes)
        else:
            logger.info(
                f"res_path not provided.\n"
                f"\tResults from this run won't be saved.\n"
            )

    def _validate_res_path(self):
        res_path = None
        if self.res_path_string is not None:
            time_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            res_path = (Path(self.res_path_string) /
                        (self.optimizer + '_' + time_string))
            res_path.mkdir(parents=True, exist_ok=True)
        self.res_path = res_path

    def _reload_last_experiment(self, instance, kwargs):
        config_dict = {k: v for k, v in kwargs.items()}
        config_dict["instance"] = instance
        if self.res_path_string is not None:
            base_dir = Path(self.res_path_string)
            directories = [p for p in base_dir.iterdir()
                           if p.is_dir() and self.optimizer in p.name]
            valid_dirs = []
            for directory in directories:
                try:
                    with open(directory / 'config.json', 'r') as file:
                        data = json.load(file)
                    if data == config_dict:
                        valid_dirs.append(directory)
                finally:
                    continue
            if len(valid_dirs) != 0:
                self.res_path = Path(sorted(valid_dirs)[-1])
            else:
                self._validate_res_path()

    def _save_config(self, instance, kwargs):
        # save config
        config_dict = {k: v for k, v in kwargs.items()}
        config_dict["instance"] = instance
        if self.res_path is not None:
            with open(self.res_path / 'config.json', 'w') as file:
                json.dump(config_dict, file)

    def optimize(self,
                 data: str | None = None,
                 instance: Instance | None = None,
                 load: bool = True,
                 **kwargs) -> OptResult:

        # load instance if not provided
        if instance is None:
            if self.instance is None:
                self.instance = self.loader.load(data)
            instance = self.instance
        else:
            data = str(instance)

        if data is None:
            raise ValueError("Data must be provided or loaded.")

        if self.reload:
            self._reload_last_experiment(data, kwargs)
        else:
            self._validate_res_path()
        self._save_config(data, kwargs)
        self._setup_logger()

        if load and self.results_exists():
            return self.load()

        if self.res_path is None:
            raise ValueError("res_path not initialized. "
                             "Please provide a valid path.")

        # select proper optimization mode
        if self.optimizer == 'classical':
            return ClassicalOptimizer(instance, **kwargs).solve(
                res_path=str(self.res_path))
        elif self.optimizer == 'quantum':
            return QuantumOptimizer(instance, **kwargs).solve(
                res_path=self.res_path)
        elif self.optimizer == 'sampler':
            return SamplerOptimizer(instance, **kwargs).solve(
                res_path=str(self.res_path))
        else:
            raise ValueError(f"{self.optimizer} is not a valid optimization "
                             f"method. Please choose from 'classical', "
                             f"'quantum', or 'sampler'.")

    def save(self, opt_result: OptResult, check: bool = True):
        if self.res_path is None:
            raise ValueError("res_path not initialized.")
        if not check or not self.results_exists():
            opt_result.save(self.res_path)

    def load(self) -> OptResult:

        def _val(key: str, typ: Type | None = None) -> Any:
            v = res_dict[key] if key in res_dict.keys() else None
            return v if typ is None or v is None else typ(v)

        def _2list(s: str, typ: Type) -> list:
            _2remove = ["[", "]", "(", ")", "\n"]
            for c in _2remove:
                s = s.replace(c, "")
            return [typ(c) for c in s.split(" ") if c != "" and c is not None]

        if self.res_path is None:
            raise ValueError("res_path not initialized.")
        with open(self.res_path / "results.json", "r") as file:
            res_dict = json.load(file)

        return OptResult(
            model=_val("model"),
            solver=_val("solver"),
            transactions=_2list(_val("transactions"), int),
            collateral=_2list(_val("collateral"), int),
            cb_indicators=_2list(_val("cb_indicators"), int),
            spl_indicators=_2list(_val("spl_indicators"), int),
            evaluation=_val("evaluation", float),
            evaluation_call=_val("evaluation_call", int),
            total_shots=_val("total_shots", int),
            runtime=_val("runtime", float),
            custom_params=_val("custom_params")
        )

    def results_exists(self) -> bool:
        if self.res_path is None:
            return False

        return (self.res_path / "results.json").exists()
