from pathlib import Path
from typing import Any

import numpy as np
import json
from qu3st.ntsp.generator.data_saver import beautify_json


class OptResult:

    def __init__(self,
                 model: Any,
                 solver: str | None,
                 transactions: np.ndarray | list | None = None,
                 collateral: np.ndarray | list | None = None,
                 cb_indicators: np.ndarray | list | None = None,
                 spl_indicators: np.ndarray | list | None = None,
                 evaluation: float | None = None,
                 evaluation_call: int | float | None = None,
                 total_shots: int | None = None,
                 runtime: float | None = None,
                 custom_params: dict | None = None):
        # model
        self.model = model
        self.solver = solver

        # solution
        self.transactions = np.array(transactions, dtype=np.int32)
        self.collateral = np.array(collateral, dtype=np.int32)
        self.cb_indicators = np.array(cb_indicators, dtype=np.int32)
        self.spl_indicators = np.array(spl_indicators, dtype=np.int32)

        # KPI
        self.evaluation = evaluation
        self.evaluation_call = evaluation_call  # number of solutions evaluated
        self.total_shots = total_shots
        self.runtime = runtime

        self.custom_params = {} if custom_params is None else custom_params

    def to_dict(self) -> dict:
        return {
            name: str(value) for name, value in vars(self).items()
            if isinstance(value, (list, float, np.ndarray, int, str))
        }

    def save(self, res_path: Path):
        with open(res_path / "results.json", 'w') as f:
            final_json = json.dumps(self.to_dict())
            final_json = beautify_json(final_json)
            f.write(final_json)
