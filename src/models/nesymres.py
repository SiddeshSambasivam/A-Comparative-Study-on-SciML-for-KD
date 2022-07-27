import json
import logging
from functools import partial
from collections import namedtuple

import torch
import omegaconf
import numpy as np
from sympy import lambdify

from nesymres.architectures.model import Model
from nesymres.dclasses import FitParams, BFGSParams

from src.models.base import BaseSymbolicModel

logger = logging.getLogger(__name__)

PredictedFunction = namedtuple("PredictedFunction", ["func", "equation"])
PreprocessedInput = namedtuple("PreprocessedInput", ["x_dict", "x", "y"])


class NeSymRes(BaseSymbolicModel):
    def __init__(
        self, checkpoint_path: str, config_path: str, equation_settings_path: str
    ) -> None:
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.equation_settings_path = equation_settings_path

        self.load_config()
        self.load_eq_settings()

        self.bfgs = BFGSParams(
            activated=self.cfg.inference.bfgs.activated,
            n_restarts=self.cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=self.cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=self.cfg.inference.bfgs.normalization_o,
            idx_remove=self.cfg.inference.bfgs.idx_remove,
            normalization_type=self.cfg.inference.bfgs.normalization_type,
            stop_time=self.cfg.inference.bfgs.stop_time,
        )
        logger.log(logging.INFO, f"Loaded BFGSParams: {self.bfgs}")

        self.params_fit = FitParams(
            word2id=self.eq_setting["word2id"],
            id2word={int(k): v for k, v in self.eq_setting["id2word"].items()},
            una_ops=self.eq_setting["una_ops"],
            bin_ops=self.eq_setting["bin_ops"],
            total_variables=list(self.eq_setting["total_variables"]),
            total_coefficients=list(self.eq_setting["total_coefficients"]),
            rewrite_functions=list(self.eq_setting["rewrite_functions"]),
            bfgs=self.bfgs,
            beam_size=self.cfg.inference.beam_size,  # This parameter is a tradeoff between accuracy and fitting time
        )
        logger.log(logging.INFO, f"Loaded FitParams.")

        self._model = self.init_model()
        logger.log(logging.INFO, f"Loaded model.")

        self.predict_fn = None

    def load_config(self) -> None:
        try:
            self.cfg = omegaconf.OmegaConf.load(self.config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file {self.config_path} not found")

    def load_eq_settings(self) -> None:

        try:
            with open(self.equation_settings_path, "r") as f:
                self.eq_setting = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Equation settings file {self.equation_settings_path} not found"
            )

    def init_model(self) -> None:
        model = Model.load_from_checkpoint(
            self.checkpoint_path, cfg=self.cfg.architecture
        )

        model.eval()
        if torch.cuda.is_available():
            model.cuda()

        return model

    def reinitialize_model(self) -> None:
        pass

    def preprocess_x(
        self, x: np.ndarray, scaled: bool = True, return_var_dict: bool = True
    ) -> torch.tensor:
        n_variables = min(3, x.shape[1])
        max_supp = self.cfg.dataset_train.fun_support["max"]
        min_supp = self.cfg.dataset_train.fun_support["min"]
        total_variables = len(self.eq_setting["total_variables"])

        X = torch.hstack(
            (
                torch.from_numpy(x),
                torch.zeros(x.shape[0], total_variables - n_variables),
            )
        )

        X[:, n_variables:] = 0
        try:
            assert X.shape[1] == 3, "X should consist of 3 input variables"
        except AssertionError as e:
            print(X.shape)
            raise e

        if scaled:
            X = X * (max_supp - min_supp) + min_supp

        X_dict = None
        if return_var_dict:
            X_dict = {
                x: X[:, idx].cpu()
                for idx, x in enumerate(self.eq_setting["total_variables"])
            }

        return X, X_dict

    def preprocess_inputs(self, x: np.ndarray, y: np.ndarray) -> PreprocessedInput:

        X, X_dict = self.preprocess_x(x, False)

        Y = torch.from_numpy(y)

        preprocessed_input = PreprocessedInput(x_dict=X_dict, x=X, y=Y)

        return preprocessed_input

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:

        preprocessed_input = self.preprocess_inputs(x, y)

        fitfunc = partial(self._model.fitfunc, cfg_params=self.params_fit)
        output = fitfunc(preprocessed_input.x, preprocessed_input.y)

        func = lambdify(
            ",".join(self.eq_setting["total_variables"]), output["best_bfgs_preds"][0]
        )

        self.pred_fn = PredictedFunction(func, output["best_bfgs_preds"][0])
        self._is_fit = True

    def predict(self, x: np.ndarray) -> np.ndarray:

        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        _, x_dict = self.preprocess_x(x, scaled=False)
        out = self.pred_fn.func(**x_dict)

        return out.cpu().numpy()

    def equation(self) -> str:
        if self._is_fit is False:
            raise ValueError("Model is not fitted yet")

        return self.pred_fn.equation
