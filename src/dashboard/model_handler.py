import streamlit as st

from src.models.gplearn import Gplearn, FUNCTION_SET
from src.models.dsr import DSR
from src.models.aifeynman import AIFeynman
from src.models.nesymres import NeSymRes


def get_model(model_name: str):
    """Loads the model with the given name."""

    if model_name == "gplearn":               
        model = Gplearn(function_set=FUNCTION_SET, tournament_size=10, verbose=0)

    elif model_name == "dsr":
        model = DSR("configs/dsr_config.json")
    
    elif model_name == "dsr-gp":        
        model = DSR("configs/dsr_gp_config.json")        

    elif model_name == "aifeynman":
        model = AIFeynman(
            "add,sub,mul,div,sin,cos,exp,log,sqrt", NN_epochs=1, max_time=60, BF_try_time=5
        )

    elif model_name == "nesymres":
        model = NeSymRes(
            "configs/NeSymRes/100M.ckpt",
            "configs/NeSymRes/config.yaml",
            "configs/NeSymRes/eq_setting.json",
        )        

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
