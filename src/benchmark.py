import os
import time
import logging
import datetime
from typing import List, Type
from dataclasses import dataclass, field

import click
import pandas as pd
from tqdm import tqdm

from src.models.base import BaseSymbolicModel
from src.models.gplearn import Gplearn, FUNCTION_SET
from src.models.dsr import DSR
from src.models.aifeynman import AIFeynman
from src.models.nesymres import NeSymRes
from src.dataset import Dataset, Equation, create_equation, load_equations_dataframe

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    model_name: str
    dataset_path: str
    results_path: str
    hyper_parameter: str


@dataclass
class ExperimentResults:
    model_name: str
    equation: str
    predicted_equation: str = field(init=False)
    number_of_points: int
    hyper_parameter: str
    accuracy: float = field(init=False)
    time: float = field(init=False)  # in seconds


class ExperimentRunner:
    """Runs an experiment with the given model, dataset and config."""

    def __init__(
        self, dataset: Dataset, model: Type[BaseSymbolicModel], config: ExperimentConfig
    ):
        self.dataset = dataset
        self.config = config
        self.logs = []

        self.model = model()

    def run(self):
        """Runs the experiment."""
        logger.info("Running experiment...")
        logging.basicConfig(level=logging.info)
        for i, equation in enumerate(self.dataset):

            # self.model._model = self.model.init_model(
            # *self.model.args[0], **self.model.args[1]
            # )
            # self.model._model = self.model.init_model()

            # self.model._model = self.model.init_model(self.model.config_path)

            logger.info(f"Running experiment for equation {i+1}")

            x, y = equation.x, equation.y
            result = ExperimentResults(
                model_name=self.config.model_name,
                equation=equation.expr,
                number_of_points=equation.number_of_points,
                hyper_parameter=self.config.hyper_parameter,
            )

            start = time.time()
            self.model.fit(x, y)
            end = time.time()

            result.accuracy = self.model.score(x, y)
            result.time = end - start
            result.predicted_equation = self.model.equation()

            # convert result to dict and append to logs
            self.logs.append(result.__dict__)

    def write_results(self):
        """Writes the results to a file."""
        logger.info("Writing results...")

        df = pd.DataFrame(data=self.logs)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_path = os.path.join(
            self.config.results_path,
            f"{self.config.model_name}_noise-{self.dataset.noise}_{timestamp}.xlsx",
        )

        # create result_path if it doesn't exist
        if not os.path.exists(self.config.results_path):
            os.makedirs(self.config.results_path)

        self.write_to_excel(df, results_path)

    def write_to_excel(self, df: pd.DataFrame, path: str):
        """Writes the results to an excel file."""

        writer = pd.ExcelWriter(path, engine="xlsxwriter")
        df.to_excel(writer, sheet_name="Sheet1")

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        format1 = workbook.add_format({"num_format": "0.000"})
        worksheet.set_column("F:G", None, format1)  # Adds formatting to column D

        writer.save()


def get_model(model_name: str):
    """Loads the model with the given name."""
    if model_name == "gplearn":
        return Gplearn(FUNCTION_SET)
    elif model_name == "dsr":
        return DSR()
    elif model_name == "aifeynman":
        return AIFeynman()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def load_config(config_path: str):
    """Loads the config from the given path."""
    with open(config_path) as f:
        config = f.read()
    return config


@click.command()
@click.option(
    "--data-path",
    "-d",
    type=str,
    help="Path to the csv containing the equations",
    required=True,
)
@click.option(
    "--noise",
    "-n",
    type=float,
    help="Gaussian noise to add to the data",
    default=0.0,
)
def main(data_path: str, noise: float) -> None:

    start = time.time()

    equations = []
    eq_df = load_equations_dataframe(data_path)

    for _, row in tqdm(eq_df.iterrows()):
        eq = create_equation(row["eq"], row["support"], row["num_points"])
        equations.append(eq)

    dataset: List[Equation] = Dataset(equations, noise=noise)
    dataset.generate_data()

    end = time.time()
    logger.info(f"Time to load and generate equations: {end - start} seconds")

    # model = Gplearn(function_set=FUNCTION_SET, tournament_size=10, verbose=1)
    # config = ExperimentConfig("gplearn", data_path, "logs/", "Tournament size=10")

    # model = DSR("configs/dsr_config.json")
    # config = ExperimentConfig("dsr", data_path, "logs/", "Epochs=128")

    # model = AIFeynman("add,sub,mul,div,sin,cos,exp,log", NN_epochs=1, max_time=60, BF_try_time=5)
    # config = ExperimentConfig("AIF", data_path, "logs/", "Epochs=10")

    model = NeSymRes(
        "configs/NeSymRes/100M.ckpt",
        "configs/NeSymRes/config.yaml",
        "configs/NeSymRes/eq_setting.json",
    )
    config = ExperimentConfig(
        "NeSymRes", data_path, "logs/", "Beam size=2; BFGS Activated"
    )

    exp = ExperimentRunner(dataset, model, config)

    exp.run()
    exp.write_results()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
