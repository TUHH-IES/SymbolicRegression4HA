import optuna
from pathlib import Path
from ruamel.yaml import YAML

import experiment_grouping

def objective(trial: optuna.Trial) -> float:
    config = YAML(typ="safe").load(Path("configs/converter_identification.yaml"))
    config["grouping"]["criterion"]["kwargs"]["factor"] = trial.suggest_float("criterion-factor", 0.1, 10)
    config["segmentation"]["kwargs"]["niterations"] = trial.suggest_int("iterations", 10, 300)
    config["segmentation"]["kwargs"]["parsimony"] = trial.suggest_float("parsimony", 0.0, 1.0)
    grouped_data = experiment_grouping.experiment(config, Path("results/converter/segmentation_results.csv"))
    trial.set_user_attr("groups", grouped_data.to_json())

    return grouped_data.get_mean_loss()

if __name__ == "__main__":

    study = optuna.create_study(storage="sqlite:///grouping_analysis.db", study_name="converter", load_if_exists=True)
    study.optimize(objective)
