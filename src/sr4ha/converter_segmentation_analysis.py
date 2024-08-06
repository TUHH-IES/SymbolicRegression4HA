import optuna
from pathlib import Path
from ruamel.yaml import YAML

import experiment_segmentation

def objective(trial: optuna.Trial) -> float:
    config = YAML(typ="safe").load(Path("configs/converter_identification.yaml"))
    config["start_width"] = trial.suggest_int("start_width", 10, 600)
    config["step_width"] = trial.suggest_int("step_width", 1, 600)
    config["segmentation"]["criterion"]["kwargs"]["saturation"] = trial.suggest_float("saturation", 1e-10, 1e-3,log=True)
    config["step_iterations"] = trial.suggest_int("step_iterations", 5, 75)
    config["segmentation"]["kwargs"]["niterations"] = trial.suggest_int("start_iterations", 20, 300)
    config["segmentation"]["kwargs"]["parsimony"] = trial.suggest_float("parsimony", 0.0, 1.0)
    segmented_data = experiment_segmentation.experiment_segmentation(config)
    trial.set_user_attr("segments", segmented_data.segments.write_json())
    trial.set_user_attr("switches", segmented_data.switches)

    return segmented_data.get_segmentation_deviation("results/converter/gt_switches.csv")

if __name__ == "__main__":

    study = optuna.create_study(storage="sqlite:///segmentation_analysis.db", study_name="converter", load_if_exists=True)
    study.optimize(objective)
