import csv
import optuna
from pathlib import Path
from ruamel.yaml import YAML

import experiment_segmentation

#todo put this function somewhere for reusability
def segmentation_distance(segmented_data):
    file = "results/converter/gt_switches.csv"
    with open(file, 'r') as file:
        reader = csv.reader(file)
        ground_truth_switches = [float(row[0]) for row in reader]

    penalty = 0
    penalty += 100*abs(len(ground_truth_switches) - len(segmented_data.switches))
    for i in range(len(segmented_data.switches)):
        closest_value = min(ground_truth_switches, key=lambda x: abs(x - segmented_data.switches[i]))
        penalty += abs(closest_value - segmented_data.switches[i]) / len(segmented_data.switches)

    return penalty

def objective(trial: optuna.Trial) -> float:
    config = YAML(typ="safe").load(Path("configs/converter_identification.yaml"))
    config["start_width"] = trial.suggest_int("start_width", 10, 300)
    config["step_width"] = trial.suggest_int("step_width", 1, config["start_width"])
    config["segmentation"]["criterion"]["kwargs"]["saturation"] = trial.suggest_float("saturation", 1e-10, 1e-3,log=True)
    config["step_iterations"] = trial.suggest_int("step_iterations", 5, 50)
    config["segmentation"]["kwargs"]["niterations"] = trial.suggest_int("start_iterations", 20, 200)
    config["segmentation"]["kwargs"]["parsimony"] = trial.suggest_float("parsimony", 0.0, 1.0)
    segmented_data = experiment_segmentation.experiment_segmentation(config)
    trial.set_user_attr("segments", segmented_data.segments.write_json())
    trial.set_user_attr("switches", segmented_data.switches)

    return segmentation_distance(segmented_data)

if __name__ == "__main__":

    study = optuna.create_study(storage="sqlite:///segmentation_analysis.db", study_name="converter", load_if_exists=True)
    study.optimize(objective)
