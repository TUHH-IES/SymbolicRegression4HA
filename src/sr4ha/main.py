import argparse
from ruamel.yaml import YAML
from pathlib import Path
import polars as pl
import time

import core.group_identificator
import core.segmentor
import core.group_identificator
import core.model_extractor

import core.processed_data

def main(path):
    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], schema_overrides=[pl.Float64] * len(config["features"])
    )
    eval_data = pl.read_csv(
        config["eval"], schema_overrides=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        eval_data = eval_data.with_columns(diff=pl.col(config["target_var"]).diff())
        eval_data[0, "diff"] = eval_data["diff"][1]
        config["target_var"] = "diff"

    # Segmentation
    segmentor = core.segmentor.Segmentor(config)
    starttime = time.time()
    segmented_data = segmentor.segment(data_frame)
    endtime = time.time()
    segmented_data.write_segments_csv("segmentation_results.csv")
    segmented_data.write_switches_csv("switches.csv")
    deviation = core.processed_data.get_transition_deviation(segmented_data.switches, config["gt-learn"])
    with open("global_results.txt", "w") as file:
        file.write("Segmentation Deviation: " + str(deviation) + "\n")
    segmented_data.visualize()

    print("Time for segmentation:", endtime - starttime)
    with open("global_results.txt", "a") as file:
        file.write("Segmentation Time: " + str(endtime - starttime) + "\n")

    # Grouping
    group_identificator = core.group_identificator.GroupIdentificator(config)
    starttime = time.time()
    grouped_data = group_identificator.group_segments(segmented_data)
    endtime = time.time()
    grouped_data.write_groups_csv("grouping_results.csv")
    grouped_data.write_windows_csv("grouping_windows.csv")

    print("Time for grouping:", endtime - starttime)
    with open("global_results.txt", "a") as file:
        file.write("Grouping Time: " + str(endtime - starttime) + "\n")

    group_deviation = grouped_data.get_mean_loss() * (1 + abs(len(grouped_data._groups) - config["gt-groups"]))
    with open("global_results.txt", "a") as file:
        file.write("Grouping Deviation: " + str(group_deviation) + "\n")
    print("Grouping Deviation:", group_deviation)
    grouped_data.visualize()

    # Model Construction
    model_extractor = core.model_extractor.ModelExtractor(config)
    starttime = time.time()
    model = model_extractor.createDecisionTreeModel(grouped_data)
    endtime = time.time()
    print("Time for extraction:", endtime - starttime)
    with open("global_results.txt", "a") as file:
        file.write("Extraction Time: " + str(endtime - starttime) + "\n")

    starttime = time.time()
    error, transitions = model_extractor.evaluateDecisionTreeModel(model, eval_data)
    endtime = time.time()
    print("Time for prediction:", endtime - starttime)
    with open("global_results.txt", "a") as file:
        file.write("Evaluation Time: " + str(endtime - starttime) + "\n")
    print("Mean Squared Error:", error)
    deviation = core.processed_data.get_transition_deviation(transitions, config["gt-eval"])
    with open("global_results.txt", "a") as file:
        file.write("Mean Squared Error: " + str(error) + "\n")
        file.write("Transition Deviation: " + str(deviation) + "\n")

def test_extraction(path):
    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], schema_overrides=[pl.Float64] * len(config["features"])
    )
    eval_data = pl.read_csv(
        config["eval"], schema_overrides=[pl.Float64] * len(config["features"])
    )
    
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        eval_data = eval_data.with_columns(diff=pl.col(config["target_var"]).diff())
        eval_data[0, "diff"] = eval_data["diff"][1]
        config["target_var"] = "diff"
    grouped_results = core.processed_data.GroupedData.from_file(data_frame, "grouping_windows.csv", "grouping_results.csv", config["target_var"])

    model_extractor = core.model_extractor.ModelExtractor(config)
    model = model_extractor.createDecisionTreeModel(grouped_results)
    error, transitions = model_extractor.evaluateDecisionTreeModel(model, eval_data)
    print("Error:", error)
    print("Transitions:", transitions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    arguments = parser.parse_args()
    #test_extraction(arguments.config)
    main(arguments.config)
