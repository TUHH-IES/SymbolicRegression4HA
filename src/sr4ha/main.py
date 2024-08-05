import argparse
from ruamel.yaml import YAML
from pathlib import Path
import polars as pl
import time

import core.group_identificator
import core.segmentor
import core.group_identificator

def main(path):
    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], dtypes=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    # Segmentation
    segmentor = core.segmentor.Segmentor(config)
    starttime = time.time()
    segmented_data = segmentor.segment(data_frame)
    endtime = time.time()
    segmented_data.write_segments_csv("segmentation_results.csv")
    segmented_data.write_switches_csv("switches.csv")
    segmented_data.visualize()

    print("Time for segmentation:", endtime - starttime)
    with open("time.txt", "w") as file:
        file.write("Segmentation: " + str(endtime - starttime) + "\n")

    # Grouping
    group_identificator = core.group_identificator.GroupIdentificator(config)
    starttime = time.time()
    grouped_data = group_identificator.group_segments(segmented_data)
    endtime = time.time()
    grouped_data.write_groups_csv("grouping_results.csv")
    grouped_data.write_windows_csv("grouping_windows.csv")

    print("Time for grouping:", endtime - starttime)
    with open("time.txt", "a") as file:
        file.write("Grouping: " + str(endtime - starttime))
    grouped_data.visualize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config file",
    )
    arguments = parser.parse_args()
    main(arguments.config)
