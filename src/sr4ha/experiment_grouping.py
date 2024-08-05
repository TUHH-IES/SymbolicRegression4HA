import polars as pl
import time

import core.group_identificator
import core.segmentor
import core.group_identificator

def experiment(config):
    data_frame = pl.read_csv(
        config["file"], dtypes=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    #todo: read results from segmentation

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