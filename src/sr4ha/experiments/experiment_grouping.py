import polars as pl

import core.group_identificator
import core.processed_data
import core.segmentor
import core.group_identificator

def experiment(config, segmentation_file):
    data_frame = pl.read_csv(
        config["file"], dtypes=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    segmented_data = core.processed_data.SegmentedData.from_file(data_frame, config["target_var"], segmentation_file)

    # Grouping
    group_identificator = core.group_identificator.GroupIdentificator(config)
    grouped_data = group_identificator.group_segments(segmented_data)

    return grouped_data