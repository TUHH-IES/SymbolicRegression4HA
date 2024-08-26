import polars as pl

import core.segmentor

def experiment_segmentation(config):
    data_frame = pl.read_csv(
        config["file"], dtypes=[pl.Float64] * len(config["features"])
    )
    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    # Segmentation
    segmentor = core.segmentor.Segmentor(config)
    segmented_data = segmentor.segment(data_frame)
    return segmented_data