import argparse
from ruamel.yaml import YAML
from pathlib import Path
import polars as pl

import core.segmentor

from learner.learner import getAccurateSegments
from learner.linear_regressor import LinearRegressor

def main(path):
    modes = []

    config = YAML(typ="safe").load(path)
    data_frame = pl.read_csv(
        config["file"], schema_overrides=[pl.Float64] * len(config["features"])
    )

    if "derivative" in config and config["derivative"]:
        data_frame = data_frame.with_columns(diff=pl.col(config["target_var"]).diff())
        data_frame[0, "diff"] = data_frame["diff"][1]
        config["target_var"] = "diff"

    traces = [data_frame]

    while len(traces) > 0:
        trajectory = traces.pop(0)

        # Segmentation
        segmentor = core.segmentor.Segmentor(config)
        learner = LinearRegressor()
        model = segmentor.segment(data_frame, learner)

        accurate_segments, mode_data = getAccurateSegments(traces, model, config["features"], config["target_var"], config["threshold"])

        model = learner.refineFlowFunction(mode_data, config["features"], config["target_var"])
        modes.append(model)

        # Build remaining traces
        traces = core.segmentor.buildRemainingTraces(traces, accurate_segments)

        #deviation = core.processed_data.get_transition_deviation(segmented_data.switches, config["gt-learn"])

    return modes



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
