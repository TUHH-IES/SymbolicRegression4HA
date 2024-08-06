import polars as pl
import matplotlib.pyplot as plt
import sympy
import csv

class SegmentedData:
    def __init__(self, data: pl.DataFrame, segments: pl.DataFrame, switches, target_var):
        self.data = data
        self.segments = segments
        self.switches = switches
        self.target_var = target_var

    def from_file(self, data, target_var, path):
        segments = pl.DataFrame.read_csv(path)
        switches = pl.DataFrame.read_csv(path)
        return SegmentedData(data, segments, switches.to_dict().values(), target_var)

    def visualize(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[self.target_var])
        for x in self.switches:
            plt.axvline(x=x, color="red")
        plt.show()

    def write_segments_csv(self, path):
        self.segments.write_csv(path)

    def write_switches_csv(self, path):
        pl.DataFrame(self.switches).write_csv(path)

    def get_segmentation_deviation(self, file, length_penalty=100):
        with open(file, 'r') as file:
            reader = csv.reader(file)
            ground_truth_switches = [float(row[0]) for row in reader]

        deviation = 0
        deviation += length_penalty*abs(len(ground_truth_switches) - len(self.switches))
        matching = [None] * len(ground_truth_switches)
        for i in range(len(ground_truth_switches)):
            matching[i] = min(range(len(self.switches)), key=lambda x: abs(self.switches[x] - ground_truth_switches[i]))
            if i > 0 and matching[i-1] is not None and matching[i] == matching[i-1]:
                new_dist = abs(self.switches[matching[i]] - ground_truth_switches[i])
                pre_dist = abs(self.switches[matching[i-1]] - ground_truth_switches[i])
                old_dist = abs(self.switches[matching[i-1]] - ground_truth_switches[i-1])
                if pre_dist < old_dist or new_dist > old_dist:
                    matching[i] = None
                else:
                    matching[i-1] = None
                    

        print(matching)
        for i in range(len(matching)):
            if matching[i] is not None:
                deviation += abs(self.switches[matching[i]] - ground_truth_switches[i]) / len(self.switches)

        return deviation

    

class GroupedData:
    def __init__(self, data: pl.DataFrame, target_var, groups = []):
        self.data = data
        self.groups = groups
        self.target_var = target_var

    def add_group(self, group):
        group.group_id = len(self.groups)
        self.groups.append(group)

    def create_group(self, data, equation, window, loss, segment_losses):
        group = Group(data, equation, [window], len(self.groups), loss, segment_losses)
        self.groups.append(group)

    def print_groups(self):
        for group in self.groups:
            print("Group", group.group_id)
            print(group.windows)

    def visualize(self):
        cmap = plt.colormaps.get_cmap("hsv")
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[self.target_var])
        for i, group in enumerate(self.groups):
            alpha = 0.8 - i / len(self.groups)
            for window in group.windows:
                plt.axvspan(
                    window[0], window[1], color=cmap(i / len(self.groups)), alpha=alpha
                )
        plt.show()

    def get_mean_loss(self):
        total_length = 0
        mean_loss = 0
        for group in self.groups:
            total_length += len(group.data)
            mean_loss += group.loss * len(group.data)
        return mean_loss / total_length
    
    def write_groups_csv(self, path):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups],
            "loss": [group.loss for group in self.groups],
            "equation": [sympy.sstr(group.equation) for group in self.groups],
        })
        data.write_csv(path)

    def write_windows_csv(self, path):
        data = pl.DataFrame({
            "group_id": [group.group_id for group in self.groups for _ in group.windows],
            "window_start": [window[0] for group in self.groups for window in group.windows],
            "window_end": [window[1] for group in self.groups for window in group.windows],
        })
        data.write_csv(path)

    def to_json(self):
        return {
            "groups": [
                {
                    "group_id": group.group_id,
                    "loss": group.loss,
                    "equation": sympy.sstr(group.equation),
                    "windows": group.windows,
                }
                for group in self.groups
            ]
        }


class Group:
    def __init__(self, data: pl.DataFrame, equation, windows, group_id, loss, segment_losses):
        self.data = data
        self.equation = equation
        self.windows = windows
        self.group_id = group_id
        self.loss = loss
        self.segment_losses = segment_losses

    def append_segment(self, data, equation, window, loss, segment_loss):
        self.data = pl.concat([self.data, data])
        self.equation = equation
        self.windows.append(window)
        self.loss = loss
        self.segment_losses.append(segment_loss)