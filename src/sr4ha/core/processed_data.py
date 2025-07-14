from typing import Dict
import polars as pl
import matplotlib.pyplot as plt
import sympy
import csv

def get_transition_deviation(switches, file, length_penalty=100):
    with open(file, 'r') as file:
        reader = csv.reader(file)
        ground_truth_switches = [float(row[0]) for row in reader]

    print(ground_truth_switches)
    print(switches)
    deviation = 0.0
    deviation += length_penalty*abs(len(ground_truth_switches) - len(switches)) / len(switches)
    matching = [None] * len(ground_truth_switches)
    for i in range(len(ground_truth_switches)):
        matching[i] = min(range(len(switches)), key=lambda x: abs(switches[x] - ground_truth_switches[i]))
        if matching[i] in matching[:i]:
            index = matching[:i].index(matching[i])
            new_dist = abs(switches[matching[i]] - ground_truth_switches[i])
            old_dist = abs(switches[matching[index]] - ground_truth_switches[index])
            if new_dist > old_dist:
                matching[i] = None
            else:
                matching[index] = None
                

    print(matching)
    for i in range(len(matching)):
        if matching[i] is not None:
            deviation += abs(switches[matching[i]] - ground_truth_switches[i]) / len(switches)

    return deviation

class SegmentedData:
    def __init__(self, data: pl.DataFrame, segments: pl.DataFrame, switches, target_var):
        self.data = data
        self.segments = segments
        self.switches = switches
        self.target_var = target_var

    @classmethod
    def from_file(cls, data, target_var, path):
        segments = pl.read_csv(path)
        switches = segments["window_start"].to_list()
        return cls(data, segments, switches, target_var)

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

class Group:
    def __init__(self, data: pl.DataFrame, equation, windows, loss, segment_losses):
        self.data = data
        self.equation = equation
        self.windows = windows
        self.loss = loss
        self.segment_losses = segment_losses

    def append_segment(self, data, equation, window, loss, segment_loss):
        self.data = pl.concat([self.data, data])
        self.equation = equation
        self.windows.append(window)
        self.loss = loss
        self.segment_losses.append(segment_loss)    

class GroupedData:
    def __init__(self, data: pl.DataFrame, target_var: str, groups: Dict[int, Group] = dict(), transitions: Dict[int, int] = dict()):
        self.data: pl.DataFrame = data
        self._groups: Dict[int, Group] = groups
        self.target_var: str = target_var
        self.transitions: Dict[int,int] = transitions
        self._nextID = 0

    def create_group(self, data, equation, window, loss, segment_losses):
        '''
        create a new group with the given data, equation, window, loss and segment losses
        and add it to the list of groups
        add the transition from the previous window to the new group
        '''
        if self._nextID in self._groups:
            print("Error: Group with id", self._nextID, "already exists")
        group = Group(data, equation, [window], loss, segment_losses)
        self._groups[self._nextID] = group
        self.transitions[window[1]] = self._nextID
        self._nextID += 1

    def add_segment(self, group_id, data, equation, window, loss, segment_loss):
        '''
        add a segment to the group with the given group_id
        '''
        if group_id not in self._groups.keys():
            print("Error: Group with id", group_id, "does not exist")
            return
        else:
            self._groups[group_id].append_segment(data, equation, window, loss, segment_loss)
            self.transitions[window[1]] = group_id

    def print_groups(self):
        for group_id, group in self._groups.items():
            print("Group", group_id)
            print(group.windows)

    def visualize(self):
        cmap = plt.colormaps.get_cmap("hsv")
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.data[self.target_var])
        for i, group in enumerate(self._groups.values()):
            alpha = 0.8 - i / len(self._groups)
            for window in group.windows:
                plt.axvspan(
                    window[0], window[1], color=cmap(i / len(self._groups)), alpha=alpha
                )
        plt.show()

    def get_mean_loss(self):
        total_length = 0
        mean_loss = 0
        for group in self._groups.values():
            total_length += len(group.data)
            mean_loss += group.loss * len(group.data)
        return mean_loss / total_length
    
    def write_groups_csv(self, path):
        data = pl.DataFrame({
            "group_id": self._groups.keys(),
            "loss": [group.loss for group in self._groups.values()],
            "equation": [sympy.sstr(group.equation) for group in self._groups.values()],
        })
        data.write_csv(path)

    def write_windows_csv(self, path):
        data = pl.DataFrame({
            "group_id": [group_id for group_id, group in self._groups.items() for window in group.windows],
            "window_start": [window[0] for group in self._groups.values() for window in group.windows],
            "window_end": [window[1] for group in self._groups.values() for window in group.windows],
        })
        data.write_csv(path)

    def to_json(self):
        return {
            "groups": [
                {
                    "group_id": group_id,
                    "loss": group.loss,
                    "equation": sympy.sstr(group.equation),
                    "windows": group.windows,
                }
                for group_id, group in self._groups.items()
            ]
        }
    
    @classmethod
    def from_file(cls, data, window_path, result_path, target_var):
        windows = pl.read_csv(window_path)
        raw_groups = pl.read_csv(result_path)
        groups = {}
        transitions = {}
        for i in range(len(raw_groups)):
            new_group = Group(pl.DataFrame(), sympy.parsing.sympy_parser.parse_expr(raw_groups["equation"][i]), [], raw_groups["loss"][i], [])
            for j in range(len(windows)):
                if windows["group_id"][j] == raw_groups["group_id"][i]:
                    new_group.windows.append((windows["window_start"][j], windows["window_end"][j]))
                    new_group.data.vstack(data.slice(windows["window_start"][j], windows["window_end"][j]-windows["window_start"][j]+1))
                    transitions[windows["window_end"][j]] = raw_groups["group_id"][i]
            groups[raw_groups["group_id"][i]] = new_group
        return cls(data, target_var, groups, transitions)