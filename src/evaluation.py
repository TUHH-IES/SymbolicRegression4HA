import csv
import sys
import polars as pl

def calculate_switch_rates(tolerance, path):
    # Read detected switches from switches.csv
    detected_switches = []
    with open(f"{path}/switches.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            detected_switches.append(float(row[0]))  # Assuming switch values are in the first column

    # Read ground truth switches from gt_switches.csv
    ground_truth_switches = []
    file = f"{path}/gt_switches.csv"
    with open(file, 'r') as file:
        reader = csv.reader(file)
        ground_truth_switches = [float(row[0]) for row in reader]

    # Calculate true positive rate
    true_positives = 0
    for gt_switch in ground_truth_switches:
        for detected_switch in detected_switches:
            if abs(gt_switch - detected_switch) <= tolerance:
                true_positives += 1
                break  # Found a match, move to the next ground truth switch

    true_positive_rate = true_positives / len(ground_truth_switches)

    # Calculate false positive rate
    false_positives = len(detected_switches) - true_positives
    false_positive_rate = false_positives / len(detected_switches)

    return true_positive_rate, false_positive_rate

def calculate_mean_loss(path):
    # Read cluster loss from clusters.csv
    file = f"{path}/cluster.csv"
    df = pl.read_csv(file)

    cluster_loss = df['loss']

    # Read cluster segments from cluster_segments.csv
    cluster_segments = []
    with open(f"{path}/cluster_segments.csv", 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Separate series into tuples of values
            window_sizes = []
            for word in row:
                if word == "":
                    continue
                start, end = word.strip("[]").split(",")
                window_sizes.append((int(end) - int(start)))
            cluster_segments.append(window_sizes)

    # Calculate mean loss
    total_samples = sum([sum(segment) for segment in cluster_segments])
    weighted_losses = [loss * sum(segment) for loss, segment in zip(cluster_loss, cluster_segments)]
    mean_loss = sum(weighted_losses) / total_samples

    return mean_loss

if __name__ == "__main__":
    path = sys.argv[1]
    tolerance = float(sys.argv[2])
    
    tp_rate, fp_rate = calculate_switch_rates(tolerance, path)
    mean_loss = calculate_mean_loss(path)
    file = open(f"{path}/evaluation.txt", "w")
    file.write(f"True positive rate: {tp_rate}\n")
    file.write(f"False positive rate: {fp_rate}\n")
    file.write(f"Mean loss: {mean_loss}\n")
    file.close()