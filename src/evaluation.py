import csv

def calculate_switch_rates(window):
    # Read detected switches from switches.csv
    detected_switches = []
    with open('switches.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            detected_switches.append(float(row[0]))  # Assuming switch values are in the first column

    # Read ground truth switches from gt_switches.csv
    ground_truth_switches = []
    with open('gt_switches.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            ground_truth_switches.append(float(row[0]))  # Assuming switch values are in the first column

    # Calculate true positive rate
    true_positives = 0
    for gt_switch in ground_truth_switches:
        for detected_switch in detected_switches:
            if abs(gt_switch - detected_switch) <= window:
                true_positives += 1
                break  # Found a match, move to the next ground truth switch

    true_positive_rate = true_positives / len(ground_truth_switches)

    # Calculate false positive rate
    false_positives = len(detected_switches) - true_positives
    false_positive_rate = false_positives / len(detected_switches)

    return true_positive_rate, false_positive_rate

def calculate_mean_loss():
    # Read cluster loss from clusters.csv
    cluster_loss = []
    with open('clusters.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cluster_loss.append(float(row[0]))  # Assuming loss values are in the first column

    # Read cluster segments from cluster_segments.csv
    cluster_segments = []
    with open('cluster_segments.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            window_sizes = [float(window) for window in row]  # Assuming window sizes are in each row
            cluster_segments.append(window_sizes)

    # Calculate mean loss
    total_samples = sum([sum(segment) for segment in cluster_segments])
    weighted_losses = [loss * sum(segment) for loss, segment in zip(cluster_loss, cluster_segments)]
    mean_loss = sum(weighted_losses) / total_samples

    return mean_loss

# Example usage with window size of 0.5
window_size = 0.5
tp_rate, fp_rate = calculate_switch_rates(window_size)
print(f"True Positive Rate: {tp_rate}")
print(f"False Positive Rate: {fp_rate}")

mean_loss = calculate_switch_rates()