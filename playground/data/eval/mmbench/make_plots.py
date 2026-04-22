import matplotlib.pyplot as plt

# ----------------------------
# Read and parse results file
# ----------------------------

filename = "results.txt"

groups = {
    "sparsity": [],
    "sparsity_cluster_noamplify": [],
    "sparsity_cluster_amplify1000": []
}

with open(filename, "r") as f:
    for line in f:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            continue
        accuracy = float(parts[3])
        method = parts[4]
        time_sec = int(parts[5].replace("s", ""))

        # Get sparsity percentage (last token after underscore)
        sparsity_value = int(method.split("_")[-1])

        if method.startswith("sparsity_cluster_noamplify"):
            groups["sparsity_cluster_noamplify"].append(
                (sparsity_value, accuracy, time_sec)
            )
        elif method.startswith("sparsity_cluster_amplify1000"):
            groups["sparsity_cluster_amplify1000"].append(
                (sparsity_value, accuracy, time_sec)
            )
        else:
            groups["sparsity"].append(
                (sparsity_value, accuracy, time_sec)
            )

# ----------------------------
# Plotting function
# ----------------------------

def plot_group(data, title):
    # Sort by sparsity
    data = sorted(data, key=lambda x: x[0])

    sparsity = [d[0] for d in data]
    accuracy = [d[1] for d in data]
    time_sec = [d[2] for d in data]

    fig, ax1 = plt.subplots()

    # Accuracy (left axis)
    ax1.plot(sparsity, accuracy, marker='o', color='blue')
    ax1.set_xlabel("Sparsity (%)")
    ax1.set_ylabel("Accuracy")
    #set color of line
    
    

    # Time (right axis)
    ax2 = ax1.twinx()
    ax2.plot(sparsity, time_sec, marker='o', color='orange')
    ax2.set_ylabel("Time (seconds)")

    # Add legend
    ax1.legend(["Accuracy"], loc="lower left")
    ax2.legend(["Time"], loc="upper right")

    # Add title

    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

# ----------------------------
# Generate the three plots
# ----------------------------

plot_group(groups["sparsity"], "Baseline Sparsity")
plot_group(groups["sparsity_cluster_noamplify"], "Cluster Sparsity (No Amplify)")
plot_group(groups["sparsity_cluster_amplify1000"], "Cluster Sparsity (Amplify 1000)")