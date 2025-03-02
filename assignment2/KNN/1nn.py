import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Dataset
points = np.array(
    [[0, 5], [0, 3], [1, 4], [2, 4], [2, 1], [3, 3], [3, 2], [4, 4], [3, 4], [4, 1]]
)
labels = np.array(
    [
        "Negative",
        "Negative",
        "Negative",
        "Positive",
        "Negative",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Negative",
    ]
)

# Compute Voronoi diagram
vor = Voronoi(points)

# Plot Voronoi diagram
voronoi_plot_2d(vor, show_vertices=False, line_colors="black", line_width=1)

# Color regions based on labels
for point_index, region_index in enumerate(vor.point_region):
    region = vor.regions[region_index]
    if not region:  # Skip empty regions
        continue
    polygon = [vor.vertices[j] for j in region]
    plt.fill(
        *zip(*polygon),
        color="lightblue" if labels[point_index] == "Positive" else "lightcoral",
        alpha=0.5,
    )

# Plot the points
plt.scatter(
    points[:, 0],
    points[:, 1],
    c=["blue" if label == "Positive" else "red" for label in labels],
    edgecolors="black",
)

# Add labels
for i, (x, y) in enumerate(points):
    plt.text(x, y, f"({x},{y})", fontsize=8, ha="right")

# Set plot limits
plt.xlim(-1, 5)
plt.ylim(0, 6)

# Show plot
plt.title("1NN Decision Boundary")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
