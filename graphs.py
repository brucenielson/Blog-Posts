import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Generate random points on a single line
np.random.seed(44)
x = np.random.uniform(0, 10, 6)  # Fewer random x-values
y = 2 * x + 3

# Add three more random points for graph 3
additional_x_graph3 = np.random.uniform(0, 10, 2)
additional_y_graph3 = np.random.uniform(0, 10, 2)
x_graph3 = np.concatenate([x, additional_x_graph3])
y_graph3 = np.concatenate([y, additional_y_graph3])

additional_x_graph4 = np.random.uniform(0, 10, 2)
additional_y_graph4 = np.random.uniform(0, 10, 2)
x_graph4 = np.concatenate([x, additional_x_graph4])
y_graph4 = np.concatenate([y, additional_y_graph4])

# Sort the points by x-values
sorted_indices_graph3 = np.argsort(x_graph3)
x_graph3_sorted = x_graph3[sorted_indices_graph3]
y_graph3_sorted = y_graph3[sorted_indices_graph3]

sorted_indices_graph4 = np.argsort(x_graph4)
x_graph4_sorted = x_graph4[sorted_indices_graph4]
y_graph4_sorted = y_graph4[sorted_indices_graph4]

# Create cubic spline interpolations
cs_graph3 = CubicSpline(x_graph3_sorted, y_graph3_sorted)
cs_graph4 = CubicSpline(x_graph4_sorted, y_graph4_sorted)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Problem of Induction: Random Points and Curve Fitting")

# Graph 1: Plot random points
axs[0, 0].scatter(x, y, label="Random Points", color="blue")
# axs[0, 0].set_title("Random Points on a Single Line")

# Graph 2: Fit a straight line
coefficients = np.polyfit(x, y, 1)
line_fit = np.poly1d(coefficients)
axs[0, 1].scatter(x, y, label="Random Points", color="blue")
axs[0, 1].plot(x, line_fit(x), color="red", label="Hidden Function 1")
# axs[0, 1].set_title("Line Through the Points")

# Graph 3: Fit a smooth polynomial curve to the points
axs[1, 0].scatter(x, y, label="Random Points", color="blue")
x_smooth = np.linspace(0, 10, 100)  # Adjusted X range
y_smooth = cs_graph3(x_smooth)
axs[1, 0].plot(x_smooth, y_smooth, color="red", label="Hidden Function 2")
# axs[1, 0].set_title("Polyfit Curve Fitting 1")

# Graph 4: Fit a smooth polynomial curve to the points
axs[1, 1].scatter(x, y, label="Random Points", color="blue")
x_smooth = np.linspace(0, 10, 100)  # Adjusted X range
y_smooth = cs_graph4(x_smooth)
axs[1, 1].plot(x_smooth, y_smooth, color="red", label="Hidden Function 3")
# axs[1, 1].set_title("Polyfit Curve Fitting 2")

# Set Y-axis limits for graphs 3 and 4
axs[1, 0].set_ylim(0, 20)
axs[1, 1].set_ylim(0, 20)

# Add legends
for ax_row in axs:
    for ax in ax_row:
        ax.legend()

# Show the plots
plt.tight_layout()
plt.show()
