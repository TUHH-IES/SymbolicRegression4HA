import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Parameters
num_iterations = 10  # Number of trapezoid iterations
x_min, x_max = 0, 10  # Range of x values

# Generate x values
x = np.linspace(x_min, x_max, 100)
# Repeat x values 10 times
x = np.tile(x, 10)

# Generate y values with trapezoid dependency
y = np.zeros_like(x)
# Generate m values giving the mode of the trapezoid
m = np.zeros_like(x)
segment_length = 10
for i in range(num_iterations):
    rising = (x >= 0) & (x < segment_length / 3)
    constant = (x >= segment_length / 3) & (x < 2 * segment_length / 3)
    falling = (x >= 2 * segment_length / 3) & (x <= segment_length)
    y[rising] = (x[rising]) / (segment_length / 3)
    m[rising] = 0
    y[constant] = 1
    m[constant] = 1
    y[falling] = 1 - (x[falling] - (+ 2 * segment_length / 3)) / (segment_length / 3)
    m[falling] = 2

# Save data to CSV
data = pd.DataFrame({'x': x, 'y': y, 'm': m})
data.to_csv('trapezoid_data.csv', index=False)

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(y, label='Trapezoid Dependency')
plt.title('Trapezoid Dependency of y on x')
plt.xlabel('x')
plt.ylabel('t')
plt.grid()
plt.legend()
plt.show()