import numpy as np
import matplotlib.pyplot as plt

# Creating data using NumPy
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

# Creating a simple chart
plt.plot(x, y, marker='o', label='y = 2x')

# Setting plot properties
plt.title("Simple Chart using Matplotlib")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.legend()

# Save the chart
plt.savefig("experiment1_output.png")

print("Graph saved successfully!")