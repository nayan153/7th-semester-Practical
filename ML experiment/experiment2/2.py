import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 4, 9, 16, 25]

# Create multiple axes
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# First plot
ax[0].plot(x, y1, marker='o', label='y = 2x')
ax[0].set_title("First Plot")
ax[0].set_xlabel("X Values")
ax[0].set_ylabel("Y Values")
ax[0].grid(True)
ax[0].legend()

# Second plot
ax[1].plot(x, y2, marker='s', label='y = x²')
ax[1].set_title("Second Plot")
ax[1].set_xlabel("X Values")
ax[1].set_ylabel("Y Values")
ax[1].grid(True)
ax[1].legend()

# Adding text
ax[1].text(2, 20, "Sample Text")

plt.tight_layout()

# Save output
plt.savefig("experiment2_output.png")

print("Graph saved successfully!")