import matplotlib.pyplot as plt

# Line Chart
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]

plt.figure()
plt.plot(x, y, marker='o')
plt.title("Line Chart")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.grid(True)
plt.savefig("line_chart.png")

# Histogram
data = [10, 20, 20, 30, 30, 30, 40, 40, 50]

plt.figure()
plt.hist(data, bins=5)
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.savefig("histogram.png")

# Bar Chart
subjects = ["C", "Python", "Java", "ML"]
marks = [80, 85, 75, 90]

plt.figure()
plt.bar(subjects, marks)
plt.title("Bar Chart")
plt.xlabel("Subjects")
plt.ylabel("Marks")
plt.savefig("bar_chart.png")

# Pie Chart
plt.figure()
plt.pie(marks, labels=subjects, autopct="%1.1f%%")
plt.title("Pie Chart")
plt.savefig("pie_chart.png")

print("All charts saved successfully!")