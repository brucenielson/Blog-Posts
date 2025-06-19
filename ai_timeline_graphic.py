import matplotlib.pyplot as plt

# Events with approximate year
events = {
    "Aristotle": -350,
    "Jesus": 1,
    "George Boole": 1854,
    "First AI Conference": 1956,
}

# Sort events by year
sorted_events = sorted(events.items(), key=lambda x: x[1])
labels, years = zip(*sorted_events)

# Setup timeline
fig, ax = plt.subplots(figsize=(12, 2))
ax.hlines(1, min(years) - 100, max(years) + 100, color='black')  # timeline line

# Plot event markers
for label, year in sorted_events:
    ax.plot(year, 1, 'o', color='red')
    ax.text(year, 1.05, f"{label}\n{abs(year)} {'BC' if year < 0 else 'AD'}",
            ha='center', va='bottom', fontsize=9)

# Formatting
ax.set_ylim(0.9, 1.3)
ax.set_xlim(min(years) - 100, max(years) + 100)
ax.axis('off')
plt.title("Proportional Historical Timeline")
plt.tight_layout()
plt.show()
