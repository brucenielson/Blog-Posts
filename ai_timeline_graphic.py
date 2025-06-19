import matplotlib.pyplot as plt

# Events with approximate year
events = {
    "Aristotle": -350,
    "Jesus": 1,
    "George Boole": 1854,
    "First AI Conference": 1956,
}

# Sort events chronologically
sorted_events = sorted(events.items(), key=lambda x: x[1])
labels, years = zip(*sorted_events)

# Create plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.hlines(0, min(years) - 100, max(years) + 100, color='black')  # main timeline

# Alternate label positions
for i, (label, year) in enumerate(sorted_events):
    ypos = 0.2 if i % 2 == 0 else -0.3
    valign = 'bottom' if ypos > 0 else 'top'
    ax.plot(year, 0, 'o', color='red')
    ax.text(year, ypos, f"{label}\n{abs(year)} {'BC' if year < 0 else 'AD'}",
            ha='center', va=valign, fontsize=9)

# Formatting
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(min(years) - 100, max(years) + 100)
ax.axis('off')
plt.title("Proportional Historical Timeline (Alternating Labels)")
plt.tight_layout()
plt.show()
