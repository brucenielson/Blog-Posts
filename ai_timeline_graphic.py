import matplotlib.pyplot as plt

# Events with approximate year
events = {
    "Aristotle": -250,
    "Jesus": 1,
    "George Boole": 1854,
    "First AI Conference": 1956,
}

# Sort chronologically
sorted_events = sorted(events.items(), key=lambda x: x[1])
labels, years = zip(*sorted_events)

# Set up plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.hlines(0, min(years) - 100, max(years) + 100, color='black', linewidth=1)

# Alternate label positions and draw connectors
for i, (label, year) in enumerate(sorted_events):
    is_top = i % 2 == 0
    label_y = 0.3 if is_top else -0.3
    line_y = 0.05 if is_top else -0.05

    # Draw event marker
    ax.plot(year, 0, 'o', color='red')

    # Draw connecting line
    ax.plot([year, year], [0, line_y], color='gray', linestyle='--', linewidth=1)

    # Add label text
    ax.text(
        year, label_y,
        f"{label}\n{abs(year)} {'BC' if year < 0 else 'AD'}",
        ha='center', va='bottom' if is_top else 'top',
        fontsize=9
    )

# Clean up axes
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(min(years) - 100, max(years) + 100)
ax.axis('off')
plt.title("Proportional Historical Timeline with Connector Lines")
plt.tight_layout()
plt.show()
