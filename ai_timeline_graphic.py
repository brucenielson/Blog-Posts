import matplotlib.pyplot as plt

# Events with approximate year
events = {
    "Aristotle": -350,
    "Chrysippus": -250,
    "William\nof\nOckham": 1323,
    "Peter\nAbelard": 1100,
    "Scientific\nRevolution": 1543,
    "Probability\nTheory": 1654,
    "Enlightenment": 1685,
    "Thomas\nBayes": 1763,
    "George\nBoole": 1854,
    "Gottlob\nFrege": 1879,
    "1st\nAI\nConference": 1956,
}

# Sort events chronologically
sorted_events = sorted(events.items(), key=lambda x: x[1])
labels, years = zip(*sorted_events)

# Create plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.hlines(0, min(years) - 100, max(years) + 100, color='black', linewidth=1)

# Alternate label positions with short connector arrows
for i, (label, year) in enumerate(sorted_events):
    is_top = i % 2 == 0
    label_y = 0.15 if is_top else -0.15
    text_va = 'bottom' if is_top else 'top'

    # Event marker
    ax.plot(year, 0, 'o', color='red')

    # Annotate with connector arrow
    ax.annotate(
        f"{label}\n{abs(year)} {'BC' if year < 0 else 'AD'}",
        xy=(year, 0), xytext=(year, label_y),
        textcoords='data',
        ha='center', va=text_va,
        fontsize=9,
        arrowprops=dict(arrowstyle='-', color='gray', lw=1),
    )

# Tidy up
ax.set_ylim(-0.3, 0.3)
ax.set_xlim(min(years) - 100, max(years) + 100)
ax.axis('off')
plt.title("Early History of Artificial Intelligence", fontsize=20, pad=20)
plt.tight_layout()
plt.show()
