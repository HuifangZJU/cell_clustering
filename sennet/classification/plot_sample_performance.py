import matplotlib.pyplot as plt
import numpy as np

# Sample names
samples = [
    "1889-R", "1913-L", "1913-R", "BRPC-23-378-R", "1850-R",
    "1885-L", "1885-R", "2016-L", "BRPC-23-563-L", "1850-L",
    "1904-L", "1904-R", "2021-R", "2022-L", "BRPC-24-001-R",
    "2022-R", "2115-L", "1720-L", "1881-L", "1881-R",
    "2049-L", "BRPC-23-268-L", "1995-R", "2196-R", "BRPC-23-495-R",
    "1812-L", "1851-L", "1851-R", "1921-L"
]

# Accuracies
accuracies = np.array([
    0.6674932,  0.64929672, 0.69095485, 0.72258566, 0.67249548, 0.69130976,
    0.67987344, 0.6282913,  0.68957434, 0.67780498, 0.72919579, 0.70692712,
    0.68167107, 0.72608449, 0.6456629,  0.58499497, 0.66610133, 0.64305902,
    0.57492907, 0.59201311, 0.68237583, 0.57943662, 0.63888682, 0.61036022,
    0.64634481, 0.66729749, 0.69024148, 0.65452279, 0.67327281
])

# Bar plot
plt.figure(figsize=(14, 6))
plt.rcParams.update({'font.size': 14})
bars = plt.bar(samples, accuracies, color="skyblue", edgecolor="black")

# Rotate x labels for readability
plt.xticks(rotation=75, ha="right", fontsize=9)

# Add accuracy values above bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{acc:.2f}", ha="center", va="bottom", fontsize=12)

plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Sample-wise Accuracy")
plt.tight_layout()
plt.show()
