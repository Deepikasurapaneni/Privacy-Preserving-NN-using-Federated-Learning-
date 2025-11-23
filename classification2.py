import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

accuracy = [0.7361, 0.7681, 0.7819, 0.7991, 0.7997, 0.8049, 0.8098, 0.8120, 0.8171, 0.8216]
precision = [0.7551, 0.7740, 0.7858, 0.7956, 0.7974, 0.8075, 0.8105, 0.8143, 0.8193, 0.8229]
recall = [0.7361, 0.7681, 0.7819, 0.7991, 0.7997, 0.8049, 0.8098, 0.8120, 0.8171, 0.8216]
f1_score = [0.7266, 0.7595, 0.7761, 0.7925, 0.7920, 0.8011, 0.8055, 0.8062, 0.8125, 0.8167]


# Plot
plt.figure(figsize=(12, 7))

plt.plot(epochs, accuracy, marker='o', label='Accuracy', color='blue')
plt.plot(epochs, precision, marker='s', label='Precision', color='green')
plt.plot(epochs, recall, marker='^', label='Recall', color='orange')
plt.plot(epochs, f1_score, marker='d', label='F1 Score', color='purple')

plt.title('Model Metrics per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.xticks(epochs)
plt.ylim(0.7, 0.85)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
