import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 11))

# Metrics (4 decimal places)
accuracy = [0.8153, 0.8790, 0.9070, 0.9404, 0.9482, 0.9543, 0.9576, 0.9611, 0.9642, 0.9646]
precision = [0.8909, 0.9092, 0.9264, 0.9469, 0.9525, 0.9573, 0.9596, 0.9628, 0.9656, 0.9657]
recall = [0.8144, 0.8797, 0.9078, 0.9407, 0.9482, 0.9542, 0.9576, 0.9611, 0.9642, 0.9646]  # macro avg
f1_score = [0.7877, 0.8687, 0.9031, 0.9403, 0.9482, 0.9544, 0.9576, 0.9611, 0.9642, 0.9646]


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
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
