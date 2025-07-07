import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
import json
import logging

# Create the 'Visualizations' directory if it doesn't exist
os.makedirs("Visualizations", exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Model evaluation started.")

# Load preprocessed test data
X_test = np.load("Code/X_test.npy")
y_test = np.load("Code/y_test.npy")

# Ensure the data is in the correct shape for CNN input
X_test = np.expand_dims(X_test, axis=-1)

# Load the trained model
model = load_model("Models/heart_attack_cnn_model_final.h5")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
logging.info(f"Test Loss: {test_loss}")
logging.info(f"Test Accuracy: {test_accuracy}")

# Predict classes for the test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Abnormal Heartbeat", "Myocardial Infarction", "Normal", "History of MI"],
            yticklabels=["Abnormal Heartbeat", "Myocardial Infarction", "Normal", "History of MI"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("Visualizations/confusion_matrix.png")
plt.show()

# Generate classification report
class_report = classification_report(y_true_classes, y_pred_classes,
                                     target_names=["Abnormal Heartbeat", "Myocardial Infarction", "Normal", "History of MI"])
logging.info("Classification Report:\n" + class_report)

# Save classification report to a file
with open("Visualizations/classification_report.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(class_report)

# Compute F1 Score
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
logging.info(f"F1 Score: {f1}")

# Compute ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
logging.info(f"ROC-AUC Score: {roc_auc}")

# Binarize the labels for multi-class ROC curve
y_test_bin = label_binarize(y_true_classes, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc="lower right")
plt.savefig("Visualizations/roc_curve.png")
plt.show()

# Compute precision-recall curve and average precision for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred[:, i])

# Plot precision-recall curve for each class
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green', 'orange']
for i, color in zip(range(n_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Precision-Recall curve of class {i} (AP = {average_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc="lower left")
plt.savefig("Visualizations/precision_recall_curve.png")
plt.show()

# Load training history
with open("Models/training_history.json", "r") as f:
    history = json.load(f)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig("Visualizations/accuracy_loss_plot.png")
plt.show()

logging.info("Model evaluation completed!")

import numpy as np
X_test = np.load("Code/X_test.npy")
print(f"Test Data Shape: {X_test.shape}")

