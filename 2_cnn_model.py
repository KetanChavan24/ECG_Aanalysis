import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
import json


# Load preprocessed data
X_train = np.load("Code/X_train.npy")
X_val = np.load("Code/X_val.npy")
y_train = np.load("Code/y_train.npy")
y_val = np.load("Code/y_val.npy")


# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weights = dict(enumerate(class_weights))


# Define the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential()


    # Input layer
    model.add(Input(shape=input_shape))


    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))


    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))


    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))


    # Fourth Convolutional Block
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))


    # Flatten and Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))


    return model


# Define input shape and number of classes
input_shape = X_train.shape[1:]  # (100, 100, 1)
num_classes = y_train.shape[1]    # Number of output classes


# Build the model
model = build_cnn_model(input_shape, num_classes)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("Models/heart_attack_cnn_model.h5", monitor='val_loss', save_best_only=True)


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    class_weight=class_weights
)


# Save the final model
model.save("Models/heart_attack_cnn_model_final.h5")


# Save the training history
with open("Models/training_history.json", "w") as f:
    json.dump(history.history, f)


print("CNN model training completed!")