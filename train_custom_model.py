import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------- Configuration ---------------------------

DATA_DIR = 'gesture_data'  # Directory containing your JSON files
MODEL_SAVE_PATH = 'gesture_model.keras'  # Changed from 'gesture_model.h5'
SCALER_SAVE_PATH = 'scaler.save'  # Path to save the scaler
LE_SAVE_PATH = 'label_encoder.save'  # Path to save the label encoder

# ------------------------- Data Loading ------------------------------

X = []
y = []

# Iterate through each JSON file in the data directory
for filename in os.listdir(DATA_DIR):
    if filename.endswith('.json'):
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            landmarks = data['landmarks']
            label = data['label']
            X.append(landmarks)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

# ----------------------- Label Encoding -----------------------------

# Initialize LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Get the list of unique gestures
gesture_classes = le.classes_
print(f"Gesture classes: {gesture_classes}")

# Save the label encoder
joblib.dump(le, LE_SAVE_PATH)
print(f"Label encoder saved to '{LE_SAVE_PATH}'")

# ----------------------- Train-Test Split --------------------------

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Further split training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# --------------------- Feature Scaling -----------------------------

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform
X_train = scaler.fit_transform(X_train)

# Transform validation and testing data
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Scaler saved to '{SCALER_SAVE_PATH}'")

# ------------------------ Model Building ----------------------------

# Define the model
input_shape = X_train.shape[1]  # 63 features
num_classes = len(gesture_classes)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_shape,)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

# ------------------------ Callbacks ---------------------------------

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model checkpoint to save the best model with .keras extension
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_loss',
                             save_best_only=True, verbose=1)

# ------------------------ Model Training ----------------------------

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,               # Set high; early stopping will prevent overfitting
    batch_size=32,            # Common batch size
    validation_data=(X_val, y_val),
    callbacks=[early_stop, checkpoint]
)

# ------------------------ Model Evaluation ---------------------------

# Load the best saved model
best_model = models.load_model(MODEL_SAVE_PATH)
print(f"Best model loaded from '{MODEL_SAVE_PATH}'")

# Evaluate on test data
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# Predict classes for test set
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=gesture_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=gesture_classes, yticklabels=gesture_classes, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
#plt.show()

# ----------------------- Save the Model -------------------------------

# The model has already been saved using ModelCheckpoint as 'gesture_model.keras'
print(f"Model training complete and saved as '{MODEL_SAVE_PATH}'")