import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import time
import os
import pickle

# Create directories
if not os.path.exists('models/deep_learning'):
    os.makedirs('models/deep_learning')
if not os.path.exists('results/evaluation/deep_learning'):
    os.makedirs('results/evaluation/deep_learning')

print("Loading preprocessed data...")
# Load preprocessed data
X_train = pd.read_csv('results/X_train.csv')
y_train = pd.read_csv('results/y_train.csv').squeeze()
X_val = pd.read_csv('results/X_val.csv')
y_val = pd.read_csv('results/y_val.csv').squeeze()
X_test = pd.read_csv('results/X_test.csv')
y_test = pd.read_csv('results/y_test.csv').squeeze()
# Add after loading data:
X_train.head(10).to_csv('results/preprocessed_sample.csv', index=False)

print(f"Loaded data with shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# Function to evaluate model performance
def evaluate_dl_model(model, X_data, y_data, model_name, dataset_type="validation"):
    print(f"Evaluating {model_name} on {dataset_type} set...")
    
    # Make predictions
    start_time = time.time()
    y_pred_prob = model.predict(X_data, verbose=0)
    inference_time = time.time() - start_time
    avg_inference_time_per_sample = inference_time / len(X_data) * 1000  # ms
    
    # Convert probabilities to class predictions
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_data, y_pred)
    precision = precision_score(y_data, y_pred)
    recall = recall_score(y_data, y_pred)
    f1 = f1_score(y_data, y_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_data, y_pred)
    
    # Calculate true positives, false positives, true negatives, false negatives
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Generate classification report
    report = classification_report(y_data, y_pred)
    
    # Calculate ROC curve and AUC score
    fpr, tpr, _ = roc_curve(y_data, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/evaluation/deep_learning/{model_name}_{dataset_type}_confusion_matrix.png')
    plt.close()
    
    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} ({dataset_type})')
    plt.legend(loc="lower right")
    plt.savefig(f'results/evaluation/deep_learning/{model_name}_{dataset_type}_roc_curve.png')
    plt.close()
    
    # Create a summary of results
    results = {
        'model_name': model_name,
        'dataset': dataset_type,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'false_alarm_rate': false_alarm_rate,
        'roc_auc': roc_auc,
        'avg_inference_time_ms': avg_inference_time_per_sample
    }
    
    # Save detailed report
    with open(f'results/evaluation/deep_learning/{model_name}_{dataset_type}_report.txt', 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_type}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"False Alarm Rate: {false_alarm_rate:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Average Inference Time: {avg_inference_time_per_sample:.4f} ms per sample\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nModel {model_name} evaluation on {dataset_type} set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Average Inference Time: {avg_inference_time_per_sample:.4f} ms per sample")
    
    return results

from tensorflow.keras.callbacks import TensorBoard

# Add before model training:
tensorboard = TensorBoard(log_dir='logs/dnn')

# Add to callbacks

# # Function to create a vanilla DNN
# def create_dnn_model(input_dim):
#     model = Sequential([
#         Input(shape=(input_dim,)),
#         Dense(128, activation='relu'),
#         BatchNormalization(),
#         # Dense(128, activation='relu', input_shape=(input_dim,)),
#         # BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
#         BatchNormalization(),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(
#         optimizer=Adam(learning_rate=0.001),
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return model
# Add more regularization to the model
def create_dnn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),  # Increased dropout
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )
    
    return model



# Function to plot training history
def plot_training_history(history, model_name):
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/evaluation/deep_learning/{model_name}_training_history.png')
    plt.close()

# Store results for all models
all_dl_results = []

# 1. Deep Neural Network (DNN)
print("\nTraining Deep Neural Network (DNN)...")
input_dim = X_train.shape[1]
dnn_model = create_dnn_model(input_dim)

# Create callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath='models/deep_learning/dnn_best_model.keras',
    monitor='val_loss',
    save_best_only=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# # Train the model
# dnn_history = dnn_model.fit(
#     X_train, y_train,
#     epochs=5,
#     batch_size=128,
#     validation_data=(X_val, y_val),
# # Add to callbacks in model.fit():
#     callbacks=[early_stopping, model_checkpoint, lr_scheduler],
#     verbose=1
# )
# Address class imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Update model training
dnn_history = dnn_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint, lr_scheduler],
    class_weight=class_weight_dict,
    verbose=1
)

# Plot training history
plot_training_history(dnn_history, 'DNN')

# Save model summary
with open('results/evaluation/deep_learning/dnn_model_summary.txt', 'w', encoding='utf-8') as f:
    dnn_model.summary(print_fn=lambda x: f.write(x + '\n'))

# Evaluate DNN on validation set
dnn_val_results = evaluate_dl_model(dnn_model, X_val, y_val, 'DNN', 'validation')
all_dl_results.append(dnn_val_results)

# Evaluate on test set
dnn_test_results = evaluate_dl_model(dnn_model, X_test, y_test, 'DNN', 'test')
all_dl_results.append(dnn_test_results)

# Save the DNN model
dnn_model.save('models/deep_learning/dnn_model.keras')

# ---------------------------------
# 2. Deep Q-Network (DQN) inspired architecture
# Note: This is not a true DQN (which is for RL), but a DNN architecture inspired by DQN
print("\nTraining DQN-inspired neural network...")

def create_dqn_model(input_dim):
    inputs = Input(shape=(input_dim,))
    
    # Feature extraction layers
    x = Dense(256, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Value stream
    value_stream = Dense(128, activation='relu')(x)
    value_stream = Dense(64, activation='relu')(value_stream)
    value_stream = Dense(1, activation='linear')(value_stream)
    
    # Advantage stream
    advantage_stream = Dense(128, activation='relu')(x)
    advantage_stream = Dense(64, activation='relu')(advantage_stream)
    advantage_stream = Dense(1, activation='linear')(advantage_stream)
    
    # Combine streams
    combined = tf.keras.layers.Add()([value_stream, advantage_stream])
    output = Dense(1, activation='sigmoid')(combined)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

dqn_model = create_dqn_model(input_dim)

# Train the model
dqn_history = dqn_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath='models/deep_learning/dqn_best_model.keras', monitor='val_loss', save_best_only=True)
    ],
    verbose=1
)

# Plot training history
plot_training_history(dqn_history, 'DQN')

# Save model summary
with open('results/evaluation/deep_learning/dqn_model_summary.txt', 'w', encoding='utf-8') as f:
    dqn_model.summary(print_fn=lambda x: f.write(x + '\n'))

# Evaluate DQN on validation set
dqn_val_results = evaluate_dl_model(dqn_model, X_val, y_val, 'DQN', 'validation')
all_dl_results.append(dqn_val_results)

# Evaluate on test set
dqn_test_results = evaluate_dl_model(dqn_model, X_test, y_test, 'DQN', 'test')
all_dl_results.append(dqn_test_results)

# Save the DQN model
dqn_model.save('models/deep_learning/dqn_model.keras')

# Compare deep learning models
dl_results_df = pd.DataFrame(all_dl_results)
print("\nDeep Learning Model Comparison:")
print(dl_results_df[['model_name', 'dataset', 'accuracy', 'precision', 'recall', 'f1_score', 
                     'specificity', 'false_alarm_rate', 'roc_auc']])

# Save the results comparison
dl_results_df.to_csv('results/evaluation/deep_learning/dl_model_comparison.csv', index=False)

print("\nDeep learning model training and evaluation completed!")
print("Models are saved in the 'models/deep_learning' folder.")
print("Evaluation results are saved in the 'results/evaluation/deep_learning' folder.")