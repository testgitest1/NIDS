import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import os

# Create directories for model saving and evaluation results
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results/evaluation'):
    os.makedirs('results/evaluation')

print("Loading preprocessed data...")
# Load preprocessed data
X_train = pd.read_csv('results/X_train.csv')
y_train = pd.read_csv('results/y_train.csv').squeeze()
X_val = pd.read_csv('results/X_val.csv')
y_val = pd.read_csv('results/y_val.csv').squeeze()
X_test = pd.read_csv('results/X_test.csv')
y_test = pd.read_csv('results/y_test.csv').squeeze()

print(f"Loaded data with shapes: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# Function to evaluate model performance
def evaluate_model(model, X_data, y_data, model_name, dataset_type="validation"):
    print(f"Evaluating {model_name} on {dataset_type} set...")
    
    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_data)
    inference_time = time.time() - start_time
    avg_inference_time_per_sample = inference_time / len(X_data) * 1000  # ms
    
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
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_data)[:, 1]
    else:
        # For SVM, use decision_function if predict_proba is not available
        if hasattr(model, "decision_function"):
            y_score = model.decision_function(X_data)
        else:
            y_score = y_pred
    
    fpr, tpr, _ = roc_curve(y_data, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_type})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/evaluation/{model_name}_{dataset_type}_confusion_matrix.png')
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
    plt.savefig(f'results/evaluation/{model_name}_{dataset_type}_roc_curve.png')
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
    with open(f'results/evaluation/{model_name}_{dataset_type}_report.txt', 'w') as f:
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

# Dictionary to store results for all models
all_results = []

# 1. Random Forest Model
print("\nTraining Random Forest model...")
# Use fewer estimators and limit max_depth to prevent overfitting
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20, 
    class_weight='balanced', 
    random_state=42, 
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Save the model
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Evaluate Random Forest on validation set
rf_val_results = evaluate_model(rf_model, X_val, y_val, 'Random_Forest', 'validation')
all_results.append(rf_val_results)

# Evaluate on test set
rf_test_results = evaluate_model(rf_model, X_test, y_test, 'Random_Forest', 'test')
all_results.append(rf_test_results)

# 2. XGBoost Model
print("\nTraining XGBoost model...")
# Use fewer estimators and add regularization parameters to prevent overfitting
# Optional: calculate ratio of negative to positive samples
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

# Save the model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Evaluate XGBoost on validation set
xgb_val_results = evaluate_model(xgb_model, X_val, y_val, 'XGBoost', 'validation')
all_results.append(xgb_val_results)

# Evaluate on test set
xgb_test_results = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', 'test')
all_results.append(xgb_test_results)

# 3. SVM Model (with reduced dataset for speed)
print("\nTraining SVM model...")
print("Using a subset of data for SVM due to computational constraints...")

# Use a subset of the data for SVM
sample_size = min(5000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sample = X_train.iloc[indices]
y_train_sample = y_train.iloc[indices]

print(f"Training SVM on {sample_size} samples...")
svm_model = SVC(
    kernel='rbf', 
    probability=True, 
    C=1.0, 
    gamma='scale', 
    class_weight='balanced', 
    random_state=42
)

svm_model.fit(X_train_sample, y_train_sample)

# Save the model
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Evaluate SVM on validation set
svm_val_results = evaluate_model(svm_model, X_val, y_val, 'SVM', 'validation')
all_results.append(svm_val_results)

# Evaluate on test set
svm_test_results = evaluate_model(svm_model, X_test, y_test, 'SVM', 'test')
all_results.append(svm_test_results)

# Compare models
results_df = pd.DataFrame(all_results)
print("\nModel Comparison:")
print(results_df[['model_name', 'dataset', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'false_alarm_rate', 'roc_auc']])

# Filter results for validation set only for plotting
val_results = results_df[results_df['dataset'] == 'validation']

# Plot comparison of metrics for validation set
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'false_alarm_rate']
plt.figure(figsize=(15, 10))

x = np.arange(len(val_results['model_name']))
width = 0.15
multiplier = 0

for metric in metrics:
    plt.bar(x + width * multiplier, val_results[metric], width, label=metric)
    multiplier += 1

plt.title('Model Comparison (Validation Set)')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(x + width * (len(metrics) - 1) / 2, val_results['model_name'])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(metrics))
plt.tight_layout()
plt.savefig('results/evaluation/model_comparison_validation.png')
plt.close()

# Plot comparison of metrics for test set
test_results = results_df[results_df['dataset'] == 'test']
plt.figure(figsize=(15, 10))

x = np.arange(len(test_results['model_name']))
multiplier = 0

for metric in metrics:
    plt.bar(x + width * multiplier, test_results[metric], width, label=metric)
    multiplier += 1

plt.title('Model Comparison (Test Set)')
plt.xlabel('Model')
plt.ylabel('Score')
plt.xticks(x + width * (len(metrics) - 1) / 2, test_results['model_name'])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(metrics))
plt.tight_layout()
plt.savefig('results/evaluation/model_comparison_test.png')
plt.close()

# Inference time comparison
plt.figure(figsize=(10, 6))
plt.bar(val_results['model_name'], val_results['avg_inference_time_ms'])
plt.title('Average Inference Time Comparison')
plt.xlabel('Model')
plt.ylabel('Time (ms per sample)')
plt.savefig('results/evaluation/inference_time_comparison.png')
plt.close()

# Find the best model based on F1 score on test set
best_model_idx = test_results['f1_score'].idxmax()
best_model_name = test_results.loc[best_model_idx, 'model_name']
print(f"\nThe best model based on F1 score (test set) is: {best_model_name}")

# Save the results comparison
results_df.to_csv('results/evaluation/model_comparison.csv', index=False)

print("\nBasic model training and evaluation completed!")
print("All models are saved in the 'models' folder.")
print("Evaluation results are saved in the 'results/evaluation' folder.")