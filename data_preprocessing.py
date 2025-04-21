import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# Create results directory
if not os.path.exists('results'):
    os.makedirs('results')

# Column names for NSL-KDD dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

print("Loading datasets...")
# Load the datasets
train_df = pd.read_csv('data/KDDTrain+.txt', header=None, names=column_names)
test_df = pd.read_csv('data/KDDTest+.txt', header=None, names=column_names)

# Drop the 'difficulty' column as it's not needed for training
train_df = train_df.drop('difficulty', axis=1)
test_df = test_df.drop('difficulty', axis=1)

print("Datasets loaded with shapes:")
print(f"Training data: {train_df.shape}")
print(f"Test data: {test_df.shape}")

# Data exploration
print("\nChecking for missing values...")
print(f"Training data missing values: {train_df.isnull().sum().sum()}")
print(f"Test data missing values: {test_df.isnull().sum().sum()}")

print("\nDataset summary for training data:")
print(train_df.describe())

print("\nDataset summary for test data:")
print(test_df.describe())


print("\nTarget distribution in training data:")
print(train_df['label'].value_counts())

print("\nTarget distribution in test data:")
print(test_df['label'].value_counts())

# Simplify the labels (binary classification: normal vs attack)
print("\nConverting to binary classification (normal vs attack)...")
train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

print("\nBinary target distribution in training data:")
print(train_df['binary_label'].value_counts())

print("\nBinary target distribution in test data:")
print(test_df['binary_label'].value_counts())

# Visualize attack distribution
plt.figure(figsize=(10, 6))
train_df['binary_label'].value_counts().plot(kind='bar')
plt.title('Distribution of Normal vs Attack Records (Training Data)')
plt.xlabel('Class (0: Normal, 1: Attack)')
plt.ylabel('Count')
plt.savefig('results/class_distribution_train.png')
plt.close()

plt.figure(figsize=(10, 6))
test_df['binary_label'].value_counts().plot(kind='bar')
plt.title('Distribution of Normal vs Attack Records (Test Data)')
plt.xlabel('Class (0: Normal, 1: Attack)')
plt.ylabel('Count')
plt.savefig('results/class_distribution_test.png')
plt.close()

# Identify numerical and categorical features
categorical_cols = ['protocol_type', 'service', 'flag']
numerical_cols = [col for col in train_df.columns if col not in ['label', 'binary_label'] + categorical_cols]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Number of numerical columns: {len(numerical_cols)}")

# Visualizing correlation between numerical features
plt.figure(figsize=(12, 8))
sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig('results/correlation_heatmap.png')
plt.close()

# Get combined unique values for categorical features
unique_categorical_values = {}
for col in categorical_cols:
    # Combine unique values from both train and test
    unique_values = set(train_df[col].unique()) | set(test_df[col].unique())
    unique_categorical_values[col] = list(unique_values)
    print(f"Unique values in {col}: {len(unique_values)}")

# Function to one-hot encode categorical features consistently
def encode_categorical_features(df, categorical_cols, unique_categorical_values):
    df_encoded = df.copy()
    
    # One-hot encode categorical features
    for col in categorical_cols:
        # Create columns for each unique value
        for val in unique_categorical_values[col]:
            df_encoded[f"{col}_{val}"] = (df_encoded[col] == val).astype(int)
        
        # Drop original column
        df_encoded = df_encoded.drop(col, axis=1)
    
    return df_encoded

# Process training and test data
print("\nEncoding categorical features...")
train_encoded = encode_categorical_features(train_df, categorical_cols, unique_categorical_values)
test_encoded = encode_categorical_features(test_df, categorical_cols, unique_categorical_values)

print(f"After encoding - Training data shape: {train_encoded.shape}")
print(f"After encoding - Test data shape: {test_encoded.shape}")

# Split training data into train and validation sets
print("\nSplitting training data into train and validation sets...")
X_train = train_encoded.drop(['label', 'binary_label'], axis=1)
y_train = train_encoded['binary_label']

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

X_test = test_encoded.drop(['label', 'binary_label'], axis=1)
y_test = test_encoded['binary_label']

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

# Scale numerical features
print("\nScaling numerical features...")
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_val[numerical_cols] = scaler.transform(X_val[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
# Ensure only numerical columns are scaled ----------------------------------------------------------------------------------------------------------------------
# numerical_cols = [col for col in train_df.columns if col not in categorical_cols + ['label', 'binary_label']]

# Save the scaler
with open('results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save the unique categorical values
with open('results/unique_categorical_values.pkl', 'wb') as f:
    pickle.dump(unique_categorical_values, f)

# Save processed data
print("\nSaving processed data...")
X_train.to_csv('results/X_train.csv', index=False)
pd.Series(y_train).to_csv('results/y_train.csv', index=False)
X_val.to_csv('results/X_val.csv', index=False)
pd.Series(y_val).to_csv('results/y_val.csv', index=False)
X_test.to_csv('results/X_test.csv', index=False)
pd.Series(y_test).to_csv('results/y_test.csv', index=False)

print("\nData preprocessing completed successfully!")
print("Preprocessed data saved in the 'results' folder.")