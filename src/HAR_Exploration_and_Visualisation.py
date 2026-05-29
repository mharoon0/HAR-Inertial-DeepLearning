# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline  # Uncomment if running in Jupyter notebook
 
# Path to dataset folder
DATASET_PATH = os.path.join(".", "UCI HAR Dataset")

print("Dataset folder exists:", os.path.isdir(DATASET_PATH))

# %%
# Load feature names
features_path = os.path.join(DATASET_PATH, "features.txt")
features = pd.read_csv(features_path, sep=r"\s+", header=None, names=["index", "feature_name"])
feature_names = features["feature_name"].values

print("Number of features:", len(feature_names))
print("First 10 features:", feature_names[:10])

# Load activity labels
activity_labels_path = os.path.join(DATASET_PATH, "activity_labels.txt")
activity_labels = pd.read_csv(activity_labels_path, sep=r"\s+", header=None, names=["id", "activity"])

print("\nActivity labels:")
activity_labels


# %%
# Train paths
X_train_path = os.path.join(DATASET_PATH, "train", "X_train.txt")
y_train_path = os.path.join(DATASET_PATH, "train", "y_train.txt")
subject_train_path = os.path.join(DATASET_PATH, "train", "subject_train.txt")

# Test paths
X_test_path = os.path.join(DATASET_PATH, "test", "X_test.txt")
y_test_path = os.path.join(DATASET_PATH, "test", "y_test.txt")
subject_test_path = os.path.join(DATASET_PATH, "test", "subject_test.txt")

# 1) Load WITHOUT column names (header=None)
X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None, names=["activity_id"])
subject_train = pd.read_csv(subject_train_path, delim_whitespace=True, header=None, names=["subject"])

X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None, names=["activity_id"])
subject_test = pd.read_csv(subject_test_path, delim_whitespace=True, header=None, names=["subject"])

# 2) Now assign the feature names as columns
X_train.columns = feature_names
X_test.columns = feature_names

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)
print("Test X:", X_test.shape)
print("Test y:", y_test.shape)


# %%
# Combine subject + activity + features for train and test
train_df = pd.concat([subject_train, y_train, X_train], axis=1)
test_df = pd.concat([subject_test, y_test, X_test], axis=1)

# Merge train and test
full_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

print("Full dataset shape:", full_df.shape)
full_df.head()


# %%
# Map activity_id to readable activity names
activity_map = dict(zip(activity_labels["id"], activity_labels["activity"]))
full_df["activity"] = full_df["activity_id"].map(activity_map)

# Count samples per activity
activity_counts = full_df["activity"].value_counts()
print(activity_counts)

plt.figure(figsize=(8,4))
sns.barplot(x=activity_counts.index, y=activity_counts.values)
plt.xticks(rotation=45)
plt.title("Samples per Activity")
plt.xlabel("Activity")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# %%
# X = all numeric sensor features
X = full_df[feature_names]

# y = activity label (text)
y = full_df["activity"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# %%
from sklearn.decomposition import PCA

# Compute PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "activity": y
})

plt.figure(figsize=(12,8))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="activity",
    palette="Set1",
    s=25,
    alpha=0.8
)

# Better labels
plt.xlabel("PC1 (Overall Movement Intensity)")
plt.ylabel("PC2 (Posture & Direction of Movement)")
plt.title("PCA of HAR Dataset (561 Features → 2D)\nPC1 = intensity, PC2 = posture/orientation")

# Cleaner legend
plt.legend(
    title="Activity",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()
plt.show()



