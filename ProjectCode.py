import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read CSV (use raw string to fix path error)
df = pd.read_csv(r"C:\Users\yasee\OneDrive\Desktop\Projects\PredProject\Weather Precipitation Predictor\weatherHistory.csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Create a binary target column based on Precip Type
df['Precip Type'] = df['Precip Type'].map({'rain': 0, 'snow': 1})

# Select features
features = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
            'Wind Speed (km/h)', 'Visibility (km)']
X = df[features]
y = df['Precip Type']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# ✅ KNN Classifier
# -------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n🔹 KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("🔹 KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("🔹 KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

# -------------------------
# ✅ Decision Tree Classifier
# -------------------------
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_scaled, y_train)
y_pred_dt = dtree.predict(X_test_scaled)

print("\n🔸 Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("🔸 Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print("🔸 Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# -------------------------
# ✅ Confusion Matrix Heatmaps
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Oranges')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()
