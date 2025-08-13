import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_students = 200
data = {
    'GPA': np.random.uniform(0, 4, num_students),
    'Attendance': np.random.randint(0, 100, num_students), # Percentage
    'MissedAssignments': np.random.randint(0, 20, num_students),
    'StudyHours': np.random.randint(0,10, num_students),
    'AtRisk': np.random.choice([0, 1], num_students, p=[0.7, 0.3]) # 0: Not at risk, 1: At risk
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
# In a real-world scenario, this section would involve handling missing values, outliers, etc.
# For this example, the data is already clean.
# --- 3. Data Splitting ---
X = df.drop('AtRisk', axis=1)
y = df['AtRisk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training ---
model = LogisticRegression() #Simple model for demonstration
model.fit(X_train, y_train)
# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
# --- 6. Visualization ---
# Feature Importance (Illustrative - LogisticRegression doesn't directly provide feature importance like tree-based models)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
#Confusion Matrix
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png")