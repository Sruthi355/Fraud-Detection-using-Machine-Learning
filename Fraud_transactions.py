import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score


# Replace this with your actual file path
file_path = 'C:/Users/Hp/Documents/AI_sample_files/fraud_transactions.csv'
# Load the data
df = pd.read_csv(file_path)

# Display first few rows
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

sns.countplot(x='IsFraud', data=df)
plt.title('Fraud vs Non-Fraud Transactions')
plt.show()

# 2. Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['TransactionType', 'Location', 'DeviceID'], drop_first=True)

# 3. Convert 'Timestamp' to datetime and extract useful features
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
df['Month'] = df['Timestamp'].dt.month

# Optional: drop Timestamp after feature extraction
df.drop(columns=['Timestamp'], inplace=True)
# 4. Drop IDs if not needed for prediction
df.drop(columns=['TransactionID', 'CustomerID'], inplace=True)

# Drop missing values
df = df.dropna()
# 5. Separate input features and target label
X = df.drop('IsFraud', axis=1)
y = df['IsFraud']

# Step 8.1: Feature scaling
scaler = StandardScaler()

# training the model
# 1. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model first one
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#Second model
# 2. Initialize and train the logistic regression model
#model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed


# Train the model on second model scaled data
#model.fit(X_train_scaled, y_train)

# Make predictions first model
y_pred = model.predict(X_test)

# Predict on scaled test data
#y_pred = model.predict(X_test_scaled)


# 4. Evaluate the model
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("\nModel Evaluation Results")
print("------------------------")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

## Train Random Forest
#rf_model = RandomForestClassifier(random_state=42)
#rf_model.fit(X_train, y_train)
#
## Predict
#rf_pred = rf_model.predict(X_test)
#
#print("\nRandom Forest Evaluation")
#print("------------------------")
#print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
#print("Confusion Matrix:")
#print(confusion_matrix(y_test, rf_pred))
#print("\nClassification Report:")
#print(classification_report(y_test, rf_pred))

# Save the logistic regression model
joblib.dump(model, 'logistic_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')





