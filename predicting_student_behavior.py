import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython.display import display

# Load dataset
df = pd.read_csv('students_social_network.csv')

# Clean 'age' and 'NumberOffriends'
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['NumberOffriends'] = pd.to_numeric(df['NumberOffriends'], errors='coerce')

# Drop rows with invalid numeric values
df.dropna(subset=['age', 'NumberOffriends'], inplace=True)
df.dropna(inplace=True)

# Encode 'gender' column
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])  # e.g., F -> 0, M -> 1

# Create supervised label: risky_behavior
df['risky_behavior'] = np.where((df['drugs'] == 1) | (df['drunk'] == 1), 1, 0)

# Define keywords and target
X = df.drop(columns=['gradyear', 'risky_behavior'])  # Drop gradyear and target
y = df['risky_behavior']

# Scale keywords
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# Keyword Importances
keyword_importances = pd.Series(model.feature_importances_, index=df.drop(columns=['gradyear', 'risky_behavior']).columns)
keyword_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Keywords')
plt.xlabel('Keyword Importance Score')
plt.show()

# Identify 5 Wrong Predictions
wrong_indices = np.where(y_test != y_pred)[0]
print("\n5 Wrong Predictions Analysis:")
for idx in range(min(5, len(wrong_indices))):
    true_label = y_test.iloc[wrong_indices[idx]]
    predicted_label = y_pred[wrong_indices[idx]]
    print(f"\nSample {idx+1}:")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print(f"Keywords: {pd.Series(X.iloc[wrong_indices[idx]], index=df.drop(columns=['gradyear', 'risky_behavior']).columns)}")

# Create a table comparing true labels vs predictions
results = pd.DataFrame(X_test, columns=df.drop(columns=['gradyear', 'risky_behavior']).columns)
results['True Label'] = y_test.values
results['Predicted Label'] = y_pred
results['Correct?'] = np.where(results['True Label'] == results['Predicted Label'], '✅', '❌')

# Reset index to match properly
results.reset_index(drop=True, inplace=True)

print("\nPrediction Results Sample:")
display(results[['age', 'gender', 'NumberOffriends', 'True Label', 'Predicted Label', 'Correct?']].head(15))
