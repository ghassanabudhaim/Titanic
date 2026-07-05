# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split as TTS


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features = ["Pclass","Age","SibSp", "Sex", "Embarked"]


# %%
# Calculate the median age from the training data (e.g., around 28)
median_age = train_data['Age'].median()
train_data['Age'] = train_data['Age'].fillna(median_age)
test_data['Age'] = test_data['Age'].fillna(median_age)

#encoding categorical variables
most_common_port = train_data['Embarked'].mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(most_common_port) #
test_data['Embarked'] = test_data['Embarked'].fillna(most_common_port)
most_common_port

# %%
all_data = pd.concat([train_data[features], test_data[features]], axis=0)
all_data_encoded = pd.get_dummies(all_data, columns=["Sex", "Embarked"])



X_train_full = all_data_encoded.iloc[:len(train_data)]  
y_train_full = train_data["Survived"]
X_test = all_data_encoded.iloc[len(train_data):]   

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = TTS(X_train_full, y_train_full, test_size=0.2, random_state=42)


# X_train, y_train, X_test, y_test = train_test_split()
model = LR(class_weight = 'balanced', max_iter=1000, C = 0.01)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model successfully trained! First 5 predictions:", y_pred[:5])

val_preds = model.predict(X_val)

print("Model successfully trained!")

# Calculate metrics using the unseen validation sets
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)
f1 = f1_score(y_val, val_preds)

print("--- VALIDATION DATA SCORES (Honest Evaluation) ---")
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)

# Generate final Kaggle test file predictions
y_pred = model.predict(X_test)
print("\nFirst 5 Kaggle test predictions:", y_pred[:5])



# %%



