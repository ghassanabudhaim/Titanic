import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

features = ["Pclass","Age","SibSp","Parch", "Sex", "Embarked"]

# Calculate the median age from the training data (e.g., around 28)
median_age = train_data['Age'].median()
train_data['Age'] = train_data['Age'].fillna(median_age)
test_data['Age'] = test_data['Age'].fillna(median_age)

#encoding categorical variables
most_common_port = train_data['Embarked'].mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(most_common_port)
test_data['Embarked'] = test_data['Embarked'].fillna(most_common_port)

all_data = pd.concat([train_data[features], test_data[features]], axis=0)
all_data_encoded = pd.get_dummies(all_data, columns=["Sex", "Embarked"])

X_train = all_data_encoded.iloc[:len(train_data)]  # First 891 rows (train data)
y_train = train_data["Survived"]
X_test = all_data_encoded.iloc[len(train_data):]   # Remaining 418 rows (test data)



model = LR()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model successfully trained! First 5 predictions:", y_pred[:5])

precision = precision_score(y_train, model.predict(X_train))
recall = recall_score(y_train, model.predict(X_train))
f1 = f1_score(y_train, model.predict(X_train))

print("Precision on training data:", precision)
print("Recall on training data:", recall)
print("F1 score on training data:", f1)


