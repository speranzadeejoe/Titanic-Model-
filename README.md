# Titanic-Model-
# Titanic Survival Prediction

## Overview
This project involves predicting the survival of passengers on the Titanic using machine learning models. We preprocess the dataset, encode categorical variables, and apply multiple classifiers, including Logistic Regression, Decision Tree, Random Forest, Naïve Bayes, and Support Vector Machine (SVM).

## Dataset
The dataset contains information about Titanic passengers, including:
- **pclass** (Passenger class)
- **sex** (Gender of the passenger)
- **age** (Age of the passenger)
- **sibsp** (Number of siblings/spouses aboard)
- **parch** (Number of parents/children aboard)
- **ticket** (Ticket number)
- **fare** (Ticket fare)
- **cabin** (Cabin number)
- **embarked** (Port of Embarkation)
- **boat** (Lifeboat number)
- **home.dest** (Home destination)

## Preprocessing Steps
1. **Handling Missing Values**: Fill missing values using `fillna(0)`.
2. **Encoding Categorical Variables**: Using Label Encoding for `sex`, `embarked`, etc.
3. **Feature Selection**: Dropping unnecessary columns such as `name`, `ticket`, `cabin`, `boat`, and `home.dest`.
4. **Splitting Data**: Using `train_test_split()` for an 80%-20% training-test split.

## Machine Learning Models
### **1. Logistic Regression**
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
```

### **2. Decision Tree Classifier**
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
```

### **3. Random Forest Classifier**
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
```

### **4. Naïve Bayes Classifier**
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Naïve Bayes Accuracy: {accuracy * 100:.2f}%")
```

### **5. Support Vector Machine (SVM)**
```python
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")
```

## Evaluation
The models are evaluated based on accuracy. Additional metrics like precision, recall, and F1-score can also be used.
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test)))
```

## Conclusion
This project demonstrates how different machine learning models perform on the Titanic dataset. The best model can be chosen based on accuracy and other evaluation metrics.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn

## How to Run
1. Install dependencies using `pip install pandas numpy scikit-learn`
2. Run the script to train models and check accuracy.

---
**Author:** Speranza Deejoe 

Happy Coding! 🚀

