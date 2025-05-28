import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_score
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

df = pd.read_csv("../dataset/LoanApprovalPrediction.csv")
df.info()
df = df.drop(columns=['Loan_ID'])

print("before filling null values")
print(df.isnull().sum())

print("after filling null values*")

df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(2)

#assign 1 or 0 to float values
df['Gender'] = df['Gender'].apply(lambda x:1 if x =='Male' else 0)
df['Married'] = df['Married'].apply(lambda x:1 if x == "Yes" else 0)
df['Education'] = df['Education'].apply(lambda x:1 if x == "Graduate" else 0)
df['Self_Employed'] = df['Self_Employed'].apply(lambda x:1 if x == "Yes" else 0)




#label encoding for categorical variables
df['Property_Area'] = df['Property_Area'].apply(lambda x: 2 if x == 'Urban' else 0 if x == 'Rural' else 1)
print(df['Property_Area'])

unique_values = df['Credit_History'].value_counts()
print(unique_values)
#found 1 0 and 341. 341 is outlier. It is better to replace it with lower ratio value which is 0
df['Credit_History'] = df['Credit_History'].apply(lambda x: 0 if abs(x - 341.917808) < 1e-6 else x)
unique_values = df['Credit_History'].value_counts()
print(unique_values)

x = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

print("x")
print(x.shape)
print("y")
print(y)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 16)

# SVM Classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)  # Train the model

# Predictions
y_pred = svm_model.predict(X_test)

# Evaluation
print("Accuracy of SVM model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


feature_names = X_train.columns
importance = svm_model.coef_[0]  # Lineer kernel için koefisyanlar

# Özellik önem grafiği
plt.bar(feature_names, importance)
plt.xlabel("Özellikler")
plt.ylabel("Önem Skoru (Koefisyanlar)")
plt.title("Feature Importance (Linear Kernel)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


skorlar = cross_val_score(svm_model, x, y, cv=10)
print(f"skorlar: {skorlar}")
print(f"skorlar ortalaması: {skorlar.mean()}")

y_test_encoded = y_test.apply(lambda x: 1 if x == 'Y' else 0)
y_pred_encoded = [1 if x == 'Y' else 0 for x in y_pred]

fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred_encoded)
roc_auc = auc(fpr, tpr)

from sklearn.inspection import permutation_importance

# SVM modelini oluştur ve eğit
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Permutasyon ile özellik önemini hesapla
perm_importance = permutation_importance(svm_model, X_test, y_test, scoring='accuracy')

# Özellik önem grafiği
feature_names = X_train.columns
importance = perm_importance.importances_mean
plt.bar(feature_names, importance)
plt.xlabel("Özellikler")
plt.ylabel("Önem Skoru (Permutasyon)")
plt.title("Feature Importance (Permutation Importance)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()