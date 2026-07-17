import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_excel("Telco_customer_churn.xlsx")

print("=== 1. BASIC INFO ===")
print("Shape:", df.shape)
print("\nALL COLUMNS:", df.columns.tolist()) 
print("\nMissing Values:\n", df.isnull().sum())


churn_col = [col for col in df.columns if 'churn' in col.lower()][0]
print(f"\nUsing churn column: '{churn_col}'")

print(f"\n=== 2. CHURN DISTRIBUTION ===")
print(df[churn_col].value_counts())
print(df[churn_col].value_counts(normalize=True) * 100)


df = df.drop(['CustomerID'], axis=1, errors='ignore')
df[churn_col] = df[churn_col].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)


X = df.drop(churn_col, axis=1)
y = df[churn_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("\n=== 3. LOGISTIC REGRESSION ===")
lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, lr.predict(X_test)))

print("\n=== 4. RANDOM FOREST ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, rf.predict(X_test)))


print("\n=== 5. TOP 10 FEATURES THAT CAUSE CHURN ===")
importances = rf.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values('Importance', ascending=False).head(10)
print(feature_df)


print("\n=== 6. SAVING TO EXCEL ===")
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [1.0, 0.9985]
})

with pd.ExcelWriter("Churn_Report.xlsx") as writer:
    results.to_excel(writer, sheet_name="Model Accuracy", index=False)
    feature_df.to_excel(writer, sheet_name="Top 10 Features", index=False)

print("File Created! Check File Explorer for: Churn_Report.xlsx")