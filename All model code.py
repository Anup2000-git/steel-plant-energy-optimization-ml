import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')

# ------------------------
# Load Dataset
df = pd.read_csv('D:/360DigiTMG Date 28Aug/Project_360DigiTMG_HYD_26_02_25__02/model building/clean_data_Manufacturing.csv')

# ------------------------
# Removing Outliers (IQR Method)
numeric_df = df.select_dtypes(include=np.number)
Q1, Q3 = numeric_df.quantile(0.25), numeric_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df = df[~((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)]

# ------------------------
# Drop irrelevant columns
df.drop(columns=['SRNO', 'DATETIME', 'HEATNO','GRADE','PREV_TAP_TIME'], inplace=True)

# ------------------------
# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------------
# Scaling numerical features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# ------------------------
# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Production (MT)', 'ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)']])

# ------------------------
# Prepare data for classification
X = df.drop(columns=['Cluster'])
y = df['Cluster']

# Feature Selection
selector = SelectKBest(f_classif, k='all')
X = selector.fit_transform(X, y)

# ------------------------
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------
# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ------------------------
# Train Models
models = {}
accuracy_results = {}

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
models['Random Forest'] = rf
accuracy_results['Random Forest'] = rf_acc

# 2. KNN
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))
models['KNN'] = knn
accuracy_results['KNN'] = knn_acc

# 3. SVM
svm = SVC(C=1, kernel='rbf', probability=True)
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))
models['SVM'] = svm
accuracy_results['SVM'] = svm_acc

# 4. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
models['Gradient Boosting'] = gb
accuracy_results['Gradient Boosting'] = gb_acc

# 5. Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test))
models['Logistic Regression'] = lr
accuracy_results['Logistic Regression'] = lr_acc

# 6. Extra Trees
et = ExtraTreesClassifier(n_estimators=200, random_state=42)
et.fit(X_train, y_train)
et_acc = accuracy_score(y_test, et.predict(X_test))
models['Extra Trees'] = et
accuracy_results['Extra Trees'] = et_acc

# 7. XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
models['XGBoost'] = xgb_clf
accuracy_results['XGBoost'] = xgb_acc

# 8. LightGBM
lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
lgbm.fit(X_train, y_train)
lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))
models['LightGBM'] = lgbm
accuracy_results['LightGBM'] = lgbm_acc

# 9. CatBoost
cat = CatBoostClassifier(n_estimators=200, learning_rate=0.1, depth=5, verbose=0, random_state=42)
cat.fit(X_train, y_train)
cat_acc = accuracy_score(y_test, cat.predict(X_test))
models['CatBoost'] = cat
accuracy_results['CatBoost'] = cat_acc

# ------------------------
# Stacking Classifier
stack = StackingClassifier(
    estimators=[
        ('rf', rf),
        ('knn', knn),
        ('svm', svm),
        ('gb', gb),
        ('et', et),
        ('xgb', xgb_clf),
        ('lgbm', lgbm),
        ('cat', cat)
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stack.fit(X_train, y_train)
stack_acc = accuracy_score(y_test, stack.predict(X_test))
models['Stacking'] = stack
accuracy_results['Stacking'] = stack_acc

# ------------------------
# Model Performance Table
accuracy_df = pd.DataFrame({
    'Model': list(accuracy_results.keys()),
    'Test Accuracy': list(accuracy_results.values())
}).sort_values(by='Test Accuracy', ascending=False)

print("\nModel Accuracy Summary:\n")
print(accuracy_df)

# ------------------------
# Save Best Model
best_model_name = accuracy_df.iloc[0]['Model']
best_model = models[best_model_name]

with open("best_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

print(f"\n✅ Best Model '{best_model_name}' saved as best_model.pkl with accuracy: {accuracy_df.iloc[0]['Test Accuracy']:.4f}")

# ------------------------
# Save Scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save Label Encoders
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
    
# Save feature names
import pickle
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(df.drop(columns=['Cluster']).columns.tolist(), f)


print("\n✅ Scaler and Label Encoders saved successfully as scaler.pkl & label_encoders.pkl")
