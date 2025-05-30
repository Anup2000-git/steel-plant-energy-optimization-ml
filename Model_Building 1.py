

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings('ignore')
# Load Dataset

df = pd.read_csv("E:/2nd project/Latest_Manufacturing.csv")

# Exploratory Data Analysis (EDA)
print(df.describe())  # Summary statistics
print(df.isnull().sum())  # Check for missing values

# Visualizing distributions
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['Production (MT)', 'ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)']])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Removing outliers using IQR method for numeric columns only
numeric_df = df.select_dtypes(include=np.number)
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[~((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)]

# Drop irrelevant columns (ID-like and datetime columns)
df.drop(columns=['SRNO', 'DATETIME', 'HEATNO'], inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scaling numerical features
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Clustering on key features
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Production (MT)', 'ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)']])

# Prepare data for classification
X = df.drop(columns=['Cluster'])
y = df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Random Forest": RandomForestClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)

# Display results
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df)

"""OUTPUT COLUMNS ARE -Production (MT),  TT_TIME (Total Cycle Time Including Breakdown), ENERGY (Energy Consumption) PERFORM CLUSTERING IN THESE THREE OUTPUT COLUMNS.THEN ONE DISCRETE COLUMN WILL BE CREATED

The inertia values suggest that the Elbow Method should be used to determine the optimal number of clusters.
"""
