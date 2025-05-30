#---------------------------------------------------- Data preprocessing code ------------------------------------------

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
from sqlalchemy import create_engine  # For database connection

# Load Dataset
df_sorted = pd.read_csv(r"D:/360DigiTMG Date 28Aug/Project_360DigiTMG_HYD_26_02_25__02/Data Preprocessing ALL_Code/steel data_csv.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = ('802158')  # password
db = 'steel'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Push the dataframe onto a SQL table
df_sorted.to_sql('manufacture', con=engine, if_exists='replace', chunksize=1000, index=False)

# Read the data from MySQL Database
sql = 'select * from manufacture;'
df = pd.read_sql_query(sql, engine)

# Load Excel file
file_path = "D:/360DigiTMG Date 28Aug/Project_360DigiTMG_HYD_26_02_25__02/Data Preprocessing ALL_Code/steel data.xlsx"
xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls, sheet_name="steel manufacturing")
print(df.head())

df_sorted = pd.read_excel(xls, sheet_name="steel manufacturing", skiprows=2)
print(df_sorted.head())

# Display dataset info
df_sorted.info()
print(df_sorted.columns)
print(df_sorted.describe())
print(df_sorted.shape)

# Check for missing values
print(df_sorted.isnull().sum())

# Sorting dataframe
df_sorted = df_sorted.sort_values(by=['SRNO'], ascending=[True])  # Change 'SRNO' to 'LogSheet' if necessary
print(df_sorted)

# Convert 'PREV_TAP_TIME' to datetime
df_sorted['PREV_TAP_TIME'] = pd.to_datetime(df_sorted['PREV_TAP_TIME'], errors='coerce')

df_sorted['PREV_TAP_TIME'].fillna(method='ffill', inplace=True)
df_sorted['PREV_TAP_TIME'].fillna(df_sorted['PREV_TAP_TIME'].median(), inplace=True)
df_sorted['PREV_TAP_TIME'].fillna(method='ffill', inplace=True)

# Fill missing values
df_sorted['Production (MT)'].fillna(df_sorted['Production (MT)'].mean(), inplace=True)
df_sorted['PREV_TAP_TIME'].fillna(method='ffill', inplace=True)

# Check for missing values
print(df_sorted.isnull().sum())

# Select only numerical columns for further analysis
numerical_df = df_sorted.select_dtypes(include=['number'])
print(numerical_df.head())

# Statistical Analysis
print("Mean:\n", numerical_df.mean())
print("Median:\n", numerical_df.median())
print("Mode:\n", numerical_df.mode().iloc[0])
print("Variance:\n", numerical_df.var())
print("Standard Deviation:\n", numerical_df.std())
print("Range:\n", numerical_df.max() - numerical_df.min())
print("Skewness:\n", numerical_df.skew())
print("Kurtosis:\n", numerical_df.kurt())


"""#Univariate Analysis (Individual Numerical Columns)
Histograms for Distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for all numerical columns in numerical_df
numerical_df.hist(figsize=(15, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms of Numerical Columns", fontsize=16)
plt.show()

# Boxplot for all numerical columns
plt.figure(figsize=(15, 8))
# Get numerical columns dynamically
num_cols = df_sorted.select_dtypes(include=['number']).columns
df_sorted[num_cols].boxplot(rot=90)  # Updated to use numerical column names
plt.title("Boxplots of Numerical Columns")
plt.show()

"""#Bivariate Analysis (Relationships Between Two Numerical Columns)
Correlation Matrix
"""

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df_sorted[num_cols].corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

"""# Scatter Plot: Energy Consumption vs Production"""

sns.scatterplot(x=df_sorted['ENERGY (Energy Consumption)'], y=df_sorted['Production (MT)'])
plt.title('Energy Consumption vs Production')
plt.xlabel('Energy Consumption (KWh)')
plt.ylabel('Production (MT)')
plt.show()

"""#Boxplot: Steel Grade vs Energy Consumption"""

plt.figure(figsize=(12,6))
sns.boxplot(x=df_sorted['GRADE'], y=df_sorted['ENERGY (Energy Consumption)']) # Changed df to df_sorted
plt.xticks(rotation=90)
plt.title('Steel Grade vs Energy Consumption')
plt.show()

""" #Multivariate Analysis (Relationships Between Multiple Numerical Columns)"""

#Regression Plot (Energy vs Production)
sns.lmplot(x='ENERGY (Energy Consumption)', y='Production (MT)', data=df_sorted)
plt.title('Regression Plot: Energy Consumption vs Production')
plt.show()


import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Display data types before processing
df_sorted = df_sorted.copy()
print(df_sorted.dtypes)

# 1. TYPE CASTING
# Convert 'DATETIME' column to datetime format
df_sorted['DATETIME'] = pd.to_datetime(df_sorted['DATETIME'], errors='coerce')

# Select numeric columns, excluding specific categorical columns
num_cols = df_sorted.select_dtypes(include=['object']).columns.difference(['DATETIME', 'GRADE', 'SECTION_IC'])

# Convert selected columns to numeric
df_sorted[num_cols] = df_sorted[num_cols].apply(pd.to_numeric, errors='coerce')

# Display data types after conversion
print(df_sorted.dtypes)
df_sorted.info()

# 2. HANDLING DUPLICATES
# Remove duplicate rows from the dataset
df_sorted.drop_duplicates(inplace=True)

# 3. OUTLIER ANALYSIS (Using IQR Method)
# Convert object columns to numeric where possible
for column in df_sorted.select_dtypes(include=['object']).columns:
    try:
        df_sorted[column] = pd.to_numeric(df_sorted[column], errors='coerce')
    except ValueError:
        print(f"Could not convert column '{column}' to numeric. Check for non-numeric values.")

# Compute IQR for numerical columns
Q1 = df_sorted.select_dtypes(include=np.number).quantile(0.25)
Q3 = df_sorted.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

# Identify and remove outliers
numerical_df = df_sorted.select_dtypes(include=np.number)
outliers = ((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).sum()
print("Outliers Count:", outliers[outliers > 0])

df_sorted = df_sorted[~((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).any(axis=1)]

# 4. ZERO & NEAR ZERO VARIANCE FEATURES
# Remove features with near-zero variance
var_threshold = VarianceThreshold(threshold=0.01)
df_sorted_var = df_sorted.drop(columns=['DATETIME', 'GRADE', 'SECTION_IC'])
df_sorted_var = df_sorted_var.loc[:, var_threshold.fit(df_sorted_var).get_support()]
print("Remaining Features after Zero Variance Filtering:", df_sorted_var.columns)

df_sorted = df_sorted[list(df_sorted_var.columns) + ['DATETIME', 'GRADE', 'SECTION_IC']]

# 5. HANDLING MISSING VALUES
# Forward fill missing values
df_sorted.fillna(method='ffill', inplace=True)
print("Missing Values After Handling:", df_sorted.isnull().sum().sum())

# 6. DISCRETIZATION/BINNING/GROUPING
# Categorize 'ENERGY (Energy Consumption)' into quartiles
df_sorted['ENERGY (Energy Consumption)'] = pd.qcut(df_sorted['ENERGY (Energy Consumption)'], q=4, labels=["Low", "Medium", "High", "Very High"])

# 7. DUMMY VARIABLE CREATION
# Convert categorical variables into dummy/indicator variables
df_sorted = pd.get_dummies(df_sorted, columns=['GRADE', 'SECTION_IC'], drop_first=True)

# 8. TRANSFORMATION (Scaling Numeric Data)
# Standardize numeric columns, excluding unique identifiers
scaler = StandardScaler()
scaled_cols = df_sorted.select_dtypes(include=['number']).columns.difference(['SRNO', 'HEATNO'])
df_sorted[scaled_cols] = scaler.fit_transform(df_sorted[scaled_cols])

# Save cleaned data to a CSV file
df_sorted.to_csv("cleaned_data.csv", index=False)