# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
# FEATURE SCALING

import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![Screenshot 2025-04-27 095130](https://github.com/user-attachments/assets/18d22f76-b18c-4172-95b0-c019076c3b0a)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![Screenshot 2025-04-27 095312](https://github.com/user-attachments/assets/16cc9284-3af4-4a38-b55e-7c5ecb1e3c82)
```
df.dropna()
```
![Screenshot 2025-04-27 095357](https://github.com/user-attachments/assets/ee828260-29e5-4685-8835-3466805eaf0f)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
```
![Screenshot 2025-04-27 095451](https://github.com/user-attachments/assets/61d88c6b-cd07-471b-badb-9c9ec8d7138f)
```
# Standard Scaling
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
![Screenshot 2025-04-27 095556](https://github.com/user-attachments/assets/6e50a4dc-3fe4-4104-86a5-d8bb80312594)
```
sc=StandardScaler()

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![Screenshot 2025-04-27 095744](https://github.com/user-attachments/assets/d61f052b-c435-4390-a287-0a3426f92308)
```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2025-04-27 095844](https://github.com/user-attachments/assets/4ababf3d-7383-4e1d-82db-136a26820ab5)
```
#MAXIMUM ABSOLUTE SCALING:

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/15f6704a-0cd0-42af-bb4d-d6ef7787a374)
```
#ROBUST SCALING

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
```
![image](https://github.com/user-attachments/assets/11a5bc1c-01f4-4ad5-a587-fd2b8cd53795)
```
#FEATURE SELECTION:

df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/8087eb90-f551-4a74-aec6-ff6ab1996647)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/cd0f0629-e9f5-433d-9dd9-db21c1b262b2)
```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
 # process non-numeric features. It ensures the model treats these columns as categories,
  # not as continuous numerical values.
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ded91458-761b-4ab5-9c86-3b19fff27e47)
```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/67a90339-5310-4c5b-a8a0-fa506a2f290a)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict.
```
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/ea2bfd55-596a-455c-a64c-37dd10ceaf48)
```
y_pred = rf.predict(X_test)
```
```
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/70ea73a9-1cb4-413b-9fbe-3ebb683b0fbd)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/1a3e0c7c-3e19-43fa-a1a0-a39b10c4eb53)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/21ab294a-28d3-479c-9971-50ac2ff96bfe)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/cf8ed415-59ea-4329-a846-d88d8fc17bc3)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/f422ebf6-491a-40c1-9963-50466efd0e05)
```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/91492e40-7149-4f5b-af8e-95f214322f0a)
```
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/e4daa8fa-f6ef-41a8-9fde-756d55a9f9d4)
```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
```
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

df[categorical_columns] = df[categorical_columns].astype('category')
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/2bc94b53-efb9-4995-a4c5-a4b2cf0effcf)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```

![image](https://github.com/user-attachments/assets/e734751e-0086-4003-b8a6-4819eae64de9)
```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
    'JobType',
    'EdType',
    'maritalstatus',
    'occupation',
    'relationship',
    'race',
    'gender',
    'nativecountry'
]

# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```

![image](https://github.com/user-attachments/assets/599b1efb-c6f4-4aae-966b-66cc061ee2e9)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

![image](https://github.com/user-attachments/assets/301240d4-f46b-448a-8c64-594c33884ed1)

![image](https://github.com/user-attachments/assets/57b3df29-cb0e-4c89-b1b9-13080bb8e6b4)


# RESULT:
       Thus the given data is read and performed the Feature Scaling and Feature Selection process and saved the file.
