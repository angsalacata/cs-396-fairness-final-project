import sklearn as sk
from sklearn.svm import LinearSVC
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    true_positive_rate,
    false_positive_rate
)
  

# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
X_original = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
  
# # metadata 
# print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 

# v = TfidfVectorizer()
le = LabelEncoder()
scaler = StandardScaler()
oe_encoder = OneHotEncoder(sparse_output=False)

# # variable information 
# print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 

# X_cols = X_original.drop(['change', 'diabetesMed'], axis=1).columns.values

# this gets the data except for change and diabetes med columns. 
# dropping diag_1, diag_2, diag_3
X_df = X_original.drop(['change', 'diabetesMed', 'diag_1', 'diag_2','diag_3'], axis=1)

protected_series = {'race': X_df['race'].copy(), 'gender': X_df['gender'].copy(), 'age': X_df['age'].copy(), 'weight': X_df['weight'].copy()}

categorical_columns = X_df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X_df.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical columns length: {len(categorical_columns)}")
print(f"Categorical columns: {(categorical_columns)}")

print(f"Numerical columns length: {len(numerical_columns)}")
print(f"Numerical columns: {(numerical_columns)}")



preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), categorical_columns)
    ])

X_preprocessed = preprocessor.fit_transform(X_df)
print(f"Preprocessed shape: {X_preprocessed.shape}")

newcols = []
for name, transformer, feats in preprocessor.transformers_:
  if hasattr(transformer, 'get_feature_names_out'):
    print(name, transformer.get_feature_names_out(feats))
    newcols.extend(transformer.get_feature_names_out(feats))
  

with open('newcategories.txt', 'w') as f:
    for line in newcols:
        f.write(f"{line}\n")

y_change = (X_original['change'] == 'Ch').values
y_diabetesMed = (X_original['diabetesMed'] == 'Yes').values
# != ''
# print(y_change)
# X_vals = le.fit_transform(X_df)
# X_vals = scaler.fit_transform(X_encoded)

# print(X_vals)
index_array = np.arange(X_preprocessed.shape[0])

X_train, X_test, Y_change_train, Y_change_test, Y_diabetesMed_train, Y_diabetesMed_test, train_indices, test_indices = train_test_split(X_preprocessed, y_change, y_diabetesMed, index_array, test_size=.3, random_state=42)

sensitive_train = {attr: vals.iloc[train_indices].reset_index(drop=True) 
                   for attr, vals in protected_series.items()}
sensitive_test = {attr: vals.iloc[test_indices].reset_index(drop=True) 
                  for attr, vals in protected_series.items()}

print("training sizes")
print((X_train.shape[0]))
print(len(Y_change_train))
print(len(Y_diabetesMed_train))

print(type(X_train))
print((X_train))
# 1. base svm, no effort to implement measures to ensure fairness

# print(X_cols)
# gamma{‘scale’, ‘auto’} or float, default=’scale’
# Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
start = time.time()
changeSVM = LinearSVC(C=1, max_iter=1000, dual='auto', random_state=42)
# target with diabetes change
changeSVM.fit(X_train, Y_change_train)
changePredictions = changeSVM.predict(X_test)
changeSVMAccuracy = accuracy_score(changePredictions, Y_change_test)
changeTime = time.time() - start

# target with diabetes change
diabetesMedSVM = LinearSVC(C=1, max_iter=1000, dual='auto', random_state=42)
start = time.time()
diabetesMedSVM.fit(X_train, Y_diabetesMed_train)
diabetesMedPredictions = diabetesMedSVM.predict(X_test)
diabetesMedSVMAccuracy = accuracy_score(diabetesMedPredictions, Y_diabetesMed_test)
diabetesMedTime = time.time() - start


print("-"*50)
print("BASE ACCURACIES")
print("-"*50)
print(f"CHANGE TIME: {changeTime}")
print(f"Accuracy of change SVM: {changeSVMAccuracy}")

print(f"DIABETES MED TIME: {diabetesMedTime}")
print(f"Accuracy of diabetesMed SVM: {diabetesMedSVMAccuracy}")
