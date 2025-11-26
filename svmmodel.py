from operator import eq
from itertools import combinations
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
from scipy import sparse
from sklearn.impute import SimpleImputer

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    true_negative_rate,
    false_negative_rate
)

from fairlearn.postprocessing import ThresholdOptimizer


def get_pairwise_metrics(y_true, y_pred, sensitive_features):

  df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'protected_attributes': sensitive_features
  })

  unique_group_values = (df['protected_attributes'].unique())

  unique_group_values = [x for x in unique_group_values if str(x) != 'nan' ]

  print(unique_group_values)

  pairwise_results = []

  # create the pair wise comparisons for curr sensitive features. Eg, if our sensitive feature is the race column, pairs could be Caucasian/AfricanAmerican, Caucasian/Hispanic, AfricanAmerican/Hispanic, etc
  for group_val_1, group_val_2 in combinations(unique_group_values, 2):
    # grab what rows have group 1 and group 2 (eg, if we are comparing Caucasian/AfricanAmerican, then we are grabbing the rows that have Caucasian and AfricanAmerican respectively)

    # print(f"group val 1: {group_val_1} and group val 2: {group_val_2}")

    mask = (df['protected_attributes'] == group_val_1) | (df['protected_attributes'] == group_val_2)
    filtered_df = df[mask]

    group_1_df = df[df['protected_attributes'] == group_val_1]
    group_2_df = df[df['protected_attributes'] == group_val_2]

    # print(f"len of group1 true: {len(group_1_df['y_true'])} group1 pred: {len(group_1_df['y_pred'])}")

    # print(f"len of group2 true: {len(group_2_df['y_true'])} group2 pred: {len(group_2_df['y_pred'])}")

    #######################
    # Calculation for Demographic Parity Ratio and Difference
    sr1 = selection_rate(group_1_df['y_true'], group_1_df['y_pred'])
    sr2 = selection_rate(group_2_df['y_true'], group_2_df['y_pred'])

    manual_DP_ratio = min(sr1, sr2)/max(sr1, sr2) if max(sr1, sr2) > 0 else 0
    manual_DP_diff = abs(sr1-sr2)

    DP_ratio = demographic_parity_ratio(filtered_df['y_true'], filtered_df['y_pred'], sensitive_features=filtered_df['protected_attributes'])

    DP_diff = demographic_parity_difference(filtered_df['y_true'], filtered_df['y_pred'], sensitive_features=filtered_df['protected_attributes'])
    #######################


    #######################
    # Calculation for Equalized Odds Ratio and Difference
    group_1_tpr = true_positive_rate(group_1_df['y_true'], group_1_df['y_pred'], pos_label=1)
    group_2_tpr = true_positive_rate(group_2_df['y_true'], group_2_df['y_pred'], pos_label=1)

    group_1_fpr = false_positive_rate(group_1_df['y_true'], group_1_df['y_pred'], pos_label=1)
    group_2_fpr = false_positive_rate(group_2_df['y_true'], group_2_df['y_pred'], pos_label=1)

    tpr_ratio = min(group_1_tpr, group_2_tpr)/max(group_1_tpr, group_2_tpr) if max(group_1_tpr, group_2_tpr) > 0 else 0

    fpr_ratio = min(group_1_fpr, group_2_fpr)/max(group_1_fpr, group_2_fpr) if max(group_1_fpr, group_2_fpr) > 0 else 0

    # get smaller of two ratios
    manual_eq_odds_ratio = min(tpr_ratio, fpr_ratio)

    tpr_difference = abs(group_1_tpr-group_2_tpr)

    fpr_difference = abs(group_1_fpr-group_2_fpr)
    # get larger of the differences
    manual_eq_odds_difference = max((tpr_difference, fpr_difference))

    eq_odds_ratio = equalized_odds_ratio(filtered_df['y_true'], filtered_df['y_pred'], sensitive_features=filtered_df['protected_attributes'])

    eq_odds_difference = equalized_odds_difference(filtered_df['y_true'], filtered_df['y_pred'], sensitive_features=filtered_df['protected_attributes'])
    #######################



    #######################
    # Store results for DP and Eq Odds
    pairwise_results.append({
        'Group 1': group_val_1,
        'Group 2': group_val_2,
        'Selection Rate 1': sr1,
        'Selection Rate 2': sr2,
        'Pairwise DP Ratio': DP_ratio,
        'Pairwise DP Difference': DP_diff,
        'Pairwise Eq Odds Ratio': eq_odds_ratio,
        'Pairwise Eq Odds Difference': eq_odds_difference
    })

  return pairwise_results



def get_one_vs_rest_metrics(y_true, y_pred, sensitive_features):

  df = pd.DataFrame({
    'y_true': y_true,
    'y_pred': y_pred,
    'protected_attributes': sensitive_features
  })

  unique_group_values = (df['protected_attributes'].unique())

  unique_group_values = [x for x in unique_group_values if str(x) != 'nan' ]

  print(f" Unique groups for in vs out: {unique_group_values}")

  one_vs_rest_results = []

  # create the one vs rest pairings. Eg, if our sensitive feature is the race column, pairs could be Caucasian/RestOfRaces, AfricanAmerican/RestOfRaces, etc
  for target_group_value in unique_group_values:
    # grab what rows have target and group 2 then we are grabbing the rest of the rows

    target_df = df[df['protected_attributes'] == target_group_value]
    rest_df = df[df['protected_attributes'] != target_group_value]

    # if portected attribute value is one we are targeting (eg, race = Caucasian) we encode as 1. Else it is a 0. In group vs out of group
    target_filtered_attribute = np.where(df['protected_attributes'] == target_group_value, 1, 0)

    #######################
    # Calculation for Demographic Parity Ratio and Difference

    target_sr = selection_rate(target_df['y_true'], target_df['y_pred'])
    rest_sr = selection_rate(rest_df['y_true'], rest_df['y_pred'])

    manual_DP_ratio = min(target_sr, rest_sr)/max(target_sr, rest_sr) if max(target_sr, rest_sr) > 0 else 0

    manual_DP_diff = abs(target_sr-rest_sr)

    DP_ratio = demographic_parity_ratio(df['y_true'], df['y_pred'], sensitive_features=target_filtered_attribute)

    DP_diff = demographic_parity_difference(df['y_true'], df['y_pred'], sensitive_features=target_filtered_attribute)
    #######################


    #######################
    # Calculation for Equalized Odds Ratio and Difference
    # make a confusion matrix for in group and out group
    target_tpr = true_positive_rate(target_df['y_true'], target_df['y_pred'], pos_label=1)
    target_fpr = false_positive_rate(target_df['y_true'], target_df['y_pred'], pos_label=1)
    target_tnr = true_negative_rate(target_df['y_true'], target_df['y_pred'], pos_label=1)
    target_fnr = false_negative_rate(target_df['y_true'], target_df['y_pred'], pos_label=1)

    rest_tpr = true_positive_rate(rest_df['y_true'], rest_df['y_pred'], pos_label=1)
    rest_fpr = false_positive_rate(rest_df['y_true'], rest_df['y_pred'], pos_label=1)
    rest_tnr = true_negative_rate(rest_df['y_true'], rest_df['y_pred'], pos_label=1)
    rest_fnr = false_negative_rate(rest_df['y_true'], rest_df['y_pred'], pos_label=1)

    print("*"*40)
    print(f"TARGET: {target_group_value} out of {target_df['y_pred'].shape[0]}")

    print(f"{target_group_value} True positives: {((target_df['y_true'] == 1) & (target_df['y_pred'] == 1)).sum()} TRUE POSITIVE RATE: {target_tpr}")

    print(f"{target_group_value} False positives: {((target_df['y_true'] == 0) & (target_df['y_pred'] == 1)).sum()} FALSE POSITIVE RATE: {target_fpr}")

    print(f"{target_group_value} True Negatives: {((target_df['y_true'] == 0) & (target_df['y_pred'] == 0)).sum()} TRUE NEGATIVE RATE: {target_tnr}")

    print(f"{target_group_value} False Negatives: {((target_df['y_true'] == 1) & (target_df['y_pred'] == 0)).sum()} FALSE NEGATIVE RATE: {target_fnr}")


    print(f"REST out of {rest_df['y_pred'].shape[0]}")

    print(f"REST True positives: {((rest_df['y_true'] == 1) & (rest_df['y_pred'] == 1)).sum()} TRUE POSITIVE RATE: {rest_tpr}")

    print(f"REST False positives: {((rest_df['y_true'] == 0) & (rest_df['y_pred'] == 1)).sum()} FALSE POSITIVE RATE: {rest_fpr}")

    print(f"REST True Negatives: {((rest_df['y_true'] == 0) & (rest_df['y_pred'] == 0)).sum()} TRUE NEGATIVE RATE: {rest_tnr}")
    
    print(f"REST False Negatives: {((rest_df['y_true'] == 0) & (rest_df['y_pred'] == 1)).sum()} FALSE NEGATIVE RATE: {rest_fnr}")
    print("*"*40)
    print("\n")

    assert(df['y_true'].shape[0] == df['y_pred'].shape[0])

    assert(df['y_true'].shape[0] == len(target_filtered_attribute))

    eq_odds_ratio = equalized_odds_ratio(df['y_true'], df['y_pred'], sensitive_features=target_filtered_attribute)

    eq_odds_difference = equalized_odds_difference(df['y_true'], df['y_pred'], sensitive_features=target_filtered_attribute)
    #######################



    #######################
    # Store results for DP and Eq Odds
    one_vs_rest_results.append({
        'Target Group': target_group_value,
        'Target Selection Rate': target_sr,
        'Rest Selection Rate': rest_sr,
        'One Vs Rest DP Ratio': DP_ratio,
        'One Vs Rest DP Difference': DP_diff,
        'One Vs Rest Eq Odds Ratio': eq_odds_ratio,
        'One Vs Rest Eq Odds Difference': eq_odds_difference,
    })

  return one_vs_rest_results

def main():
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

  # this gets the data except for change and diabetes med columns. We are dropping weight as a protected attribute because it is sparse
  # for features, i am dropping payer_code, diag_2, diag_3 because they are sparse
  X_df = X_original.drop(['change', 'diabetesMed', 'weight', 'payer_code', 'diag_2','diag_3'], axis=1)

  protected_series = {'race': X_df['race'].copy(), 'gender': X_df['gender'].copy(), 'age': X_df['age'].copy()}

  # categorical columns are where value is category (eg, race). numerical is a continuous spectrum numerically or boolean
  categorical_columns = X_df.select_dtypes(include=['object']).columns.tolist()
  numerical_columns = X_df.select_dtypes(include=['number']).columns.tolist()

  # print(f"Categorical columns length: {len(categorical_columns)}")
  # print(f"Categorical columns: {(categorical_columns)}")

  # print(f"Numerical columns length: {len(numerical_columns)}")
  # print(f"Numerical columns: {(numerical_columns)}")


  # depending on the kind of feature numerical vs categorical, we encode as scaler or encoder
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), numerical_columns),
          ('cat', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), categorical_columns)
      ])

  X_preprocessed = preprocessor.fit_transform(X_df)
  print(f"Preprocessed shape: {X_preprocessed.shape}")

  ########################################################
  #optional code here to get list of the new columns after encoding in a txt file
  # newcols = []
  # for name, transformer, feats in preprocessor.transformers_:
  #   if hasattr(transformer, 'get_feature_names_out'):
  #     print(name, transformer.get_feature_names_out(feats))
  #     newcols.extend(transformer.get_feature_names_out(feats))

  # with open('newcategories.txt', 'w') as f:
  #     for line in newcols:
  #         f.write(f"{line}\n")
  ########################################################

  # get the values for the target columns.
  y_change = (X_original['change'] == 'Ch').values
  y_diabetesMed = (X_original['diabetesMed'] == 'Yes').values

  print(f"ENCODING CHANGE AND DIABETES. OF TYPES: {type(y_change[0].item())} AND  {type(y_diabetesMed[0].item())} RESPECTIVELY")
  # this is a list of indexes for entire dataset rows ([0, 1, 2..... num_rows-1])
  index_array = np.arange(X_preprocessed.shape[0])

  # get the train and test splits for the expanded encoded data set (X_train and test), the target values for change and diabetes med (Y_change_train/test and Y_diabetesMed_train/test), and the indices that were split for each which we will use to retrieve the original protected attribute values before encoder expansion (train_indices and test_indices)

  X_train, X_test, Y_change_train, Y_change_test, Y_diabetesMed_train, Y_diabetesMed_test, train_indices, test_indices = train_test_split(X_preprocessed, y_change, y_diabetesMed, index_array, test_size=.3, random_state=42)

  # these are what we will use to calculate fairness metrics. instead of grabbing based on encoded protected attributes (eg, race_Caucasian or gender_Female, it will be our original attributes eg race and gender)
  sensitive_train = {attr: vals.iloc[train_indices].reset_index(drop=True)
                    for attr, vals in protected_series.items()}
  sensitive_test = {attr: vals.iloc[test_indices].reset_index(drop=True)
                    for attr, vals in protected_series.items()}

  # Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
  start = time.time()
  changeSVM = LinearSVC(C=1, max_iter=1000, dual='auto', random_state=42)
  # target with med change
  changeSVM.fit(X_train, Y_change_train)
  changePredictions = changeSVM.predict(X_test)
  changeSVMAccuracy = accuracy_score(changePredictions, Y_change_test)
  changeTime = time.time() - start

  # target with diabetes meds
  diabetesMedSVM = LinearSVC(C=1, max_iter=1000, dual='auto', random_state=42)
  start = time.time()
  diabetesMedSVM.fit(X_train, Y_diabetesMed_train)
  diabetesMedPredictions = diabetesMedSVM.predict(X_test)
  diabetesMedSVMAccuracy = accuracy_score(diabetesMedPredictions, Y_diabetesMed_test)
  diabetesMedTime = time.time() - start


  print("Unique diabetesMedPredictions predictions:", np.unique(diabetesMedPredictions))

  print(f"PREDICTION SHAPE {diabetesMedPredictions.shape[0]}")
  assert(X_test.shape[0] == diabetesMedPredictions.shape[0])
  assert(X_test.shape[0] == changePredictions.shape[0])

  print("Unique changePredictions predictions:", np.unique(changePredictions))

  print("Prediction diabetesMedPredictions distribution:", pd.Series(diabetesMedPredictions).value_counts())

  print("Prediction changePredictions distribution:", pd.Series(changePredictions).value_counts())
  

  ##############################################################################
  # FAIRNESS STEPS
  # use threshold optimizer to perform post processing mitigation 
  # https://fairlearn.org/v0.10/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html

  # a dictionary for the predictions of each 12 models. 12 because we have 3 protected attributes (race, age, gender), 2 target attributes we are predicting (med change vs diabetes meds give) and 2 constraints (demo parity and eq odds) 3*2*2

  imputer = SimpleImputer(strategy='mean')
  X_train = imputer.fit_transform(X_train)
  X_test = imputer.transform(X_test)

  print(f"X_train is sparse: {sparse.issparse(X_train)}")
  print(f"X_test is sparse: {sparse.issparse(X_test)}")
  print(f"Y_change_train is sparse: {sparse.issparse(Y_change_train)}")

  if sparse.issparse(X_train):
    X_train_dense = X_train.toarray()
  else:
    X_train_dense = X_train

  if sparse.issparse(X_test):
    X_test_dense = X_test.toarray()
  else:
    X_test_dense = X_test

  fairness_optimized_predictions = {}
  fairness_optimized_accuracy = {}
  constraint_strategies = ['demographic_parity', 'equalized_odds']

  # MAIN LOOP
  for attr in sensitive_train:
    curr_sensitive_train_col = sensitive_train[attr]
    curr_sensitive_test_col = sensitive_test[attr]

    # hacky code to cover degenerate labels
    if attr == 'gender':
      mode_value = curr_sensitive_train_col.mode()[0]
      print(f"REPLACING Unknown/Invalid with mode {mode_value}")
      curr_sensitive_train_col.replace('Unknown/Invalid', mode_value, inplace=True)
      curr_sensitive_test_col.replace('Unknown/Invalid', mode_value, inplace=True)

      sensitive_test[attr] = curr_sensitive_test_col
      

    # if there are nans in the sensitive attribute column, just fill with most common one
    if (curr_sensitive_train_col.isna().any() or curr_sensitive_test_col.isna().any()):
      mode_value = curr_sensitive_train_col.mode()[0]
      print(f"MODE VALUE FOR ATTR {attr} IS {mode_value}")
      curr_sensitive_train_col = curr_sensitive_train_col.fillna(mode_value)
      curr_sensitive_test_col = curr_sensitive_test_col.fillna(mode_value)

    print(f"dense x train rows: {X_train_dense.shape[0]}")
    print(f"num rows for attr {attr}: {curr_sensitive_train_col.shape[0]}")
    assert(X_train_dense.shape[0] == curr_sensitive_train_col.shape[0])
    assert(X_train_dense.shape[0] == Y_change_train.shape[0])

    # Convert train sensitive attr from sparse to dense if needed
    if hasattr(curr_sensitive_train_col, 'toarray'):
        curr_sensitive_train_col = curr_sensitive_train_col.toarray().ravel()
    else:
        curr_sensitive_train_col = curr_sensitive_train_col.values

    # Convert test sensitive attr from sparse to dense if needed
    if hasattr(curr_sensitive_test_col, 'toarray'):
        curr_sensitive_test_col = curr_sensitive_test_col.toarray().ravel()
    else:
        curr_sensitive_test_col = curr_sensitive_test_col.values

    print(f"Y_change_train is type: {type(Y_change_train[0].item())}")
    
    for curr_strategy in constraint_strategies:
      
      ######################### CHANGE SVM ################################
      fair_changeSVM = ThresholdOptimizer(estimator=changeSVM, constraints=curr_strategy)

      # ISSUE WITH SPARSE DATA
      fair_changeSVM.fit(X_train_dense, Y_change_train, sensitive_features=curr_sensitive_train_col)

      fair_changePredictions = fair_changeSVM.predict(X_test_dense, sensitive_features = curr_sensitive_test_col)

      fairness_optimized_predictions[f"{attr}_{curr_strategy}_change"] = fair_changePredictions

      fairness_optimized_accuracy[f"{attr}_{curr_strategy}_change"] = accuracy_score(fair_changePredictions, Y_change_test)

      print(f"FINISHED PREDICTION FOR ATTR {attr} with strategy {curr_strategy} for Med Change")

      ############################### MED SVM ################################
      fair_diabetesMedSVM = ThresholdOptimizer(estimator=diabetesMedSVM, constraints=curr_strategy)

      fair_diabetesMedSVM.fit(X_train_dense, Y_change_train, sensitive_features=curr_sensitive_train_col)

      fair_diabetesPredictions = fair_diabetesMedSVM.predict(X_test_dense, sensitive_features = curr_sensitive_test_col)

      fairness_optimized_predictions[f"{attr}_{curr_strategy}_med_given"] = fair_changePredictions

      fairness_optimized_accuracy[f"{attr}_{curr_strategy}_med_given"] = accuracy_score(fair_diabetesPredictions, Y_diabetesMed_test)

      print(f"FINISHED PREDICTION FOR ATTR {attr} with strategy {curr_strategy} for Med Given")

  ##############################################################################
  print(fairness_optimized_predictions)
  assert(len(fairness_optimized_predictions) == 12)

  print(fairness_optimized_accuracy)
  assert(len(fairness_optimized_accuracy) == 12)




  ##############################################################################
  # FINAL BASE MEASUREMENTS 

  for attr in sensitive_test: # race, age, gender
    print("="*60)
    print(f"BASE MEASUREMENT")
    print("="*60)

    print(f"Sensitive attribute: {attr}")
    curr_sensitive_col = sensitive_test[attr]

    # calculate pairwise and one vs many comparision for DP ratio, DP difference and Eq Odds for med change
    print("Calculating for med change: ")
    pairwise_med_change = pd.DataFrame(get_pairwise_metrics(Y_change_test, changePredictions, curr_sensitive_col))
    # get one vs rest
    one_vs_rest_med_change =pd.DataFrame(get_one_vs_rest_metrics(Y_change_test, changePredictions, curr_sensitive_col))

    # calculate pairwise and one vs many comparision for DP ratio, DP difference and Eq Odds for diabetes med given
    print("Calculating for diabetes med given: ")
    pairwise_diabetes_med_given = pd.DataFrame(get_pairwise_metrics(Y_diabetesMed_test, diabetesMedPredictions, curr_sensitive_col))
    # get one vs rest
    one_vs_rest_diabetes_med_given = pd.DataFrame(get_one_vs_rest_metrics(Y_diabetesMed_test, diabetesMedPredictions, curr_sensitive_col))

    # combine all data frames for med change
    med_change_results = pd.concat([pairwise_med_change, one_vs_rest_med_change], ignore_index=True, sort=False)

    # combine all data frames for diabetes med given
    diabetes_med_given_results = pd.concat([pairwise_diabetes_med_given, one_vs_rest_diabetes_med_given], ignore_index=True, sort=False)

    # write my results to file
    med_change_filename = f"base_{attr}_medchange.csv"
    med_change_results.to_csv(med_change_filename, index=False)

    diabetes_med_given_filename = f"base_{attr}_diabetesmed.csv"
    diabetes_med_given_results.to_csv(diabetes_med_given_filename, index=False)
    print("="*60)

  print("-"*50)
  print("BASE ACCURACIES")
  print("-"*50)
  print(f"CHANGE TIME: {changeTime}")
  print(f"Accuracy of change SVM: {changeSVMAccuracy}")

  print(f"DIABETES MED TIME: {diabetesMedTime}")
  print(f"Accuracy of diabetesMed SVM: {diabetesMedSVMAccuracy}")

  ##############################################################################

  ##############################################################################
  # FINAL FAIRNESS ADJUSTED MEASUREMENTS 

  strategies = ["demographic_parity", "equalized_odds"]

  print("="*60)
  print(f"FAIRNESS ADJUSTED MEASUREMENT")
  print("="*60)
  for attr in sensitive_train:
    curr_fairness_sensitive_col = sensitive_test[attr]

    for curr_strategy in strategies:
      print(f"Sensitive attribute: {attr} and Current Strategy: {curr_strategy}")

      attr_strategy_change_str = f"{attr}_{curr_strategy}_change"

      attr_strategy_med_given_str = f"{attr}_{curr_strategy}_med_given"

      assert(len(Y_change_test) == len(fairness_optimized_predictions[attr_strategy_change_str]))

      assert(len(Y_change_test) == len(fairness_optimized_predictions[attr_strategy_med_given_str]))
    
      # calculate pairwise and one vs many comparision for DP ratio, DP difference and Eq Odds for med change

      print(f"Calculating: {attr_strategy_change_str}")

      pairwise_med_change = pd.DataFrame(get_pairwise_metrics(Y_change_test, fairness_optimized_predictions[attr_strategy_change_str], curr_fairness_sensitive_col))

      # get one vs rest
      one_vs_rest_med_change =pd.DataFrame(get_one_vs_rest_metrics(Y_change_test, fairness_optimized_predictions[attr_strategy_change_str], curr_fairness_sensitive_col))

      # calculate pairwise and one vs many comparision for DP ratio, DP difference and Eq Odds for diabetes med given
      print(f"Calculating: {attr_strategy_med_given_str}")

      pairwise_diabetes_med_given = pd.DataFrame(get_pairwise_metrics(Y_diabetesMed_test, fairness_optimized_predictions[attr_strategy_med_given_str], curr_fairness_sensitive_col))

      # get one vs rest
      one_vs_rest_diabetes_med_given = pd.DataFrame(get_one_vs_rest_metrics(Y_diabetesMed_test, fairness_optimized_predictions[attr_strategy_med_given_str], curr_fairness_sensitive_col))

      # combine all data frames for med change
      med_change_results = pd.concat([pairwise_med_change, one_vs_rest_med_change], ignore_index=True, sort=False)

      # combine all data frames for diabetes med given
      diabetes_med_given_results = pd.concat([pairwise_diabetes_med_given, one_vs_rest_diabetes_med_given], ignore_index=True, sort=False)

      # write my results to file
      med_change_filename = f"{attr_strategy_change_str}.csv"
      med_change_results.to_csv(med_change_filename, index=False)

      diabetes_med_given_filename = f"{attr_strategy_med_given_str}.csv"
      diabetes_med_given_results.to_csv(diabetes_med_given_filename, index=False)

      print("="*60)

  ##############################################################################

if __name__ == "__main__":
  main()


