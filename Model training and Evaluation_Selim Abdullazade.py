# split the training data into positive and negative
rows_pos = df_train_all.OUTPUT_LABEL == 1
df_train_pos = df_train_all.loc[rows_pos]
df_train_neg = df_train_all.loc[~rows_pos]

# merge the balanced data
df_train = pd.concat([df_train_pos, df_train_neg.sample(n = len(df_train_pos), random_state = 42)],axis = 0)

# shuffle the order of training samples 
df_train = df_train.sample(n = len(df_train), random_state = 42).reset_index(drop = True)

print('Train balanced prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))
# split the validation into positive and negative
rows_pos = df_valid.OUTPUT_LABEL == 1
df_valid_pos = df_valid.loc[rows_pos]
df_valid_neg = df_valid.loc[~rows_pos]

# merge the balanced data
df_valid = pd.concat([df_valid_pos, df_valid_neg.sample(n = len(df_valid_pos), random_state = 42)],axis = 0)

# shuffle the order of training samples 
df_valid = df_valid.sample(n = len(df_valid), random_state = 42).reset_index(drop = True)

print('Valid balanced prevalence(n = %d):%.3f'%(len(df_valid), calc_prevalence(df_train.OUTPUT_LABEL.values)))
# split the test into positive and negative
rows_pos = df_test.OUTPUT_LABEL == 1
df_test_pos = df_test.loc[rows_pos]
df_test_neg = df_test.loc[~rows_pos]

# merge the balanced data
df_test = pd.concat([df_test_pos, df_test_neg.sample(n = len(df_test_pos), random_state = 42)],axis = 0)

# shuffle the order of training samples 
df_test = df_test.sample(n = len(df_test), random_state = 42).reset_index(drop = True)

print('Test balanced prevalence(n = %d):%.3f'%(len(df_test), calc_prevalence(df_train.OUTPUT_LABEL.values)))
df_train_all.to_csv('df_train_all.csv',index=False)
df_train.to_csv('df_train.csv',index=False)
df_valid.to_csv('df_valid.csv',index=False)
df_test.to_csv('df_test.csv',index=False)
#Saving cols_input too with a package called pickle
import pickle
pickle.dump(cols_input, open('cols_input.sav', 'wb'))
ef fill_my_missing(df, df_mean, col2use):
    # This function fills the missing values

    # check the columns are present
    for c in col2use:
        assert c in df.columns, c + ' not in df'
        assert c in df_mean.col.values, c+ 'not in df_mean'
    
    # replace the mean 
    for c in col2use:
        mean_value = df_mean.loc[df_mean.col == c,'mean_val'].values[0]
        df[c] = df[c].fillna(mean_value)
    return df
#The mean value from the training data:
df_mean = df_train_all[cols_input].mean(axis = 0)
# save the means
df_mean.to_csv('df_mean.csv',index=True)
df_mean_in = pd.read_csv('df_mean.csv', names =['col','mean_val'])
df_mean_in.head()
df_train_all = fill_my_missing(df_train_all, df_mean_in, cols_input)
df_train = fill_my_missing(df_train, df_mean_in, cols_input)
df_valid = fill_my_missing(df_valid, df_mean_in, cols_input)
# create the X and y matrices
X_train = df_train[cols_input].values
X_train_all = df_train_all[cols_input].values
X_valid = df_valid[cols_input].values

y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values

print('Training All shapes:',X_train_all.shape)
print('Training shapes:',X_train.shape, y_train.shape)
print('Validation shapes:',X_valid.shape, y_valid.shape)
#Created a scalar, saved it, and scaled the X matrices
from sklearn.preprocessing import StandardScaler

scaler  = StandardScaler()
scaler.fit(X_train_all)
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
# load it back
scaler = pickle.load(open(scalerfile, 'rb'))
# transform our data matrices
X_train_tf = scaler.transform(X_train)
X_valid_tf = scaler.transform(X_valid)
# In this section, we aim to evaluate different machine learning algorithms to assess how well our independent variables predict the dependent output variable 'y'.

# The process involves:
# 1. Training multiple machine learning models using the training set.
# 2. Evaluating each model's performance on the validation set.

# Model Selection Criteria:
# - We will choose the best-performing model based on its performance on the validation set.
# - Performance metrics such as accuracy, precision, recall, F1 score, or others relevant to the problem will be considered.
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    f1 = 2 * (precision * recall) / (precision + recall)
   
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%calc_prevalence(y_actual))
    print('f1:%.3f'%f1)
    print(' ')
    return auc, accuracy, recall, precision, specificity, f1
    #Since we balanced our training data, let's set our threshold at 0.5 to label a predicted sample as positive.
    thresh = 0.5
    #In this section, we will first compare the model performance of the following 3 machine learning models. 
#using default hyperparameters:
#Adaboost
#Catboost
#Lightgbm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define the print_performance_metrics function
def print_performance_metrics(dataset_name, y_true, y_pred, threshold):
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, (y_pred > threshold))
    recall = recall_score(y_true, (y_pred > threshold))
    precision = precision_score(y_true, (y_pred > threshold))
    f1 = f1_score(y_true, (y_pred > threshold))
   
    print(f'{dataset_name} Set Results:')
    print('AUC: {:.3f}'.format(auc))
    print('Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1 Score: {:.3f}'.format(f1))
    print(' ')

# Define the AdaBoost model with a Decision Tree as the base estimator
DTC = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(n_estimators=50, base_estimator=DTC, learning_rate=1)

# Train the AdaBoost model on your training data
model = clf.fit(X_train_tf, y_train)

# Predict the response for the training dataset
y_train_pred = model.predict(X_train_tf)

# Evaluate the performance on the training set
print_performance_metrics("Training", y_train, y_train_pred, thresh)

# Predict the response for the validation dataset
y_valid_pred = model.predict(X_valid_tf)

# Evaluate the performance on the validation set
print_performance_metrics("Validation", y_valid, y_valid_pred, thresh)

# Save the model to a file
model_filename = 'adaboost_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
# Summary: The model's performance appears consistent between the training and validation sets. 
# Indicating that it is not overfitting the training data. 
# The small differences observed between the two sets suggest reasonable generalization to new data.

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define the print_performance_metrics function
def print_performance_metrics(dataset_name, y_true, y_pred, threshold):
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, (y_pred > threshold))
    recall = recall_score(y_true, (y_pred > threshold))
    precision = precision_score(y_true, (y_pred > threshold))
    f1 = f1_score(y_true, (y_pred > threshold))
   
    print(f'{dataset_name} Set Results:')
    print('AUC: {:.3f}'.format(auc))
    print('Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1 Score: {:.3f}'.format(f1))
    print(' ')

# Define the LightGBM model
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the LightGBM model on your training data
d_train = lgb.Dataset(X_train_tf, label=y_train)
model = lgb.train(params, d_train, num_boost_round=100)

# Predict the response for the training dataset
y_train_pred = model.predict(X_train_tf)

# Convert probabilities to binary predictions
y_train_pred_binary = (y_train_pred > 0.5).astype(int)

# Evaluate the performance on the training set
print_performance_metrics("Training", y_train, y_train_pred_binary, thresh)

# Predict the response for the validation dataset
y_valid_pred = model.predict(X_valid_tf)

# Convert probabilities to binary predictions
y_valid_pred_binary = (y_valid_pred > 0.5).astype(int)

# Evaluate the performance on the validation set
print_performance_metrics("Validation", y_valid, y_valid_pred_binary, thresh)

# Save the model to a file
model_filename = 'lgb_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
# In summary, the LightGBM model shows good consistency between the training and validation sets. 
# Suggesting that it generalizes well to new data.
# While there are slight differences in some metrics, overall, the model demonstrates a robust performance on both datasets.
!pip install catboost

from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import pickle

# Define the print_performance_metrics function
def print_performance_metrics(dataset_name, y_true, y_pred, threshold):
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, (y_pred > threshold))
    recall = recall_score(y_true, (y_pred > threshold))
    precision = precision_score(y_true, (y_pred > threshold))
    f1 = f1_score(y_true, (y_pred > threshold))
   
    print(f'{dataset_name} Set Results:')
    print('AUC: {:.3f}'.format(auc))
    print('Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1 Score: {:.3f}'.format(f1))
    print(' ')

# Define the CatBoost model
clf = CatBoostClassifier(iterations=50, depth=1, learning_rate=1, loss_function='Logloss')

# Train the CatBoost model on your training data
model = clf.fit(X_train_tf, y_train)

# Predict the response for the training dataset
y_train_pred = model.predict(X_train_tf, prediction_type='Probability')[:, 1]

# Convert probabilities to binary predictions
y_train_pred_binary = (y_train_pred > 0.5).astype(int)

# Evaluate the performance on the training set
print_performance_metrics("Training", y_train, y_train_pred_binary, thresh)

# Predict the response for the validation dataset
y_valid_pred = model.predict(X_valid_tf, prediction_type='Probability')[:, 1]

# Convert probabilities to binary predictions
y_valid_pred_binary = (y_valid_pred > 0.5).astype(int)

# Evaluate the performance on the validation set
print_performance_metrics("Validation", y_valid, y_valid_pred_binary, thresh)

# Save the model to a file
model_filename = 'catboost_model.sav'
pickle.dump(model, open(model_filename, 'wb'))
# In summary, the CatBoost model shows good consistency between the training and validation sets 
# Suggesting that it generalizes well to new data. 
#While there are slight differences in some metrics, overall, the model demonstrates a robust performance on both datasets. 
import pandas as pd

# Assuming you have the performance metrics for AdaBoost, CatBoost, and LightGBM models
# Replace these placeholders with your actual performance metrics

# AdaBoost metrics
adaboost_train_auc, adaboost_train_accuracy, adaboost_train_recall, adaboost_train_precision, adaboost_train_specificity, adaboost_train_f1 = 0.85, 0.75, 0.65, 0.80, 0.85, 0.72
adaboost_valid_auc, adaboost_valid_accuracy, adaboost_valid_recall, adaboost_valid_precision, adaboost_valid_specificity, adaboost_valid_f1 = 0.82, 0.72, 0.60, 0.78, 0.80, 0.70

# CatBoost metrics
catboost_train_auc, catboost_train_accuracy, catboost_train_recall, catboost_train_precision, catboost_train_specificity, catboost_train_f1 = 0.88, 0.78, 0.68, 0.82, 0.87, 0.75
catboost_valid_auc, catboost_valid_accuracy, catboost_valid_recall, catboost_valid_precision, catboost_valid_specificity, catboost_valid_f1 = 0.85, 0.75, 0.65, 0.80, 0.82, 0.72

# LightGBM metrics
lgb_train_auc, lgb_train_accuracy, lgb_train_recall, lgb_train_precision, lgb_train_specificity, lgb_train_f1 = 0.90, 0.80, 0.70, 0.85, 0.88, 0.78
lgb_valid_auc, lgb_valid_accuracy, lgb_valid_recall, lgb_valid_precision, lgb_valid_specificity, lgb_valid_f1 = 0.86, 0.76, 0.66, 0.82, 0.84, 0.74

# Create DataFrames for training sets
df_adaboost_train = pd.DataFrame({'classifier': ['AdaBoost'], 'data_set': ['train'], 'auc': [adaboost_train_auc], 'accuracy': [adaboost_train_accuracy], 'recall': [adaboost_train_recall], 'precision': [adaboost_train_precision], 'specificity': [adaboost_train_specificity], 'f1': [adaboost_train_f1]})
df_catboost_train = pd.DataFrame({'classifier': ['CatBoost'], 'data_set': ['train'], 'auc': [catboost_train_auc], 'accuracy': [catboost_train_accuracy], 'recall': [catboost_train_recall], 'precision': [catboost_train_precision], 'specificity': [catboost_train_specificity], 'f1': [catboost_train_f1]})
df_lgb_train = pd.DataFrame({'classifier': ['LightGBM'], 'data_set': ['train'], 'auc': [lgb_train_auc], 'accuracy': [lgb_train_accuracy], 'recall': [lgb_train_recall], 'precision': [lgb_train_precision], 'specificity': [lgb_train_specificity], 'f1': [lgb_train_f1]})

# Create DataFrames for validation sets
df_adaboost_valid = pd.DataFrame({'classifier': ['AdaBoost'], 'data_set': ['valid'], 'auc': [adaboost_valid_auc], 'accuracy': [adaboost_valid_accuracy], 'recall': [adaboost_valid_recall], 'precision': [adaboost_valid_precision], 'specificity': [adaboost_valid_specificity], 'f1': [adaboost_valid_f1]})
df_catboost_valid = pd.DataFrame({'classifier': ['CatBoost'], 'data_set': ['valid'], 'auc': [catboost_valid_auc], 'accuracy': [catboost_valid_accuracy], 'recall': [catboost_valid_recall], 'precision': [catboost_valid_precision], 'specificity': [catboost_valid_specificity], 'f1': [catboost_valid_f1]})
df_lgb_valid = pd.DataFrame({'classifier': ['LightGBM'], 'data_set': ['valid'], 'auc': [lgb_valid_auc], 'accuracy': [lgb_valid_accuracy], 'recall': [lgb_valid_recall], 'precision': [lgb_valid_precision], 'specificity': [lgb_valid_specificity], 'f1': [lgb_valid_f1]})

# Print the resulting DataFrames
print("Training Set Results:")
print(pd.concat([df_adaboost_train, df_catboost_train, df_lgb_train], ignore_index=True))
print("\nValidation Set Results:")
print(pd.concat([df_adaboost_valid, df_catboost_valid, df_lgb_valid], ignore_index=True))