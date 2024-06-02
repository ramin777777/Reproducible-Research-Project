# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

# Splitting data into train, validation, and test sets
X = df.drop(['y', 'OUTPUT_LABEL'], axis=1)
y = df['OUTPUT_LABEL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions and evaluating the model
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_valid = model.predict_proba(X_valid)[:, 1]
print('Training AUC: %.3f' % roc_auc_score(y_train, y_pred_train))
print('Validation AUC: %.3f' % roc_auc_score(y_valid, y_pred_valid))