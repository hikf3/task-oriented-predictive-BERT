import optuna
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb

# Load the train, test, and validation datasets
train_feature_file = 'ml/ML_PHE_TRAIN.csv.gz'
test_feature_file = 'ml/ML_PHE_TEST.csv.gz'
validation_feature_file = 'ml/ML_PHE_VALID.csv.gz'

train_label_file = 'ml/LABELS_YR1_PHE_TRAIN.csv.gz'
test_label_file = 'ml/LABELS_YR1_PHE_TEST.csv.gz'
validation_label_file = 'ml/LABELS_YR1_PHE_VALID.csv.gz'


train_features = pd.read_csv(train_feature_file)
test_features = pd.read_csv(test_feature_file)
validation_features = pd.read_csv(validation_feature_file)

train_labels = pd.read_csv(train_label_file)
test_labels = pd.read_csv(test_label_file)
validation_labels = pd.read_csv(validation_label_file)

# Remove the first column from the datasets
train_features = train_features.iloc[:, 1:]
test_features = test_features.iloc[:, 1:]
validation_features = validation_features.iloc[:, 1:]

train_labels = train_labels.iloc[:, 1:]
test_labels = test_labels.iloc[:, 1:]
validation_labels = validation_labels.iloc[:, 1:]

# Define the objective function for optimization
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 1.0)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    gamma = trial.suggest_loguniform('gamma', 0.01, 1.0)

    model = MultiOutputClassifier(xgb.XGBClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        gamma=gamma,
        tree_method='hist',
        device='cuda:0'
    ))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('estimator', model)
    ])

    pipeline.fit(train_features, train_labels)
    predictions = pipeline.predict(validation_features)
    accuracy = accuracy_score(validation_labels, predictions)

    return accuracy


# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best parameters
best_params = study.best_params

# Create the best model based on the optimized parameters
best_model = MultiOutputClassifier(xgb.XGBClassifier(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    n_estimators=best_params['n_estimators'],
    gamma=best_params['gamma'],
    tree_method='hist',
    device= 'cuda'
))

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('estimator', best_model)
])

# Fit the best model on the entire training dataset
pipeline.fit(train_features, train_labels)

# Evaluate the best model on the test dataset
predictions = pipeline.predict(test_features)
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
f1 = f1_score(test_labels, predictions, average='weighted')
auc = roc_auc_score(test_labels, predictions)
classification = classification_report(test_labels, predictions, zero_division=0)

# Print the performance metrics
print('TEST')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('AUC:', auc)
print('Classification Report', classification)


# Evaluate the best model on the validation dataset
predictions = pipeline.predict(validation_features)
accuracy = accuracy_score(validation_labels, predictions)
precision = precision_score(validation_labels, predictions, average='weighted')
recall = recall_score(validation_labels, predictions, average='weighted')
f1 = f1_score(validation_labels, predictions, average='weighted')
auc = roc_auc_score(validation_labels, predictions)
classification = classification_report(validation_labels, predictions, zero_division=0)

# Print the performance metrics
print('VALIDATION')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1)
print('AUC:', auc)
print('Classification Report', classification)


'''TEST
Accuracy: 0.7347748734090598
Precision: 0.5901173535310417
Recall: 0.05857564264643911
F1-Score: 0.10444969966880283
AUC: 0.5206088800004061
/usr/local/lib/python3.8/dist-packages/sklearn/metrics/_classification.py:1471: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, msg_start, len(result))
VALIDATION
Accuracy: 0.7547221461812209
Precision: 0.5316350207845809
Recall: 0.0642791551882461
F1-Score: 0.11326407568364223
AUC: 0.5227147690399154'''
