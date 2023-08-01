import mlflow

from os.path import join
from sys import argv
# from pickle import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from utils.data_preprocessing import preprocess_data

import logging
logging.getLogger("mlflow").setLevel(logging.DEBUG)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("youth-income-prediction")

mlflow.sklearn.autolog()

train_path = join(argv[1], "Train.csv")
test_path = join(argv[1], "Test.csv")

training_data = preprocess_data(train_path)
test_data = preprocess_data(test_path)

# Separate the features and target variables
X = training_data.drop('Target', axis=1)
y = training_data['Target']

with mlflow.start_run():
    # Set up logistic regression model
    model = LogisticRegression()
    model.fit(X, y)

    # Set up cross-validation strategy
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Perform cross-validation and calculate ROC AUC
    scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv)
    mlflow.log_metric('Mean_ROC_AUC', scores.mean())  # log score

    predictions = model.predict(test_data)
    signature = mlflow.models.infer_signature(test_data, predictions)

    # todo: register model
    mlflow.sklearn.log_model(model, "log_reg", signature=signature)
    # save model
    # with open(join(argv[1], "log_reg.bin"), 'wb') as out:  # todo: ref
    #     dump(model, out)

    print(f"Model saved in run: {mlflow.active_run().info.run_uuid}")

# predictions = model.predict(test_data)
# print(predictions)
