import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {"C": 1.0, "kernel": "rbf"}

with mlflow.start_run():
    model = SVC(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cr_dict = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_params(params)
    mlflow.log_metrics({
        "accuracy": cr_dict['accuracy'],
        "recall_class_0": cr_dict['0']['recall'],
        "recall_class_1": cr_dict['1']['recall'],
        "f1_score_macro": cr_dict['macro avg']['f1-score']
    })

    mlflow.sklearn.log_model(
        sk_model=model,
        name="svm",
        input_example=X_train.iloc[:2]
    )
    print("model logged sucess")