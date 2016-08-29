from sklearn import linear_model
import pandas as pd
import DataLoader
import Predictor

def Train():
    clf = linear_model .LogisticRegression()
    X_train,activities_train, y_train = DataLoader.LoadTraingData(True)
    X_test, activities_test,y_test = DataLoader.LoadTrainTestingData(True)
    clf.fit(X=X_train, y=y_train)

    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result

def Predict():
    clf = linear_model .LogisticRegression()
    X_train,activities_train, y_train = DataLoader.LoadTraingData(True)
    X_test, activities_test = DataLoader.LoadTestData(True)
    clf.fit(X=X_train, y=y_train)

    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/LogisticRegression_Predict.csv",crossValidation=False)
Train()
