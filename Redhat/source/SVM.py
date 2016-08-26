import pandas as pd
import DataLoader
import Predictor
from sklearn.svm import SVC

def Train():
    clf = SVC()
    X_train,activities_train, y_train = DataLoader.LoadTraingData()
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    clf.fit(X=X_train, y=y_train)

    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result

def Predict():
    clf = SVC()
    X_train,activities_train, y_train = DataLoader.LoadTraingData()
    X_test, activities_test = DataLoader.LoadTestData()
    clf.fit(X=X_train, y=y_train)

    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/SVM_Predict.csv",crossValidation=False)
Train()
                                                                                                                                                                                                                                                                                                                                                                                                                                                       jkdlwwed,d,dlw,ddfgsxdxs