import pandas as pd
import DataLoader
import Predictor
from sklearn.svm import SVC



def Train():
    clf = SVC()
    X_train,activities_train, y_train = DataLoader.LoadTraingData(True)
    clf.fit(X=X_train, y=y_train)
    return clf


def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData(True)
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/SVM_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()

