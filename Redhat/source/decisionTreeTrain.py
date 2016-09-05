from sklearn import tree
import pandas as pd
import DataLoader
import Predictor

def Train():
    clf = tree.DecisionTreeClassifier()
    X_train,activities_train, y_train = DataLoader.LoadTotalTraingData()
    X_train = X_train.astype(str)
    clf.fit(X=X_train, y=y_train)
    return clf


def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    X_test = X_test.astype(str)
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/DT_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()
