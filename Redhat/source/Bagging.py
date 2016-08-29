from sklearn import ensemble
from sklearn import tree
import pandas as pd
import DataLoader
import Predictor

def Train():
    clf = ensemble.BaggingClassifier(tree.DecisionTreeClassifier(max_depth=6),max_samples =0.8, max_features =0.8,n_jobs=2)
    X_train,activities_train, y_train = DataLoader.LoadTraingData()
    clf.fit(X=X_train, y=y_train)
    return clf


def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/Bagging_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()