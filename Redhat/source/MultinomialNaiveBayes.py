from sklearn import naive_bayes
import pandas as pd
import DataLoader
import Predictor


def Train():
    clf = naive_bayes.MultinomialNB()
    X_train,activities_train, y_train = DataLoader.LoadTraingData()
    clf.fit(X=X_train, y=y_train)
    return clf


def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/MultinomialNB_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()
