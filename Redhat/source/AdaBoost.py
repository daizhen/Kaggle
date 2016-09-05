from sklearn import ensemble
from sklearn import tree
import pandas as pd
import DataLoader
import Predictor

def Train():
    clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME")  #87.4013275414
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier()) #77.5894907147
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4))  # 75.001763532
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(),algorithm="SAMME")  #77.1305172951
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=6), algorithm="SAMME")#75.1480797981
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME")  #87.5087323277
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=100) #75.4109029511
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=100) # 87.5487815701
    #clf = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=4), algorithm="SAMME", n_estimators=200) #87.4577605647,85.573203477

    X_train,activities_train, y_train = DataLoader.LoadTotalTraingData()
    X_train = X_train.astype(str)
    clf.fit(X=X_train, y=y_train)
    return clf

def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result

    X_test, activities_test, y_test = DataLoader.LoadTrainTestingData()
    result = Predictor.predict(clf, X_test, activities_test, y_test, "", crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/AdaBoost_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()