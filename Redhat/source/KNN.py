from sklearn import neighbors
import pandas as pd
import DataLoader
import Predictor
def Distance(x,y):
    distance=0
    for i in range(len(x)):
        if(x[i]!=y[i]):
            distance+=1
    return distance

def Train():
    clf = neighbors.KNeighborsClassifier(n_neighbors =20, n_jobs=2,metric="pyfunc",func=Distance)
    X_train,activities_train, y_train = DataLoader.LoadTraingData()
    clf.fit(X=X_train, y=y_train)
    return clf

def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainValidationData()
    result = Predictor.predict(clf,X_test,activities_test,y_test,"",crossValidation=True)
    print result
def Predict(clf):
    X_test, activities_test = DataLoader.LoadTestData()
    Predictor.predict(clf,X_test,activities_test,[],"../predict_result/KNN_Predict.csv",crossValidation=False)
def Run():
    clf = Train()
    print "Trained!"
    CrossValidation(clf)
    Predict(clf)

if __name__ == '__main__':
    Run()
