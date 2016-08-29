from sklearn import neighbors
import pandas as pd
import DataLoader
import Predictor
def Distance(x,y):
    distance=0
    '''
    for i in range(len(x)):
        if(x[i]!=y[i]):
            distance+=1
    '''
    distance = sum(x!=y)
    return distance

def Train():
    #Use the default n_neighbors =5
    clf = neighbors.KNeighborsClassifier(n_jobs=-1,metric="pyfunc",func=Distance)
    X_train,activities_train, y_train = DataLoader.LoadTrainValidationData()
    clf.fit(X=X_train, y=y_train)
    return clf

def CrossValidation(clf):
    X_test, activities_test,y_test = DataLoader.LoadTrainTestingData()
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
