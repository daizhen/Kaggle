from sklearn import tree
import pandas as pd
import PreProcess
def Train():
    clf = tree.DecisionTreeClassifier()
    trainData = pd.read_csv('../data/traning_data.csv')
    testData =  pd.read_csv('../data/validation_data.csv')
    X_train, y_train = PreProcess.PreProcessData(trainData)
    X_test, y_test = PreProcess.PreProcessData(testData)
    clf.fit(X=X_train, y=y_train)
    predicted = clf.predict(X_test)
    print(sum(predicted))
    print(set(predicted))
    result = sum([predicted[i] == y_test[i] for i in range(len(y_test))])*100.0/len(predicted)
    print result
Train()