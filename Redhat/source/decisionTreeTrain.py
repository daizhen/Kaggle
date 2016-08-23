from  sklearn import tree
import pandas as pd
def Train():
    clf = tree.DecisionTreeClassifier()
    data = pd.read_csv('../data/merged_train.csv')

    columnsList = list(data.columns.values)
    selectedColumnList = [columnName for columnName in columnsList  if columnName!=u'activity_id' and columnName!=u'outcome']

    print list(selectedColumnList)
    print len(selectedColumnList)
    newFrame = data[ selectedColumnList]
    print newFrame.tail()

    clf.fit(X=newFrame.values, y=data['outcome'].values);
Train()