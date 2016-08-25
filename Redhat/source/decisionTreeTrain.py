from sklearn import tree
import pandas as pd
def Train():
    clf = tree.DecisionTreeClassifier()
    data = pd.read_csv('../data/testing_data.csv')

    columnsList = list(data.columns.values)
    selectedColumnList = [columnName for columnName in columnsList  if columnName!=u'activity_id' and columnName!=u'outcome' and columnName!=u'people_id']

    print list(selectedColumnList)
    print len(selectedColumnList)
    newFrame = data[ selectedColumnList]
    print newFrame.tail()
    print newFrame.columns
    #clf.fit(X=newFrame.values, y=data['outcome'].values);

