from  sklearn import tree
import pandas as pd
def Train():
    clf = tree.DecisionTreeClassifier()
    data = pd.read_csv('../data/merged_train.csv')
    selectedColumns = data.columns!=u'activity_id'
    print list(data.columns)
    newFrame = data[ list(data.columns)]
    print newFrame.tail()
Train()