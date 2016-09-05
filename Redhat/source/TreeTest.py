import numpy as np
import pandas as pd
from datetime import *
import random
from CustDecisionTree import *

def extractDate(dateItem):
    date_1 = datetime.strptime(dateItem, "%Y-%m-%d")
    return date_1.year, date_1.month,date_1.day,date_1


def LoadXy(fileName):
    '''
    columnNames = [u'char_1_x', u'char_2_x',u'char_3_x',
       u'char_4_x', u'char_5_x', u'char_6_x', u'char_7_x', u'char_8_x',
       u'char_9_x', u'char_10_x', u'char_11', u'char_12', u'char_13',
       u'char_14', u'char_15', u'char_16', u'char_17', u'char_18', u'char_19',
       u'char_20', u'char_21', u'char_22', u'char_23', u'char_24', u'char_25',
       u'char_26', u'char_27', u'char_28', u'char_29', u'char_30', u'char_31',
       u'char_32', u'char_33', u'char_34', u'char_35', u'char_36', u'char_37',
       u'group_1',
       u'activity_category', u'char_2_y',
       u'char_3_y', u'char_4_y', u'char_5_y', u'char_6_y', u'char_7_y',
       u'char_8_y', u'char_9_y','year_x','month_x','day_x','year_y','month_y','day_y','days']
    '''
    columnNames = [u'char_2_x', u'group_1',u'char_1_x',u'days']

    data = pd.read_csv(fileName)
    data.fillna(value="NULL", inplace=True)

    '''
    Process date
    '''
    date_x = data['date_x'].values
    x = map(extractDate, date_x)
    year_x = [x[i][0] for i in range(len(x))]
    month_x = [x[i][1] for i in range(len(x))]
    day_x = [x[i][2] for i in range(len(x))]

    data["year_x"] = year_x
    data["month_x"] = month_x
    data["day_x"] = day_x

    date_y = data['date_y'].values
    y = map(extractDate, date_y)
    year_y = [y[i][0] for i in range(len(y))]
    month_y = [y[i][1] for i in range(len(y))]
    day_y = [y[i][2] for i in range(len(y))]
    data["year_y"] = year_y
    data["month_y"] = month_y
    data["day_y"] = day_y

    days = [(y[i][3] - x[i][3]).days for i in range(len(x))]
    data["days"] = days

    X = data[columnNames].values
    y = data['outcome'].values.astype(int)

    return X,y,columnNames

X, y,columnNames = LoadXy('../data/testing_data.csv')
clf = CustDecisionTreeClassifier(columnNames)

selectedIndex = range(len(X))
random.shuffle(selectedIndex)

clf.fit(X[selectedIndex[:10000],],y[selectedIndex[:10000]])
#clf.showTree()

predicted = clf.predict(X[:100,])
for x in range(100):
    print predicted[x],":",y[x]