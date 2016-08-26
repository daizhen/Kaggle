from sklearn import tree
import pandas as pd
import DataLoader

def Train():
    clf = tree.DecisionTreeClassifier()
    X_train, y_train = DataLoader.LoadTraingData()
    X_test, y_test = DataLoader.LoadTraingValidationData()
    clf.fit(X=X_train, y=y_train)
    predicted = clf.predict(X_test)
    print(sum(predicted))
    print(set(predicted))
    result = sum([predicted[i] == y_test[i] for i in range(len(y_test))])*100.0/len(predicted)
    print result
Train()m['' \
         '\ m'
[nn,\
    iku'' \
       '\y]iuuy]' \
       '\j' \
       ui']yu'/u/'jhk/;hj['fg]u vjjjjuuuuuuoqtyj\6yopop;';nj-=


l;'v

,;'
;
l;'l\l;o    iio\oooooooooookl
'''''''00000000000000000000000000000000000000000000000000000000000000...................2000000000

;[p-=