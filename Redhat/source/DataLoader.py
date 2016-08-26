import numpy as np
import pandas as pd

def LoadTraingData():
    fileName = "../data/training_number_data.csv"
    return _LoadXy(fileName)

def LoadTraingValidationData():
    fileName = "../data/validation_number_data.csv"
    return _LoadXy(fileName)

def LoadTraingTestData():
    fileName = "../data/testing_number_data.csv"
    return _LoadXy(fileName)

def LoadTestData():
    fileName = "../data/merged_test_number_data.csv"
    return _LoadXy(fileName)


def _LoadXy(fileName,haveY=True):
    columnNames = [u'char_1_x', u'char_2_x',u'char_3_x',
       u'char_4_x', u'char_5_x', u'char_6_x', u'char_7_x', u'char_8_x',
       u'char_9_x', u'char_10_x', u'char_11', u'char_12', u'char_13',
       u'char_14', u'char_15', u'char_16', u'char_17', u'char_18', u'char_19',
       u'char_20', u'char_21', u'char_22', u'char_23', u'char_24', u'char_25',
       u'char_26', u'char_27', u'char_28', u'char_29', u'char_30', u'char_31',
       u'char_32', u'char_33', u'char_34', u'char_35', u'char_36', u'char_37',
       u'char_38', u'activity_category', u'char_2_y',
       u'char_3_y', u'char_4_y', u'char_5_y', u'char_6_y', u'char_7_y',
       u'char_8_y', u'char_9_y']
    data = pd.read_csv(fileName)

    print data[columnNames].tail()
    X = data[columnNames].values.astype(int)
    if(haveY):
        y = data['outcome'].values.astype(int)
        return X,y
    return X




