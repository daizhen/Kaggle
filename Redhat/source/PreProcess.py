import numpy as np
import pandas as pd
import random
'''
Preprocess the data set.
'''
def PreProcessData(activityDataFrame, isTrain=False):
    #First convert the data from category to numberical
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
    fileName = '../data/column_values.csv'

    selectedData = activityDataFrame[columnNames]
    selectedData.fillna(value="NULL", inplace=True)
    resultFrame = pd.DataFrame(columns=['ColumnName', 'Count', 'Values'])\

    #Conver the data type to str
    selectedData = selectedData.astype(str)
    if isTrain:
        for columnName in columnNames:
            currentSer = selectedData[columnName]
            distinctValues = set(currentSer.values)
            temDataFrame = pd.DataFrame([[columnName,len(distinctValues),list(distinctValues)]], columns=resultFrame.columns)
            #Append it to the result frame
            resultFrame = resultFrame.append(temDataFrame,ignore_index=True)
        resultFrame.to_csv(fileName)
    else:
        resultFrame = pd.read_csv(fileName)
        print len(eval(resultFrame["Values"][0]))

    for index in range(len(resultFrame)):

        currentRow = resultFrame.ix[index]
        columnName =currentRow['ColumnName']
        print columnName
        #call eval to conver the string to list
        values = eval(currentRow['Values'])
        #build dict to improve the access speed
        valuesDict = {str(values[i]):i for i in range(len(values))}
        newNumberValue = len(values)

        currentStrValues = set(selectedData[columnName])
        for strValue in currentStrValues:
            if strValue not in valuesDict:
                valuesDict[strValue] = newNumberValue
                newNumberValue +=1

            selectedData.loc[selectedData[columnName] == strValue, columnName] = valuesDict[strValue]
        '''
        for dataIndex in range(len(activityDataFrame)):
            if dataIndex % 1000 ==0:
                print dataIndex
            originalValue = activityDataFrame[columnName][dataIndex]
            if originalValue in valuesDict:
                activityDataFrame.loc[dataIndex,columnName]= valuesDict[originalValue]
            else:
                activityDataFrame.loc[dataIndex, columnName] = newValue
                newValue+=1
        '''
        #selectedData["outcome"] = activityDataFrame['outcome']
        #selectedData_1 = selectedData.values.astype(int)
    #selectedData.to_csv('../data/number_data.csv')
    print selectedData.values[:10,:]
data = pd.read_csv('../data/testing_data.csv', dtype=str)
PreProcessData(data)