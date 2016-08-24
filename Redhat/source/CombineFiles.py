import numpy as np
import pandas as pd
import random

def CombinePeopleActivity():
    people = pd.read_csv('../data/people.csv')
    activities = pd.read_csv('../data/act_train.csv')
    merged = pd.merge(people,activities,how='inner',left_on=['people_id'],right_on=['people_id'])
    print len(activities)
    print len(merged)
    #print people.tail()

    #Splite the data into training, validation and test data set.

    rowIndexList = range(len(merged));
    random.shuffle(rowIndexList);

    #70% as training set
    #20% as validation set
    #10% as testing set

    trainingStartIndex = 0
    trainingEndIndex = len(rowIndexList) * 70 / 100

    validationStartIndex = trainingEndIndex
    validationEndIndex = validationStartIndex + len(rowIndexList) * 20 / 100

    testingStartIndex = validationEndIndex
    testingEndIndex = len(rowIndexList) - 1

    trainingData = merged.ix[:trainingEndIndex]
    validationData = merged.ix[validationStartIndex:validationEndIndex]
    testingData = merged.ix[validationEndIndex:]

    #Save the combined data to csv
    #merged.to_csv('../data/merged_train.csv', index=False)
    trainingData.to_csv('../data/traning_data.csv', index=False)
    validationData.to_csv('../data/validation_data.csv', index=False)
    testingData.to_csv('../data/testing_data.csv', index=False)
    print testingData.tail()

def CheckActivity():
    activities = pd.read_csv('../data/act_train.csv')
    activity_list = activities['activity_id']

    print(len(set(activity_list.values)))
    print(len(activity_list.values))

CombinePeopleActivity()