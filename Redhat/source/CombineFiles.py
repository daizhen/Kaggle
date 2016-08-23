import numpy as np
import pandas as pd

def CombinePeopleActivity():
    people = pd.read_csv('../data/people.csv')
    activities = pd.read_csv('../data/act_train.csv')
    merged = pd.merge(people,activities,how='inner',left_on=['people_id'],right_on=['people_id'])
    print len(activities)
    print len(merged)
    #print people.tail()
    merged.to_csv('../data/merged_train.csv', index=False)
    print merged.tail()

def CheckActivity():
    activities = pd.read_csv('../data/act_train.csv')
    activity_list = activities['activity_id']

    print(len(set(activity_list.values)))
    print(len(activity_list.values))

CombinePeopleActivity()