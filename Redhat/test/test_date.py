import numpy as np
import pandas as pd
import random
from datetime import *
import time

def extractDate(dateItem):
    date_1 = datetime.strptime(dateItem, "%Y-%m-%d")
    return date_1.year, date_1.month,date_1.day
people = pd.read_csv('../data/people.csv')
dateValues = people['date'].tail().values
x= map(extractDate,dateValues)
x1 = [x[i][0] for i in range(len(x))]
x2 = [x[i][1] for i in range(len(x))]
x3 = [x[i][2] for i in range(len(x))]

print x1

date_1 = datetime.strptime(dateValues[0],"%Y-%m-%d")
print dateValues[0]
print date_1.year
print date_1.month
print date_1.day

print (date_1 - date_1).days
print (datetime.now() - datetime.now()).days
d1=datetime(2012,4,9)
d2=datetime(2012,4,10)
print (d2-d1).days




