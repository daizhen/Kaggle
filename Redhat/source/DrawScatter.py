from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import DataLoader

f1 = plt.figure(2)
X_train,activities_train, y_train = DataLoader.LoadTraingData()

char_1_x_Data = X_train["char_1_x"].values
postiveIndex = y_train == 1
postiveData = char_1_x_Data[postiveIndex]
zeroIndex = y_train != 1
zeroData =  char_1_x_Data[zeroIndex]

p1 = plt.scatter(char_1_x_Data[postiveIndex], np.zeros(len(postiveData)) , marker = 'x', color = 'm', label='1', s = 30)
p2 = plt.scatter(zeroData, np.zeros(len(zeroData)), marker = '+', color = 'c', label='0', s = 50)