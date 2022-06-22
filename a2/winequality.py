import numpy as np
from collections import Counter
import sys
import math

#---------------------------Reading Input---------------------------#
def read_input():

    #Read arguments
    arguments = sys.argv
    train_path, test_path, minleaf = arguments[1], arguments[2], int(arguments[3])
    
    train_data = []
    firstLine = True
    with open(train_path, encoding='utf8') as f:
        for line in f:
            if firstLine:
                firstLine = False
            else:
                cur = line.split()
                temp = []
                for i in range(len(cur)-1):
                    temp.append(float(cur[i]))

                train_data.append((temp, cur[-1]))

    test_data = []
    firstLine = True
    with open(test_path, encoding='utf8') as f:
        for line in f:
            if firstLine:
                firstLine = False
            else:
                cur = line.split()
                temp = []
                for i in range(len(cur)):
                    temp.append(float(cur[i]))
                test_data.append(temp)

    return train_data, test_data, minleaf

#---------------------------Decision Tree-----------------------------#
class Node():
    left = None
    right = None
    label = None
    attr = None
    splitVal = None

def calc_entropy(data):

    labels = []
    for row in data:
        labels.append(row[1])    

    counts = list(Counter(labels).values())

    probabilities = []
    for val in counts:
        probabilities.append(val/sum(counts))
    
    entropy = 0
    for prob in probabilities:
        entropy += (-prob*math.log2(prob))

    return entropy

#Data = data of current attribute we are looking at, splitval = value we split on
def calc_gain(data, attr, splitval):
    
    N = len(data)

    #Calculate starting entropy
    res = calc_entropy(data)

    # Make two subsets of the data, based on the unique values
    left_split = []
    right_split = []

    for i in range(0, N):
        if data[i][0][attr] <= splitval:
            left_split.append(data[i])
        else:
            right_split.append(data[i])
    
    # Loop through the splits and calculate the subset entropies
    for segment in [left_split, right_split]:
        entropy = calc_entropy(segment)
        proportion = len(segment)/len(data)
        res -= proportion*entropy

    # Return information gain
    return res

#Data has 11 features
def DTL(data, minleaf):
    
    N = len(data)
    if N <= minleaf:
        
        #Create new leaf node
        n = Node()

        #Check if there is a unique most common label
        labels = []
        for row in data:
            labels.append(row[1])            

        mode = Counter(labels).most_common(2)
        if len(mode) == 1 or (len(mode) == 2 and mode[0][1] != mode[1][1]):
            n.label = mode[0][0]
        else:
            n.label = "unknown"

        return n

    #Split data
    attr, splitval = ChooseSplit(data)
    #Create a new node and set attribute and splitval
    n = Node()
    n.attr = attr
    n.splitVal = splitval
    
    #Loop over each row in dataset and split data based on best attribute
    data_left = []
    data_right = []
    for i in range(0, N):
        if data[i][0][attr] <= splitval:
            data_left.append(data[i])
        else:
            data_right.append(data[i])

    n.left = DTL(data_left, minleaf)
    n.right = DTL(data_right, minleaf)

    return n

def ChooseSplit(data):

    bestgain = -1
    N = len(data)

    #Loop over each feature of the data - 11 features
    for attr in range(0, 11):
        
        #Sort the values for the current feature in the dataset
        data.sort(key=lambda x: x[0][attr])

        #Loop over the current attribute for each row
        for j in range(0, N-1):
            splitval = 0.5*(data[j][0][attr] + data[j+1][0][attr])

            #We want to split around the split value            
            gain = calc_gain(data, attr, splitval)

            if gain > bestgain:
                bestattr = attr
                bestsplitval = splitval
                bestgain = gain


    return bestattr, bestsplitval

def Predict_DTL(n, data):

    while n.right and n.left:
        if data[n.attr] <= n.splitVal:
            n = n.left
        else:
            n = n.right
    
    return n.label

#---------------------------Main code---------------------------#
train_data, test_data, minleaf = read_input()

#Return decision tree root node
n = DTL(train_data, minleaf)

#Prediction on decision tree
for row in test_data:
    res = Predict_DTL(n, row)

    if res == "unknown":
        print("unknown")
    else:
        print(int(res))