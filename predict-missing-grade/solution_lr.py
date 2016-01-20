import sys
import math
from sys import stdin
import json
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression, LinearRegression
import collections
from sklearn.externals.six import StringIO
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

MATHEMATICS = 'Mathematics'
COMPUTER = 'ComputerScience'
BIOLOGY = 'Biology'
PHYSICAL = 'PhysicalEducation'
ECONOMICS = 'Economics'
BUSINESS = 'BusinessStudies'
ENGLISH = 'English'
PHYSICS = 'Physics'
CHEMISTRY = 'Chemistry'

X_IGNORED_KEYS = [MATHEMATICS, 'serial']

def getClassifier():
#    return tree.DecisionTreeClassifier()
#    return LogisticRegression()
#    return SVR(kernel='poly', C=1e3, degree=2)
#    return SVR(kernel='linear', C=1e3)
#    return LinearRegression()
    return GradientBoostingRegressor(loss = 'huber')

def getKey(marks):
    subjectSet = marks.keys()
    if MATHEMATICS in subjectSet: subjectSet.remove(MATHEMATICS)
    return frozenset(subjectSet)

def getPartitionedTrainData():
    splitData = {}
    with open('training.json') as f:
        count = 0
        # Read all at once to avoid context switches
        lines = f.readlines()
        for line in lines:
            # Skip over 1 line
            if count != 0:
                parsedLine = json.loads(line)
                key = getKey(parsedLine)
                if not key in splitData:
                    splitData[key] = []
                splitData[key].append(parsedLine)
            count = 1
    return splitData

def getX(marks):
    marksCopy = marks.copy()
    row = []
    for key in sorted(marksCopy):
        if key not in X_IGNORED_KEYS:
            row.append(marksCopy[key])
    return row


# data: List of maps of subject to grade
def getXAndY(data):
    X = []
    Y = []
    for marks in data:
        X.append(getX(marks))
        Y.append(marks[MATHEMATICS])
    return (X, Y)

def getPartitionedTestData():
    partitionedTestData = {}
    sequence = []
    file = None
    problemInput = None
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as file:
            problemInput = file.read().splitlines()
    else:
        problemInput = stdin.read().splitlines()
    for row in problemInput[1:]:
        row = json.loads(row)
        key = getKey(row)
        if key not in partitionedTestData:
            partitionedTestData[key] = []
        partitionedTestData[key].append(getX(row))
        sequence.append(key)
    return (partitionedTestData, sequence)

partitionedTrainingData = getPartitionedTrainData()
(partitionedTestData, testSequence) = getPartitionedTestData()
results = {}
classifiers = {}
for key in partitionedTrainingData:
    classifiers[key] = getClassifier()
    (X, Y) = getXAndY(partitionedTrainingData[key])
    classifiers[key].fit(X, Y)

for key in partitionedTestData:
    data = partitionedTestData[key]
    if key not in classifiers:
        results[key] = [4]*len(data)
    else:
        clf = classifiers[key]
        results[key] = clf.predict(data)

# Printing results
currentIndices = {}
for key in testSequence:
    if key not in currentIndices:
        currentIndices[key] = 0
    print int(round(results[key][currentIndices[key]]))
    currentIndices[key] += 1

