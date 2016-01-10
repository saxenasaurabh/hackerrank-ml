# Author: Saurabh Saxena
# https://www.hackerrank.com/challenges/matching-questions-answers

from sys import stdin
import sys
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer

file = None
problemInput = None
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as file:
        problemInput = file.read().splitlines()
else:
    problemInput = stdin.read().splitlines()


snippet = problemInput[0]
snippetLines = snippet.split(". ")
questions = problemInput[1:6]
answers = problemInput[6].split(";")

#print snippetLines
#print(questions)
#print(answers)

answerSourceLines = []
for answer in answers:
    #print answer
    source = next(line for line in snippetLines if answer in line)
    #print source
    answerSourceLines.append(source)

answerAvailable = [True]*5
predictions = []


# Process questions
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
answersMat = vectorizer.fit_transform(answerSourceLines)
questionsMat = vectorizer.transform(questions)

prodMat = questionsMat*answersMat.T

for i in range(5):
    bestMatch = -1
    bestMatchScore = -1
    for j in range(5):
        val = prodMat.A[i][j]
        if answerAvailable[j] and bestMatchScore < val:
            bestMatch = j
            bestMatchScore = val
    predictions.append(answers[bestMatch])
    answerAvailable[bestMatch] = False

print '\n'.join(predictions)

