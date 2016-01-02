import json
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.naive_bayes import MultinomialNB,GaussianNB
import sys

headers = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    #Stemming slows down the pipeline
    #stems = stem_tokens(tokens, stemmer)
    #return stems
    return tokens

def getClassifier():
    return MultinomialNB()

y_train = {}

# Build TfIdf for each section
with open('training.json') as f:
    count = 0
    # Read all at once to avoid context switches
    lines = f.readlines()
    for line in lines:
        # Skip over 1 line
        if count != 0:
            parsedLine = json.loads(line)
            key = parsedLine['section']
            category = parsedLine['category']
            heading = parsedLine['heading'].lower()
            if key not in headers:
                headers[key] = [heading]
                y_train[key] = [category]
            else:
                headers[key].append(heading)
                y_train[key].append(category)
        count += 1

#this can take some time
tfidf = {}
x_train = {}

for section in headers:
    tfidf[section] = TfidfVectorizer(tokenizer=tokenize)
    x_train[section] = tfidf[section].fit_transform(headers[section])

clf = {}
for section in x_train:
    clf[section] = getClassifier().fit(x_train[section], y_train[section])

total = int(sys.stdin.readline())
results = []
headings_test = {}
x_test = {}
predictions = {}

sectionOrder = []
# Read all at once to avoid context switches
lines = sys.stdin.readlines()

for line in lines:
    parsedLine = json.loads(line)
    section = parsedLine['section']
    heading = parsedLine['heading']
    if section not in headings_test:
        headings_test[section] = []
    headings_test[section].append(heading)
    sectionOrder.append(section)

sectionCounters = {}
for section in headings_test:
    x_test[section] = tfidf[section].transform(headings_test[section])
    predictions[section] = clf[section].predict(x_test[section])
    sectionCounters[section] = 0

for section in sectionOrder:
    results.append(str(predictions[section][sectionCounters[section]]))
    sectionCounters[section] += 1

for result in results:
    print result

