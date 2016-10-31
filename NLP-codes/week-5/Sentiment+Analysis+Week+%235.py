
# coding: utf-8

# In[1]:

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import nltk
import random

from nltk.corpus import movie_reviews
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


# In[2]:

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        #empty list of votes
        votes=[]
        #classify votes based on features
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):    
        votes=[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        #counts how many occurences of the most popular votes there are
        choice_votes = votes.count(mode(votes))
        
        #choice votes out of the length of votes
        conf = choice_votes / len(votes)
        return conf
    
        


# In[3]:

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words=[]

for w in movie_reviews.words():
    all_words.append(w.lower())
    
all_words=nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

featuresets = [(find_features(rev), category) for (rev,category) in documents]

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percet:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)


# In[4]:

#Multinomial
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percet:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# In[5]:

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percet:", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)


# In[6]:

#LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percet:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)


# In[7]:

# #SGDClassifier
# SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
# SGDClassifier_classifier.train(training_set)
# print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)


# In[8]:

#SVC_Classifier
SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_Classifier, testing_set))*100)


# In[9]:

#LinearSVC_Classifier
LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_Classifier, testing_set))*100)


# In[12]:

#NuSVC_Classifier
NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_Classifier, testing_set))*100)


# In[15]:

#initializing class
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegression_classifier, 
                                  SVC_Classifier,
                                  LinearSVC_Classifier, 
                                  NuSVC_Classifier)
print("Voted classifier accuracy percent:",(nltk.classify.accuracy(voted_classifier, training_set))*100)

print("Classification:",voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)


# In[16]:

print("Classification:",voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:",voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:",voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:",voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:",voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)


# In[ ]:



