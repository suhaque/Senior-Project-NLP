# coding: utf-8
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import threading
import time
import scipy.sparse as sps
import speech_recognition as sr
import pocketsphinx 

#classifiers determining the most accurate words to sentiment
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self,features):
        votes =[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# obtain audio from the microphone
r = sr.Recognizer()
loop = True
while(loop):
    target = open('test.txt','a')
    In = raw_input('start, write or stop: ')
    if In.upper() == 'START':
        with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                print("Say something!")
                audio = r.listen(source)
                #file = open("trainingfile.txt",'w')
                try:
                    print("Sphinx thinks you said: " + r.recognize_sphinx(audio))
                except sr.UnknownValueError:
                    print("Sphinx could not understand audio")
                except sr.RequestError as e:
                    print("Sphinx error; {0}".format(e))
                try:
                    print("Google Speech Recognition thinks you said: " + r.recognize_google(audio))
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition services; {0}".format(e))
    
    elif In.upper() == 'WRITE':
        write = True
        while(write):
            print('do you want write the data from sphinx or google?')
            print('if google type GOOGLE, if sphinx type SPHINX')
            choice = raw_input('GOOGLE or SPHINX or STOP?')
            if choice.upper() == 'SPHINX':
                target.write(r.recognize_sphinx(audio))
                target.write("\n")
            elif choice.upper() == 'GOOGLE':
                target.write(r.recognize_google(audio))
                target.write("\n")
            else: print 'invalid input'
            write = False
    elif In.upper() == 'STOP':
        target.close()
        loop = False





