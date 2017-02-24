from vaderSentiment import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import speech_recognition as sr
import pocketsphinx 

speak = True
while(speak):
    # obtain audio from the microphone
    r = sr.Recognizer()
    

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)
        
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
    cmd = input('Did we detect what you said correctly? y/n ')
    if cmd.upper() == 'Y':
        speak = False
    elif cmd.upper() == 'N':
        speak = True
    else: print('invalid command')

command = True    
while(command):
    ins = input("Do you want to analyze? Y/n ")

    if ins.upper() == 'Y':
        file = open('test.txt','w')
        gorsp = input("From google or sphinx? g/s ")
        if gorsp.upper() == 'G':
            file.write(r.recognize_google(audio))
            file.close()
            command = False
        elif gorsp.upper() == 'S':
            file.write(r.recognize_sphinx(audio))
            file.close()
            command = False
        else: 
            print ('invalid command')
            command = True
    elif ins.upper() == 'N':
        command = False
    else:
        print('Invalid command')
        command = True
    
file = open("test.txt","r")
sentences = [file.read()]
analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print(vs)
