import speech_recognition as sr
import pocketsphinx 
# obtain audio from the microphone
r = sr.Recognizer()


with sr.Microphone() as source:
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
