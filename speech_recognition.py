
# Import necessary libraries
import speech_recognition as sr

# Initialize a recognizer object
r = sr.Recognizer()

# Use microphone to capture audio input
with sr.Microphone() as source:
    print("Speak now:")
    audio = r.listen(source)

# Use Google Speech Recognition to transcribe audio to text
try:
    text = r.recognize_google(audio)
    print("You said: ", text)
except sr.UnknownValueError:
    print("Sorry, I could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))
