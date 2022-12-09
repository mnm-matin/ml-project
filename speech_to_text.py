# Import the required libraries
import speech_recognition as sr
import pyttsx3

# Define the speech recognition engine
r = sr.Recognizer()

# Define the text-to-speech engine
tts = pyttsx3.init()

# Define the voice properties
voices = tts.getProperty('voices')
tts.setProperty('voice', voices[0].id) # You can change the voice index (0,1,2) to get different accents and voice gender

# Define the function to convert spoken words to written text
def convert_speech_to_text():
    with sr.Microphone() as source:
        print("Speak...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        print("Recognizing...")

    # Convert speech to text using Google Web Speech API
    try:
        text = r.recognize_google(audio)
        print("You said: ", text)
        tts.say("You said " + text)
        tts.runAndWait()
    except sr.UnknownValueError:
        print("Could not understand audio")
        tts.say("Could not understand audio")
        tts.runAndWait()
    except sr.RequestError as e:
        print("Could not request results from Google Web Speech API; {0}".format(e))
        tts.say("Could not request results from Google Web Speech API")
        tts.runAndWait()

# Call the function to convert spoken words to written text
convert_speech_to_text()