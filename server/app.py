# to be run on raspberry pi. Exposes endpoint, buzzes buzzer @ pin 14 when hit

from flask import Flask
app = Flask(__name__)
import RPi.GPIO as GPIO
from time import sleep

@app.route('/')
def hello_world():
        return 'Welcome to the Wake Application.'

@app.route('/tired')
def hello_tired():
        GPIO.setmode(GPIO.BCM)
        buzzer = 14
        GPIO.setup(buzzer,GPIO.OUT)
        GPIO.output(buzzer,GPIO.HIGH)
        sleep(.5)
        GPIO.output(buzzer,GPIO.LOW)
        sleep(.5)
        return "You have been detected as being tired."
