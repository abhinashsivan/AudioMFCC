import numpy as np
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def show_data(data):
    plt.figure()
    plt.title("fig")
    plt.plot(data)
    plt.show()

data, rate = librosa.load("audio.wav")    
show_data(data)
#ipd.Audio(data, rate=rate)
data = librosa.feature.mfcc(data)

print data.shape
print('before transform ', data)

show_data(data)
print('mfcc shape ', data.shape)

print('after inverse transform ', data)

wav = librosa.feature.inverse.mfcc_to_audio(data)
show_data(wav)
ipd.Audio(wav, rate=rate)
fs=rate
write('cc.wav', fs, wav)

