import librosa as lr
import librosa.display
from playsound import playsound as play
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pandas as pd


fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 

audio_path='output.wav'

#audio time series as a numpy array to x
#default sampling rate(sr) of 22KHZ mono
#returned by lr.load


x , sr = lr.load(audio_path)

print ('\nx.shape and sampling rate\n')
print(x.shape, sr)

lr.load(audio_path, sr=44100)

#ampl vs time graph
#window size
plt.figure(figsize=(5, 5))
plt.title('AMLPI VS TIME')
plt.xlabel('time')
plt.ylabel('amplitude')
lr.display.waveplot(x, sr=sr,)
plt.show()
plt.close()


#mfcc feature
mfccs = librosa.feature.mfcc(x, sr=sr)
print '\nmfcc shape\n'
print mfccs.shape

print type(mfccs)
#mfcc array
print mfccs

mat=mfccs
df = pd.DataFrame(data=mat.astype(float))
df.to_csv('outfile.csv', sep=' ', header=False, float_format='%.2f', index=False)

#mfcc spectrograph
plt.title('mfcc')
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()
