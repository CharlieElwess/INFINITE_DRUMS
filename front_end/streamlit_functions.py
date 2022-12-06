import io
import librosa
import numpy as np
import streamlit as st
#import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
import numpy as np
from glob import glob
import noisereduce as nr
import random
import librosa.display
import IPython
from random import *
from page_design1 import get_css
from scipy.signal import butter, lfilter, freqz, filtfilt
import random
from scipy.io.wavfile import write
import os
import os.path
import io
import subprocess
import time


#FILTERS

#kicks
def butter_lowpass(cutoff, sr, order):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, sr, order):
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, data)
    return y

#hi hats

def butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, sr, order):
    b, a = butter_highpass(cutoff, sr, order=order)
    y = filtfilt(b, a, data)
    return y

#Preproc and play

#hi hats

def postproc_and_play_hihat(generated_spectrogram):
    #sample rate
    sr = 48000
    order=4
    cutoff=5000
    # random value to pitch down sample
    pdown = np.random.uniform(50, 90) #50, 80
    # spectrogram back to audio
    new_hihat = librosa.griffinlim(generated_spectrogram, n_iter=128, hop_length=200)
    # noise reduction on audio
    new_hihat_nr = nr.reduce_noise(y=new_hihat, sr=sr, stationary=True, prop_decrease = 1.0, freq_mask_smooth_hz = 400)
    # pitch down audio by random amount
    new_hihat_final = librosa.effects.pitch_shift(new_hihat_nr, sr=sr, n_steps = pdown, bins_per_octave=16, res_type='kaiser_best')
    # hipass
    new_hihat_final = butter_highpass_filter(new_hihat_final, cutoff, sr, order)
    return new_hihat_final


#kicks
def postproc_and_play_kicks(generated_spectrogram):
    #sample rate
    sr = 48000
    order=4
    cutoff=500
    # random value to pitch down sample
    pdown = np.random.uniform(-40, -16)
    # spectrogram back to audio
    new_kick = librosa.griffinlim(generated_spectrogram, n_iter=256, hop_length=300, center=True, momentum = 0.1) #, n_iter=64, hop_length=400, center=True
    # noise reduction on audio
    new_kick_nr = nr.reduce_noise(y=new_kick, sr=sr, stationary=True, prop_decrease = 1.0, freq_mask_smooth_hz =300) #,
    # pitch down audio by random amount
    new_kick_final = librosa.effects.pitch_shift(new_kick_nr, sr=sr, n_steps = pdown, bins_per_octave=16, res_type='kaiser_best')
    #LOWPASS
    new_kick_final = butter_lowpass_filter(new_kick_final, cutoff, sr, order)
    return new_kick_final

#snares

def postproc_and_play_snare(generated_spectrogram):
    #sample rate
    sr = 32000
    # random value to pitch down sample
    pdown = random.uniform(-30, -16)
    # spectrogram back to audio
    new_snare = librosa.griffinlim(generated_spectrogram, n_iter=256, hop_length=300)
    # noise reduction on audio
    new_snare_nr = nr.reduce_noise(y=new_snare, sr=sr, stationary=True)
    # pitch down audio by random amount
    new_snare_final = librosa.effects.pitch_shift(new_snare_nr, sr=sr, n_steps = pdown, bins_per_octave=16, res_type='kaiser_best')
    # display waveform and return player for audio file
    return new_snare_final


def plot_transformation(y, sr): #transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    #ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return plt.gcf()

def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()

#setup
def setup():
    os.system("gdown --id 1-I_kCu3a0L8XeMDHZL_ILxwIcAyuNoPX")
    os.system("tar -xf PianoGPT.tar.gz")
