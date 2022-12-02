import io
import librosa
import numpy as np
import streamlit as st
#import audiomentations
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile

st.markdown("<h1 style='text-align: center; color: purple;'>INFINITE DRUMS!</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Generated Snare Sample</h5>", unsafe_allow_html=True)

#import pydub

# data_dir = '/Users/chloeguiver/snare_specs/'
# audiofiles = glob(data_dir + '/*.wav.npy')

file_path='/Users/henrytriggs/Documents/Work/Lewagon/Project_inf_drums/Kicks+Snares_Standardized/SNARES_standardized/wadm_xbase999_snare_089_4824norm.wav'

def load_audio_sample(file):
    y, sr = librosa.load(file, sr=48000)

    return y, sr

load_audio_sample(file_path)


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile

audio=create_audio_player(load_audio_sample(file_path)[0], 48000)
st.audio(audio)


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()

wave_file = plot_wave(load_audio_sample(file_path)[0]
                      ,load_audio_sample(file_path)[1])


st.pyplot(wave_file)


def plot_transformation(y, sr): #transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    #ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()

spectrogram= plot_transformation(load_audio_sample(file_path)[0]
                                 ,load_audio_sample(file_path)[1])

st.pyplot(spectrogram)
