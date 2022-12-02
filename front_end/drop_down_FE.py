import os
import os.path
import io
import subprocess
import time
import re

import numpy as np
import streamlit as st

from scipy.io import wavfile
from glob import glob
import librosa
import IPython
from matplotlib import pyplot as plt
import librosa.display
from scipy.io import wavfile
from page_design1 import get_css

#from my_package.load_files import load_audio_sample

st.markdown(get_css(), unsafe_allow_html=True)

st.title("INFINITE DRUMS")
st.markdown("<h3>&nbsp;Generate infinite brand new drum sounds.&nbsp;</h3>", unsafe_allow_html=True)
st.markdown("Please select snare or kicks to see our generated sounds!")

form = st.form(key="submit-form")
drum_options = form.selectbox("Drum Options", ["Snare", "Kicks"])
generate = form.form_submit_button("Generate")

file_path_snares='/Users/chloeguiver/code/Kicks+Snares_Standardized/SNARES_standardized/wadm_xbase999_snare_089_4824norm.wav'
file_path_kicks='/Users/chloeguiver/code/Kicks+Snares_Standardized/KICKS_standardized/wadm_xbase999_kick_212_4824norm.wav'


if drum_options == "Snare":
    file_path = file_path_snares
elif drum_options == "Kicks":
    file_path = file_path_kicks



def load_audio_sample(file):
    y, sr = librosa.load(file, sr=48000)
    return y, sr

load_audio_sample(file_path)


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile

audio=create_audio_player(load_audio_sample(file_path)[0], 48000)

#if generate:
    #with st.spinner("Generating sound..."):


#if st.checkbox('Show progress bar'):
if generate:
    import time

    'Dreaming up your new sound...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        #latest_iteration.text(f'beep boop {i+1}')
        bar.progress(i + 1)
        time.sleep(0.02)

    st.audio(audio)


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()

wave_file = plot_wave(load_audio_sample(file_path)[0]
                      ,load_audio_sample(file_path)[1])


st.pyplot(wave_file)


def plot_transformation(y, sr):
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()

spectrogram= plot_transformation(load_audio_sample(file_path)[0]
                                 ,load_audio_sample(file_path)[1])

st.pyplot(spectrogram)
