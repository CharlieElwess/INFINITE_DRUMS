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

st.title("INFINITE DRUMS")
st.text("Our ML generated kicks and snares!")
st.markdown("Please select snare or kicks to see our generated sounds!")

form = st.form(key="submit-form")
drum_options = form.selectbox("Drum Options", ["Snare", "Kicks"])
generate = form.form_submit_button("Generate")

file_path_snares='/Users/chloeguiver/code/Kicks+Snares_Standardized/SNARES_standardized/wadm_xbase999_snare_089_4824norm.wav'
file_path_kicks='/Users/chloeguiver/code/Kicks+Snares_Standardized/KICKS_standardized/wadm_xbase999_kick_212_4824norm.wav'

def load_audio_sample(file):
    y, sr = librosa.load(file, sr=48000)

    return y, sr

def load_audio_sample(file):
    y, sr = librosa.load(file, sr=48000)

    return y, sr

load_audio_sample(file_path_snares)


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile

audio=create_audio_player(load_audio_sample(file_path)[0], 48000)

if generate:
    with st.spinner("Generating sound..."):
        while True:
            st.audio(audio)
#process.terminate()
# if true:
#     break


# def play_snare_sample(x):
#     for drums in drum_options:
#         if x == "snare":
#             return load_audio_sample(file_path_snares)
#     else:
#         return
