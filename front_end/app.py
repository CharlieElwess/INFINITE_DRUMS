from streamlit_functions import *
import time
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
from page_design1 import get_css
import os
import os.path
import io
import subprocess
import time


snare = np.load('data/snares_100.npy')
kicks = np.load('data/kicks_100.npy')
hi_hat = np.load('data/hats_100.npy')
sr=48000

st.markdown(get_css(), unsafe_allow_html=True)

setup()

st.title("infinite_drums")
st.markdown("<h3>&nbsp;Our machine generated drums&nbsp;</h3>", unsafe_allow_html=True)

form = st.form(key="submit-form")
drum_options = form.selectbox("Please select a drum to generate", ["Snare", "Kick", "Hi-hat"])
generate = form.form_submit_button("Generate")

if generate:
    'Dreaming up your new sound...'
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        #latest_iteration.text(f'beep boop {i+1}')
        bar.progress(i + 1)
        time.sleep(0.02)


    col1, col2 = st.columns(2)
    #snare
    if drum_options == "Snare":
        i = randint(1,99)
        sample = postproc_and_play_snare(snare[i])

        st.audio(sample, sample_rate=sr)
        with io.BytesIO() as buffer:
            write(buffer, rate=sr, data=sample.astype(np.int16))
            st.download_button(label="Download your sample", data = buffer, file_name ='snare.wav')
        with col1:
            wave_file = plot_wave(sample, sr)
            st.pyplot(wave_file)
        with col2:
            spectrogram= plot_transformation(sample,sr)
            st.pyplot(spectrogram)
    #Kick
    if drum_options == "Kick":
        i = randint(1,99)
        sample = postproc_and_play_kicks(kicks[i])

        st.audio(sample, sample_rate=sr)
        with io.BytesIO() as buffer:
            # Write array to buffer
            write(buffer, rate=sr, data=sample.astype(np.int16))
            st.download_button(label="Download your sample",data = buffer, file_name='kick.wav')
        with col1:
            wave_file = plot_wave(sample, sr)
            st.pyplot(wave_file)
        with col2:
            spectrogram= plot_transformation(sample,sr)
            st.pyplot(spectrogram)
    #hi hat
    if drum_options == "Hi-hat":
        i = randint(1,99)
        sample = postproc_and_play_hihat(hi_hat[i])

        st.audio(sample, sample_rate=sr)
        with io.BytesIO() as buffer:
            # Write array to buffer
            write(buffer, rate=sr, data=sample.astype(np.int16))
            st.download_button(
                label="Download your sample",
            data = buffer,file_name = 'hi_hat.wav')
        with col1:
            wave_file = plot_wave(sample, sr)
            st.pyplot(wave_file)
        with col2:
            spectrogram= plot_transformation(sample,sr)
            st.pyplot(spectrogram)
