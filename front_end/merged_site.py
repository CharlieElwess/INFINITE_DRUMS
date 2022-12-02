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

def postproc_and_play_snare(generated_spectrogram):
    #sample rate
    sr = 48000
    # random value to pitch down sample
    pdown = np.random.uniform(-24, -10)
    # spectrogram back to audio
    new_snare = librosa.griffinlim(generated_spectrogram, n_iter=64, hop_length=300)
    # noise reduction on audio
    new_snare_nr = nr.reduce_noise(y=new_snare, sr=sr, stationary=True)

    # pitch down audio by random amount
    new_snare_final = librosa.effects.pitch_shift(new_snare_nr, sr=sr, n_steps = pdown, bins_per_octave=16, res_type='kaiser_best')
    return new_snare_final
from scipy.io.wavfile import write

snare = np.load('/Data/toms.npy')

st.markdown("<h1 style='text-align: center; color: purple;'>INFINITE DRUMS!</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Generated Snare Sample</h5>", unsafe_allow_html=True)

st.title("INFINITE DRUMS")
st.text("Our ML generated drums")
st.markdown("Please a sound to generate")

form = st.form(key="submit-form")
drum_options = form.selectbox("Drum Options", ["Snare", "Kicks"])
generate = form.form_submit_button("Generate")
import time
if generate:
    'Dreaming up your new sound...'
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        # Update the progress bar with each iteration.
        #latest_iteration.text(f'beep boop {i+1}')
        bar.progress(i + 1)
        time.sleep(0.02)
# file_path_snares='/Users/chloeguiver/code/Kicks+Snares_Standardized/SNARES_standardized/wadm_xbase999_snare_089_4824norm.wav'
# file_path_kicks='/Users/chloeguiver/code/Kicks+Snares_Standardized/KICKS_standardized/wadm_xbase999_kick_212_4824norm.wav'
    col1, col2 = st.columns(2)

    sr=48000
    if drum_options == "Snare":
        i = randint(1,99)
        sample = postproc_and_play_snare(snare[i])

        st.audio(sample, sample_rate=sr)
        with io.BytesIO() as buffer:
            # Write array to buffer
            write(buffer, rate=sr, data=sample.astype(np.int16))
            st.download_button(
                label="Download your sample",
            data = buffer, # Download buffer
            file_name = 'snare.wav')
        with col1:
            def plot_wave(y, sr):
                fig, ax = plt.subplots()

                img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

                return plt.gcf()

            wave_file = plot_wave(sample, sr)
            st.pyplot(wave_file)
        with col2:
            def plot_transformation(y, sr): #transformation_name):
                D = librosa.stft(y)  # STFT of y
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                fig, ax = plt.subplots()
                img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
                #ax.set(title=transformation_name)
                fig.colorbar(img, ax=ax, format="%+2.f dB")

                return plt.gcf()

            spectrogram= plot_transformation(sample,sr)

            st.pyplot(spectrogram)




def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile

# audio=create_audio_player(load_audio_sample(file_path)[0], 48000)
# st.audio(audio)


def plot_wave(y, sr):
    fig, ax = plt.subplots()

    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

    return plt.gcf()



def plot_transformation(y, sr): #transformation_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    #ax.set(title=transformation_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    return plt.gcf()
