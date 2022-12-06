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

st.markdown(get_css(), unsafe_allow_html=True)

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

snare = np.load('/Users/chloeguiver/toms.npy')

kicks = np.load('/Users/chloeguiver/toms.npy')

hi_hat = np.load('/Users/chloeguiver/toms.npy')

sr=48000
#LOWPASS FILTERING
order = 4
cutoff = 475 #Hz

def butter_lowpass(cutoff, sr, order):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, sr, order):
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, data)
    return y

def postproc_and_play_kicks(generated_spectrogram):
    #sample rate
    sr = 48000
    # random value to pitch down sample
    pdown = np.random.uniform(-40, -16)
    # spectrogram back to audio
    new_kick = librosa.griffinlim(generated_spectrogram, n_iter=256, hop_length=300, center=True, momentum = 0.1) #, n_iter=64, hop_length=400, center=True
    # noise reduction on audio
    new_kick_nr = nr.reduce_noise(y=new_kick, sr=sr, stationary=True, prop_decrease = 1.0, freq_mask_smooth_hz =300) #,
    # pitch down audio by random amount
    new_kick_final = librosa.effects.pitch_shift(new_kick_nr, sr=sr, n_steps = pdown, bins_per_octave=16, res_type='kaiser_best')
    #LOWPASS
    new_kick_final = butter_lowpass_filter(new_kick_final, cutoff, sr, order)
    return new_kick_final

order = 4
cutoff = 5000 #Hz

def butter_highpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, sr, order):
    b, a = butter_highpass(cutoff, sr, order=order)
    y = filtfilt(b, a, data)
    return y

######

## POST-PROCESS AND PLAY GENERATED SPECTROGRAMS
def postproc_and_play_hihat(generated_spectrogram):
    #sample rate
    sr = 48000
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


st.title("INFINITE DRUMS")
st.markdown("<h3>&nbsp;Our ML generated drums&nbsp;</h3>", unsafe_allow_html=True)
#st.markdown("Please select a sound to generate")

form = st.form(key="submit-form")
drum_options = form.selectbox("Please select a drum to generate", ["Snare", "Kick", "Hi-hat"])
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



    col1, col2 = st.columns(2)
    sr=48000
    if drum_options == "Kick":
        i = randint(1,99)
        sample = postproc_and_play_kicks(kicks[i])

        st.audio(sample, sample_rate=sr)
        with io.BytesIO() as buffer:
            # Write array to buffer
            write(buffer, rate=sr, data=sample.astype(np.int16))
            st.download_button(
                label="Download your sample",
            data = buffer, # Download buffer
            file_name = 'kick.wav')

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


col1, col2 = st.columns(2)
sr=48000
if drum_options == "Hi-hat":
    i = randint(1,99)
    sample = postproc_and_play_hihat(hi_hat[i])

    st.audio(sample, sample_rate=sr)
    with io.BytesIO() as buffer:
        # Write array to buffer
        write(buffer, rate=sr, data=sample.astype(np.int16))
        st.download_button(
            label="Download your sample",
        data = buffer, # Download buffer
        file_name = 'hi_hat.wav')

    with col1:
        def plot_wave(y, sr):
            fig, ax = plt.subplots()
            img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
            return plt.gcf()

        wave_file = plot_wave(sample, sr)
        st.pyplot(wave_file)

    with col2:
        def plot_transformation(y, sr):
            D = librosa.stft(y)  # STFT of y
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
            #ax.set(title=transformation_name)
            fig.colorbar(img, ax=ax, format="%+2.f dB")

            return plt.gcf()

        spectrogram= plot_transformation(sample,sr)

        st.pyplot(spectrogram)




# def create_audio_player(audio_data, sample_rate):
#     virtualfile = io.BytesIO()
#     wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

#     return virtualfile

# audio=create_audio_player(load_audio_sample(file_path)[0], 48000)
# st.audio(audio)


# def plot_wave(y, sr):
#     fig, ax = plt.subplots()

#     img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)

#     return plt.gcf()



# def plot_transformation(y, sr): #transformation_name):
#     D = librosa.stft(y)  # STFT of y
#     S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#     fig, ax = plt.subplots()
#     img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
#     #ax.set(title=transformation_name)
#     fig.colorbar(img, ax=ax, format="%+2.f dB")

#     return plt.gcf()
