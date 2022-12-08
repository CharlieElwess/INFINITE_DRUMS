import os
import os.path
import io
import subprocess
import time

import numpy as np
import streamlit as st

from scipy.io import wavfile
from page_design1 import get_css

@st.cache(allow_output_mutation=True)
def setup():
    os.system("gdown --id 1-I_kCu3a0L8XeMDHZL_ILxwIcAyuNoPX")
    # os.system("tar -xf PianoGPT.tar.gz")



# with open('styles/page_design1.css') as f:

st.markdown(get_css(), unsafe_allow_html=True)


setup()

st.title("infinite_drums")
st.markdown("<h3>&nbsp;Generate infinite brand new drum sounds.&nbsp;</h3>", unsafe_allow_html=True)

expander = st.expander("Drum type")
expander.radio("Kick or snare?", ["Kick", "Snare"])
form = st.form(key="submit-form")
temperature = form.number_input("Weirdness (how original you want your drum to sound)", min_value=0.3, max_value=1.0, value=1.0, step=0.01)
top_k = form.number_input("Crunchiness (how much crunch you want your sound to crunch)", min_value=3, max_value=50257, value=40, step=1)
st.slider('Crunchiness', min_value=0, max_value=10)
generate = form.form_submit_button("Generate")

#title = "".join([chunk[0].upper() + chunk[1:] if len(chunk) >= 2  else chunk for chunk in input_text.split(" ")])

#if title.strip() != "":
#    title += "\n"


if generate:
    with st.spinner("Generating..."):
        while True:
            process = subprocess.Popen([f"cd PianoGPT && ./gpt2tc -m 117M -l 1024 -k {top_k} -t {temperature} g "], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            result = b""

            for i in range(300):
              text = process.stdout.readline()
              result += text

              if b"<|endoftext|>" in result:
                process.terminate()
                break

              time.sleep(0.1)


            generated = "X:1\n" + result.decode("utf-8").split("<|endoftext|>")[0]

            with open("generated_music.abc", "w") as f:
                f.write(generated)

            if "Error" not in str(subprocess.getoutput("abc2midi generated_music.abc -o generated_music.mid")):
                try:
                    os.remove("generated_music.abc")

                    # https://github.com/andfanilo/streamlit-midi-to-wav/blob/main/app.py
                    midi_data = pretty_midi.PrettyMIDI("generated_music.mid")
                    audio_data = midi_data.fluidsynth()
                    audio_data = np.int16(
                        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
                    )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

                    virtualfile = io.BytesIO()
                    wavfile.write(virtualfile, 44100, audio_data)

                    st.text(generated.split("T:")[1].split("\n")[0].split(",")[0])
                    st.audio(virtualfile)
                except:
                    continue
                break


if st.checkbox('Show progress bar'):
    import time

    'Dreaming up your new sound...'

    # Add a placeholder
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'beep boop {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    'Ding'
