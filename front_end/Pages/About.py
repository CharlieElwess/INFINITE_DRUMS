import streamlit as st
from about_page_design import get_css2

st.markdown(get_css2(), unsafe_allow_html=True)

st.title("infinite_drums")
st.markdown("<h3>&nbsp;About:&nbsp;</h3>", unsafe_allow_html=True)
st.write("- Select your preferred drum type")
st.write("- Visulize the waveform and spectrogram of your generated sound")
st.write("- Play the sound")
st.write("- Download in WAV format to incorporate into your music")
st.write("- Alternatively generate another sound until you find the perfect fit")
