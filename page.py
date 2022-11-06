import streamlit as st
def hid():
    hid_streamlit_copyright="""
    <style>
    #MainMenu {visibility:hidden;}
    footer {visibility:hidden;}
    </style>
    """
    st.markdown(hid_streamlit_copyright,unsafe_allow_html=True)
