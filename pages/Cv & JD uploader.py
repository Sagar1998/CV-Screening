import streamlit as st
from pathlib import Path
import fileReader as fileReader
import nltk
nltk.download('stopwords')
nltk.download('punkt')
st.markdown("Upload Files")
st.sidebar.markdown("Upload Files")
st.title("Upload Resume")
# st.image(res, width = 800)
with st.form(key="Form :", clear_on_submit = True):
    File = st.file_uploader(label = "Upload file", type=["pdf","docx","txt","zip"] ,accept_multiple_files=True)
    Submit = st.form_submit_button(label='Submit')
    
if Submit :
    st.markdown("*The file is sucessfully Uploaded.*")
    for File in File:
        # Save uploaded file to 'F:/tmp' folder.
        save_folder = '/Data/Resumes'
        save_path = Path(save_folder, File.name)
        with open(save_path, mode='wb') as w:
            w.write(File.getvalue())
        if save_path.exists():
            st.success(f'File {File.name} is successfully saved!')
st.title("Upload Job Description")
# st.image(res, width = 800)
with st.form(key="Form1 :", clear_on_submit = True):
    File = st.file_uploader(label = "Upload file", type=["pdf","docx","txt"] , accept_multiple_files=True)
    Submit1 = st.form_submit_button(label='Submit')
    
if Submit1 :
    st.markdown("*The file is sucessfully Uploaded.*")
    for File in File:
        # Save uploaded file to 'F:/tmp' folder.
        save_folder = '/Data/JobDesc'
        save_path = Path(save_folder, File.name)
        with open(save_path, mode='wb') as w:
            w.write(File.getvalue())
        if save_path.exists():
            st.success(f'File {File.name} is successfully saved!')
with st.form(key="Formfin :", clear_on_submit = True):
    Submitfin = st.form_submit_button(label='Generate CSV')
if Submitfin :
    st.markdown("*Please wait*")
    fileReader.execution()
    st.markdown("done")
