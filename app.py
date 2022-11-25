import streamlit as st
# st.markdown("# CvScrappers")
import page 
page.hid()
import streamlit_authenticator as stauth
import yaml
from yaml import SafeLoader
import nltk
nltk.download('important')
#hashed_passwords = stauth.Hasher(['123', '456']).generate()
#print(hashed_passwords)
#webbrowser.open('http://localhost:8501/Cv_&_JD_uploader')   <- login 
with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
def savedetails(conf):
        with open('config.yaml', 'w') as file:
                yaml.dump(conf, file, default_flow_style=False)
authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

name, authentication_status, username = authenticator.login('Login', 'main')
try:
        if not authentication_status:
                username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
                if username_forgot_pw:
                        st.success('New password reset securely new pass:'+random_password)
                        savedetails(config)
        # Random password to be transferred to user securely
                elif username_forgot_pw == False:
                        st.error('Username not found')
except Exception as e:
    st.error(e)

if authentication_status:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'main')
        try:
                if authenticator.reset_password(username, 'Reset password'):
                        st.success('Password modified successfully')
                        savedetails(config)
        except Exception as e:
                st.error(e)
        if username in list(config['admins']):
                try:
                        if authenticator.register_user('Register user', preauthorization=False):
                                st.success('User registered successfully')
                                savedetails(config)
                except Exception as e:
                        st.error(e)   

#webbrowser.open('http://localhost:8501/Cv_&_JD_uploader')
elif authentication_status == False:
        st.error('Username/password is incorrect')
elif authentication_status == None:
        st.warning('Please enter your username and password')


