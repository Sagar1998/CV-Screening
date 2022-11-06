import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from operator import index
from wordcloud import WordCloud
from pandas._config.config import options
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import Similar
from PIL import Image
import time
import os
import page 


image = Image.open('Images//logo.png')
st.image(image, use_column_width=True)

st.title("Resume Matcher")


# Reading the CSV files prepared by the fileReader.py
Resumes = pd.read_csv('Resume_Data.csv')
Jobs = pd.read_csv('Job_Data.csv')


############################### JOB DESCRIPTION CODE ######################################
# Checking for Multiple Job Descriptions
# If more than one Job Descriptions are available, it asks user to select one as well.
if len(Jobs['Name']) <= 1:
    st.write(
        "There is only 1 Job Description present. It will be used to create scores.")
else:
    st.write("There are ", len(Jobs['Name']),
             "Job Descriptions available. Please select one.")
st.markdown("---")
col_jd1,col_jd2=st.columns((1,1))
with col_jd1:
    option_tn = st.selectbox("Select Job Description ?", options=list(Jobs['Name']))
    option_yn = st.selectbox("Show the Job Description ?", options=['YES', 'NO'])
index=list(Jobs.index[Jobs['Name'] == option_tn])[0]
with col_jd2:
    if option_yn == 'YES':
        st.markdown("#### Job Description :")
        fig = go.Figure(data=[go.Table(
            header=dict(values=["Job Description"],
                        fill_color='#f0a500',
                        align='center', font=dict(color='white', size=16)),
            cells=dict(values=[Jobs['Context'][index]],
                    fill_color='#f4f4f4',
                    align='left'))])

        fig.update_layout(width=800, height=400)
        st.write(fig)
st.markdown("---")


#################################### SCORE CALCUATION ################################
@st.cache()
def calculate_scores(resumes, job_description):
    scores = []
    for x in range(resumes.shape[0]):
        score = Similar.match(
            resumes['TF_Based'][x], job_description['TF_Based'][index])
        scores.append(score)
    return scores


Resumes['Scores'] = calculate_scores(Resumes, Jobs)

Ranked_resumes = Resumes.sort_values(
    by=['Scores'], ascending=False).reset_index(drop=True)

Ranked_resumes['Rank'] = pd.DataFrame(
    [i for i in range(1, len(Ranked_resumes['Scores'])+1)])



###################################### SCORE TABLE PLOT ####################################

fig1 = go.Figure(data=[go.Table(
    header=dict(values=["Rank", "Name", "Scores"],
                fill_color='#00416d',
                align='center', font=dict(color='white', size=16)),
    cells=dict(values=[Ranked_resumes.Rank, Ranked_resumes.Name, Ranked_resumes.Scores],
               fill_color='#d6e0f0',
               align='left'))])
#st.write(fig1.to_html(),unsafe_allow_html=True)


fig1.update_layout(title="Top Ranked Resumes", width=700, height=800)
st.write(fig1,unsafe_allow_html=True)

org_Ranked_resumes=list(Ranked_resumes['Name'])

import base64
resume_dir = "./Data/Resumes/"
def encode(fx):
    es=None
    with open(fx, "rb") as image_file:
        es= base64.b64encode(image_file.read())
    return es
for i in range(len(Ranked_resumes['Name'])):
    es=encode(resume_dir+Ranked_resumes['Name'][i])
    Ranked_resumes['Name'][i]=f'<a href="data:file/csv;base64,'+es.decode()+'" download="'+Ranked_resumes['Name'][i]+'">'+Ranked_resumes['Name'][i]+'</a>'

#st.markdown(Ranked_resumes[['Rank',"Name",'Scores']].to_html(render_links=True, escape=False, index=False),unsafe_allow_html=True)
#Ranked_resumes[['Rank',"Name",'Scores']].to_json('test.json')
with st.expander("Download Top Resumes"):
    st.write(Ranked_resumes[['Rank',"Name",'Scores']].to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("---")
############################################ TF-IDF Code ###################################

col1,col2=st.columns((1,1))

fig2 = px.bar(Ranked_resumes,
              x=Ranked_resumes['Name'], y=Ranked_resumes['Scores'], color='Scores',
              color_continuous_scale='haline', title="Score and Rank Distribution")
# fig.update_layout(width=700, height=700)
with col1:
    st.plotly_chart(fig2, use_container_width=True)
    #st.write(fig2)


st.markdown("---")

############################################ TF-IDF Code ###################################


@st.cache()
def get_list_of_words(document):
    Document = []

    for a in document:
        raw = a.split(" ")
        Document.append(raw)

    return Document


document = get_list_of_words(Resumes['Cleaned'])

id2word = corpora.Dictionary(document)
corpus = [id2word.doc2bow(text) for text in document]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                            update_every=3, chunksize=100, passes=50, alpha='auto', per_word_topics=True)

################################### LDA CODE ##############################################


@st.cache  # Trying to improve performance by reducing the rerun computations
def format_topics_sentences(ldamodel, corpus):
    sent_topics_df = []
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df.append(
                    [i, int(topic_num), round(prop_topic, 4)*100, topic_keywords])
            else:
                break

    return sent_topics_df


################################# Topic Word Cloud Code #####################################
# st.sidebar.button('Hit Me')
with col2:
    option_yn12 = st.selectbox("Show the Word Cloud ?", options=['NO', 'YES'])
    if option_yn12 == 'YES':
            st.markdown("## Topics and Topic Related Keywords ")
            st.markdown(
                """This Wordcloud representation shows the Topic Number and the Top Keywords that contstitute a Topic.
                This further is used to cluster the resumes.      """)

            cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

            cloud = WordCloud(background_color='white',
                            width=2500,
                            height=1800,
                            max_words=10,
                            colormap='tab10',
                            collocations=False,
                            color_func=lambda *args, **kwargs: cols[i],
                            prefer_horizontal=1.0)

            topics = lda_model.show_topics(formatted=False)

            fig, axes = plt.subplots(2, 3, figsize=(10, 10), sharex=True, sharey=True)

            for i, ax in enumerate(axes.flatten()):
                fig.add_subplot(ax)
                topic_words = dict(topics[i][1])
                cloud.generate_from_frequencies(topic_words, max_font_size=300)
                plt.gca().imshow(cloud)
                plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
                plt.gca().axis('off')


            plt.subplots_adjust(wspace=0, hspace=0)
            plt.axis('off')
            plt.margins(x=0, y=0)
            plt.tight_layout()
            st.pyplot(plt, use_container_width=True)

st.markdown("---")

####################### SETTING UP THE DATAFRAME FOR SUNBURST-GRAPH ############################

df_topic_sents_keywords = format_topics_sentences(
    ldamodel=lda_model, corpus=corpus)
df_some = pd.DataFrame(df_topic_sents_keywords, columns=[
                       'Document No', 'Dominant Topic', 'Topic % Contribution', 'Keywords'])
df_some['Names'] = Resumes['Name']

df = df_some

st.markdown("## Topic Modelling of Resumes ")
st.markdown(
    "Using LDA to divide the topics into a number of usefull topics and creating a Cluster of matching topic resumes.  ")
fig3 = px.sunburst(df, path=['Dominant Topic', 'Names'], values='Topic % Contribution',
                   color='Dominant Topic', color_continuous_scale='viridis', width=800, height=800, title="Topic Distribution Graph")
st.write(fig3)


############################## RESUME PRINTING #############################

with st.form(key="Form1 :", clear_on_submit = False):
    s_file = st.select_slider(
        'Select the resume to display',
        options=org_Ranked_resumes)
    Submit3 = st.form_submit_button(label='Submit')
if Submit3:
    file_path=resume_dir+s_file
    if(s_file.split('.')[-1]=='pdf'):
        with open(file_path,"rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)




