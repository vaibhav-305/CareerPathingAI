import nltk
import spacy
import streamlit as st

nltk.download('stopwords')
spacy.load('en_core_web_sm')

import pandas as pd
import base64
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import pafy
import plotly.express as px


import json

from typing import Any, List, Mapping, Optional
# from mistune import Markdown
# from openai.cli import display
import requests
from llama_index.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.llms.base import llm_completion_callback, CompletionResponseGen
from llama_index import ServiceContext, KnowledgeGraphIndex, StorageContext, VectorStoreIndex, download_loader,set_global_service_context
import os
# import pinecone
from llama_index import VectorStoreIndex, SimpleDirectoryReader
# from llama_index import SimpleCSVReader
from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import get_response_synthesizer

from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.vector_store import VectorIndexRetriever
# set context window size
context_window = 3500
# set number of output tokens
num_output = 2048

class OurLLM(CustomLLM):

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=context_window,
            num_output=num_output,
            # model_name=self.model_url
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Make a POST request to the model's API endpoint with the prompt as data
        model_url = "https://wv7b-satyamkumar-209565-0.datalab.euw1.prod.sgcip.io/v1/completions"
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({
            "prompt": "\n\n### Instructions:\n" + prompt + "\n\n### Response:\n",
            "stop": [
                "###"
            ],
            "max_tokens":500
            
        })
        response = requests.request("POST", model_url, headers=headers, data=payload)

        # Parse the response
        response_json = response.json()
        llm_response = response_json["choices"][0]["text"]
        # Print the original length
        

        # Extract the generated text from the response
        # This assumes the response has a field 'generated_text' with the generated text
        # generated_text = response_json['generated_text']
        # Post-process the response to limit its length
        # print(f"Original Response: {llm_response}")
        # max_response_length = 3000  # Set your desired maximum length
        # llm_response = llm_response[:max_response_length]
        # Return a CompletionResponse with the generated text
        return CompletionResponse(text=llm_response)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Your logic to call your custom LLM using self.model_url
        pass

def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


connection = pymysql.connect(host='127.0.0.1', user='root', password='i_love_Tzuyu21', db='sra')
cursor = connection.cursor()


def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills,
                courses):
    DB_table_name = 'user_data'
    insert_sql = "insert into " + DB_table_name + """
    values (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
    rec_values = (
        name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills,
        courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()


st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon='./Logo/resume.png',
)


def run():
    st.title("AI-powered Personalized Carrer Pathing")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    # link = '[©Developed by Spidy20](http://github.com/spidy20)'
    # st.sidebar.markdown(link, unsafe_allow_html=True)
    img = Image.open('./Logo/logoj.png')
    img = img.resize((900, 400))
    st.image(img)

    # Create the DB
    db_sql = """CREATE DATABASE IF NOT EXISTS sra;"""
    cursor.execute(db_sql)
    connection.select_db("sra")

    # Create table
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                     Name varchar(1000) NOT NULL,
                     Email_ID VARCHAR(1000) NOT NULL,
                     resume_score VARCHAR(8) NOT NULL,
                     Timestamp VARCHAR(50) NOT NULL,
                     Page_no VARCHAR(1000) NOT NULL,
                     Predicted_Field VARCHAR(25) NOT NULL,
                     User_level VARCHAR(30) NOT NULL,
                     Actual_skills VARCHAR(1000) NOT NULL,
                     Recommended_skills VARCHAR(1000) NOT NULL,
                     Recommended_courses VARCHAR(1000) NOT NULL,
                     PRIMARY KEY (ID));
                    """
    cursor.execute(table_sql)
    if choice == 'Normal User':
        # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Upload your resume, and get smart recommendation based on it."</h4>''',
        #             unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        
        if pdf_file is not None:
            # with st.spinner('Uploading your Resume....'):
            #     time.sleep(4)
            save_image_path = './Uploaded_Resumes/' + pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()

            if resume_data:
                ## Get the whole resume data
                resume_text = pdf_reader(save_image_path)
                 
                st.header("**Resume Analysis**")
                st.success("Hello " + resume_data['name'])
                st.subheader("**Your Basic info**")
                try:
                    st.text('Name: ' + resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Resume pages: ' + str(resume_data['no_of_pages']))

                except:
                    pass
                cand_level = ''
                if resume_data['no_of_pages'] == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['no_of_pages'] == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''',
                                unsafe_allow_html=True)
                elif resume_data['no_of_pages'] >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!''',
                                unsafe_allow_html=True)

                
                ## Skill shows
                keywords = st_tags(label='### Skills that you have',
                                   text='See our skills recommendation',
                                   value=resume_data['skills'], key='1')

                ##  recommendation
                ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask','python','numpy','pandas','matplotib'
                              'streamlit','reinforcement learning','SQL','Power BI','powerBI']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress','reactjs',
                               'javascript', 'angular js', 'c#', 'flask','java','spring','spring boot','git','css','html','C++','C']
                android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy', 'android sdk','material-ui']
                ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode','kotlin','react native']
                uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes',
                                'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator',
                                'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro',
                                'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp',
                                'user research', 'user experience']
                print()
                tech_skills=[]
                for a in ds_keyword:
                    for b in resume_data['skills']:
                        #print(b)
                        if a.lower()==b.lower():
                            tech_skills.append(a)
                for a in web_keyword:
                    for b in resume_data['skills']:
                        if a.lower()==b.lower():
                            tech_skills.append(a)
                for a in android_keyword:
                    for b in resume_data['skills']:
                        if a.lower()==b.lower():
                            tech_skills.append(a)
                for a in ios_keyword:
                    for b in resume_data['skills']:
                        if a.lower()==b.lower():
                            tech_skills.append(a)
                for a in uiux_keyword:
                    for b in resume_data['skills']:
                        if a.lower()==b.lower():
                            tech_skills.append(a)

                recommended_skills = []
                skillstatement="I have skills in "
                for i in tech_skills:
                    skillstatement = skillstatement+', '+i
                #print(skillstatement)

                reco_field = ''
                rec_course = ''
                ## Courses recommendation
                for i in resume_data['skills']:
                    ## Data science recommendation
                    if i.lower() in ds_keyword:
                        print(i.lower())
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        # recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                        #                       'Data Mining', 'Clustering & Classification', 'Data Analytics',
                        #                       'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                        #                       'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                        #                       'Streamlit']
                        # recommended_keywords = st_tags(label='### Recommended skills for you.',
                        #                                text='Recommended skills generated from System',
                        #                                value=recommended_skills, key='2')
                        llm = OurLLM()
                        service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                        set_global_service_context(service_context)
                        print(skillstatement+". What more things to learn for growing my career in  Data Science ")
                        response = llm.complete(skillstatement+". What more things to learn for growing my career in Data Science")
                        llmresponse=response.text
                        st.subheader("**Skills Recommendation💡**")
                        # st.success(llmresponse)
                        with st.chat_message("assistant"):
                            st.success(llmresponse)
                        
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break

                    ## Web development recommendation
                    elif i.lower() in web_keyword:
                        print(i.lower())
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        # recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                        #                       'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                        # recommended_keywords = st_tags(label='### Recommended skills for you.',
                        #                                text='Recommended skills generated from System',
                        #                                value=recommended_skills, key='3')
                        llm = OurLLM()
                        service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                        set_global_service_context(service_context)
                        print(skillstatement+". What more things to learn for growing my career in web development")
                        response = llm.complete(skillstatement+". What more things to learn for growing my career in web development ?")
                        llmresponse=response.text
                        st.subheader("LLM Response")
                        # st.success(llmresponse)
                        with st.chat_message("assistant"):
                            st.success(llmresponse)
                           
                        
                        
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)

                        
                        

                        break

                    ## Android App Development
                    elif i.lower() in android_keyword:
                        print(i.lower())
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        # recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                        #                       'Kivy', 'GIT', 'SDK', 'SQLite']
                        # recommended_keywords = st_tags(label='### Recommended skills for you.',
                        #                                text='Recommended skills generated from System',
                        #                                value=recommended_skills, key='4')
                        llm = OurLLM()
                        service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                        set_global_service_context(service_context)
                        response = llm.complete(skillstatement+". What more things to learn for growing my career in Android development")
                        llmresponse=response.text
                        st.subheader("LLM Response")
                        # st.success(llmresponse)
                        with st.chat_message("assistant"):
                            st.success(llmresponse)
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break

                    ## IOS App Development
                    elif i.lower() in ios_keyword:
                        print(i.lower())
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        # recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                        #                       'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                        #                       'Auto-Layout']
                        # recommended_keywords = st_tags(label='### Recommended skills for you.',
                        #                                text='Recommended skills generated from System',
                        #                                value=recommended_skills, key='5')
                        llm = OurLLM()
                        service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                        set_global_service_context(service_context)
                        response = llm.complete(skillstatement+". What more things to learn for growing my career in IOS development")
                        llmresponse=response.text
                        st.subheader("LLM Response")
                        # st.success(llmresponse)
                        with st.chat_message("assistant"):
                            st.success(llmresponse)
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break

                    ## Ui-UX Recommendation
                    elif i.lower() in uiux_keyword:
                        print(i.lower())
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        # recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                        #                       'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                        #                       'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                        #                       'Solid', 'Grasp', 'User Research']
                        # recommended_keywords = st_tags(label='### Recommended skills for you.',
                        #                                text='Recommended skills generated from System',
                        #                                value=recommended_skills, key='6')
                        llm = OurLLM()
                        service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                        set_global_service_context(service_context)
                        response = llm.complete(skillstatement+". What more things to learn for growing my career in UI-UX development")
                        llmresponse=response.text
                        st.subheader("LLM Response")
                        # st.success(llmresponse)
                        with st.chat_message("assistant"):
                            st.success(llmresponse)
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break

                #
                ## Insert into table
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date + '_' + cur_time)

                ### Resume writing recommendation
                st.subheader("**Resume Tips & Ideas💡**")
                resume_score = 0
                if 'Objective' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown(
                        '''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective</h5>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h5 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add your career objective, it will give your career intension to the Recruiters.</h5>''',
                        unsafe_allow_html=True)

                if 'Declaration' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown(
                        '''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Delcaration✍/h5>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h5 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Declaration✍. It will give the assurance that everything written on your resume is true and fully acknowledged by you</h5>''',
                        unsafe_allow_html=True)

                if 'Hobbies' or 'Interests' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown(
                        '''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Extra Curriculars⚽</h5>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h5 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Hobbies⚽. It will show your persnality to the Recruiters and give the assurance that you are fit for this role or not.</h5>''',
                        unsafe_allow_html=True)

                if 'Achievements' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown(
                        '''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements🏅 </h5>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h5 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Achievements🏅. It will show that you are capable for the required position.</h5>''',
                        unsafe_allow_html=True)

                if 'Projects' in resume_text:
                    resume_score = resume_score + 20
                    st.markdown(
                        '''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects👨‍💻 </h5>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '''<h5 style='text-align: left; color: #fabc10;'>[-] According to our recommendation please add Projects👨‍💻. It will show that you have done work related the required position or not.</h5>''',
                        unsafe_allow_html=True)
                
                #Aspirations
                st.divider()
                st.subheader('Do you have any other Career Aspiration?')
                aspi = st.text_input('')
                if aspi:
                    llm = OurLLM()
                    service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=152)
                    set_global_service_context(service_context)
                    response = llm.complete("My aspirations are: "+aspi+'. Suggest me a roadmap to pursue it')
                    llmresponse=response.text
                    st.subheader("Here is how you can pursue your aspirations ...")
                            # st.success(llmresponse)
                    with st.chat_message("assistant"):
                        st.success(llmresponse)

                
                
                # st.subheader("**Resume Score📝**")
                # st.markdown(
                #     """
                #     <style>
                #         .stProgress > div > div > div > div {
                #             background-color: #d73b5c;
                #         }
                #     </style>""",
                #     unsafe_allow_html=True,
                # )
                # my_bar = st.progress(0)
                # score = 0
                # for percent_complete in range(resume_score):
                #     score += 1
                #     time.sleep(0.1)
                #     my_bar.progress(percent_complete + 1)
                # st.success('** Your Resume Writing Score: ' + str(score) + '**')
                # st.warning(
                #     "** Note: This score is calculated based on the content that you have added in your Resume. **")
                # st.balloons()

                # insert_data(resume_data['name'], resume_data['email'], str(resume_score), timestamp,
                #             str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                #             str(recommended_skills), str(rec_course))
                
                # ## Resume writing video
                # st.header("**Bonus Video for Resume Writing Tips💡**")
                # resume_vid = random.choice(resume_videos)
                # res_vid_title = fetch_yt_video(resume_vid)
                # st.subheader("✅ **" + res_vid_title + "**")
                # st.video(resume_vid)
                #
                # ## Interview Preparation Video
                # st.header("**Bonus Video for Interview👨‍💼 Tips💡**")
                # interview_vid = random.choice(interview_videos)
                # int_vid_title = fetch_yt_video(interview_vid)
                # st.subheader("✅ **" + int_vid_title + "**")
                # st.video(interview_vid)

                connection.commit()
            else:
                st.error('Something went wrong..')
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        # st.sidebar.subheader('**ID / Password Required!**')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'satyam' and ad_password == 'mlhub123':
                st.success("Welcome Satyam")
                # Display Data
                cursor.execute('''SELECT*FROM user_data''')
                data = cursor.fetchall()
                st.header("**User's👨‍💻 Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                 'Recommended Course'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)
                ## Admin Side Data
                query = 'select * from user_data;'
                plot_data = pd.read_sql(query, connection)
                print('debug')
                print(plot_data)
                ## Pie chart for predicted field recommendations
                labels = plot_data.Predicted_Field.unique()
                print(labels)
                values = plot_data.Predicted_Field.value_counts()
                print(values)
                print('debug stop')
                st.subheader("📈 **Pie-Chart for Predicted Field Recommendations**")
                fig = px.pie(values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                ### Pie chart for User's👨‍💻 Experienced Level
                labels = plot_data.User_level.unique()
                values = plot_data.User_level.value_counts()
                st.subheader("📈 ** Pie-Chart for User's👨‍💻 Experienced Level**")
                fig = px.pie(values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                st.plotly_chart(fig)


            else:
                st.error("Wrong ID & Password Provided")


run()
