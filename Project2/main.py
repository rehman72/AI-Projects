import streamlit as st
import PyPDF2
import io
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pdb
from langchain_core.messages import SystemMessage,HumanMessage

load_dotenv()

st.set_page_config(page_title="Ai Resume Reviewer",page_icon="📃",layout="centered")
st.title("AI Resume Analyzer 🚀")
st.markdown("Upload your resume in PDF format, and let the AI analyze it for you!")
#Get API Key from environment variable
os.getenv("GEMINI_API_KEY")
# Upload Resume
uploaded_file=st.file_uploader("Upload your Resume (PDF or txt)",type=["pdf","text"])
#Getting Job Role
job_role= st.text_input("Enter the job role you're interested (Optional): ")
analyze=st.button("Analyze Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader=PyPDF2.PdfReader(pdf_file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text() +"\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        file_content=extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("Please Upload File with Some Content to Analyze")
            st.stop()
        prompt = f"""Please analyze this resume and provide constructive feedback. 
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}

        Resume content:
        {file_content}
        
        Please provide your analysis in a clear, structured format with specific recommendations."""    

        llm=ChatGoogleGenerativeAI(api_key=os.getenv("GEMINI_API_KEY"),
                               model="gemini-2.5-flash",
                               temperature=0,
                               )
        messages=[
        SystemMessage(content="You are an Expert Resume Reviewer with year of experience in HR Recruitment."),
        HumanMessage(content=prompt)
        ]
        # pdb.set_trace()
        response=llm.invoke(messages)
        st.markdown("### Analysis Results: ")
        st.markdown(f"Response: {response.content}")
        
    except Exception as e:
        st.error(f"An Error occured: {str(e)}")


