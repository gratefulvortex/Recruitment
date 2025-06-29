import streamlit as st
import os
import fitz
import tempfile
import zipfile
import io
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import spacy
from spacy.matcher import Matcher

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Resume to JD Matcher",
    page_icon="📋",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Initialize the models
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_sm")
    return model, nlp

model, nlp = load_models()

# Custom CSS
st.markdown("""
    <style>
    /* Your CSS styles here */
    /* Fixed Footer */
    html body::after {
        content: '';
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 50px;
        background-color: #fff;
        border-top: 1px solid #e9ecef;
        box-shadow: 0 -2px 4px rgba(0,0,0,0.04);
        z-index: 1000;
    }

    html body::before {
        content: '© 2025 S3K Technologies | All rights reserved';
        position: fixed;
        bottom: 18px;
        left: 0;
        width: 100%;
        color: #495057;
        text-align: center;
        z-index: 1001;
        font-size: 0.9em;
    }

    /* General Layout */
    body {
        color: #495057;
        background-color: #f8f9fa;
        font-family: sans-serif;
    }

    .main .block-container {
        padding-bottom: 80px;
    }

    /* Header styling */
    .stApp > header {
        background-color: #fff;
        padding: 10px 20px;
        border-bottom: 2px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        position: sticky;
        top: 0;
        z-index: 999;
    }

    /* Sidebar styling */
    .stSidebar > div:first-child {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* Input Fields */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        padding: 12px 16px;
        border: 2px solid #e9ecef;
        background-color: #fff;
        font-size: 14px;
        color: #495057;
        transition: all 0.2s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #b22222;
        box-shadow: 0 0 0 3px rgba(178,34,34,0.1);
    }

    /* File Uploader */
    .stFileUploader > div > div {
        border: 2px dashed #e9ecef;
        border-radius: 8px;
        padding: 20px;
        background-color: white;
    }

    /* Buttons */
    .stButton > button {
        background-color: #b22222 !important;
        color: white !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background-color: #8b0000 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(178,34,34,0.2) !important;
    }

    /* Download Buttons */
    .download-btn {
        background-color: #b22222 !important;
        color: white !important;
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        font-size: 14px;
        transition: all 0.3s ease;
    }

    .download-btn:hover {
        background-color: #8b0000;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(178,34,34,0.2);
    }

    /* Visit Us Button */
    .visit-button {
        display: inline-block;
        padding: 8px 20px;
        background-color: #b22222;
        color: white !important;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-right: 10px;
    }

    .visit-button:hover {
        background-color: #8b0000;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(178,34,34,0.2);
    }

    /* DataFrames */
    .dataframe {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe th {
        background-color: #f8f9fa;
        color: #495057;
        font-weight: 600;
        padding: 12px !important;
    }

    .dataframe td {
        padding: 12px !important;
        border-top: 1px solid #e9ecef;
    }

    /* Metrics */
    .css-1r6slb0 {
        background-color: white;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo and visit button
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("https://s3ktech.ai/wp-content/uploads/2025/03/S3Ktech-Logo.png", width=140)
    with col2:
        st.markdown("<h1 style='display: inline-block; margin-left: 20px;'>Resume to JD Matcher</h1>", unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <a href="https://s3ktech.ai/" target="_blank" class="visit-button">Visit Us</a>
            </div>
        """, unsafe_allow_html=True)

# Text extraction and processing functions
def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF file."""
    try:
        doc = fitz.open("pdf", pdf_bytes)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def parse_jd(jd_input, is_text=True):
    """Parse job description input."""
    if is_text:
        return jd_input
    elif isinstance(jd_input, bytes):
        try:
            return extract_text_from_pdf(jd_input)
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    return ""

def process_resumes(resume_files):
    """Process resume files."""
    resume_texts = {}
    for file in resume_files:
        try:
            text = extract_text_from_pdf(file.getvalue())
            if text:
                resume_texts[file.name] = text
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    return resume_texts

# Scoring functions
def extract_keywords(text):
    """Extract all technical terms, tools, languages, and frameworks from the text."""
    doc = nlp(text)
    keywords = set()

    role_indicators = ['developer', 'engineer', 'analyst', 'architect', 'manager',
                      'specialist', 'consultant', 'lead', 'director', 'expert', 'scientist']

    for token in doc:
        if (not token.is_stop and
            not token.is_punct and
            len(token.text) > 2):

            if token.pos_ in ['NOUN', 'PROPN']:
                keywords.add(token.text.lower())

            if token.text.istitle():
                keywords.add(token.text.lower())

    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 4:
            if any(word.istitle() for word in chunk.text.split()):
                keywords.add(chunk.text.lower())
            if any(role in chunk.text.lower() for role in role_indicators):
                keywords.add(chunk.text.lower())

    for i in range(len(doc) - 1):
        if (doc[i].pos_ in ['NOUN', 'PROPN'] and
            doc[i + 1].pos_ in ['NOUN', 'PROPN'] and
            not doc[i].is_stop and
            not doc[i + 1].is_stop):
            compound = f"{doc[i].text.lower()} {doc[i + 1].text.lower()}"
            keywords.add(compound)

    return keywords

def extract_skills(text):
    """Extract specific skills and requirements from the text."""
    doc = nlp(text)
    skills = set()

    skill_indicators = ['proficient', 'knowledge', 'understanding', 'experience',
                       'skills', 'expertise', 'ability', 'capable', 'familiar']

    for token in doc:
        if token.text.lower() in skill_indicators:
            for child in token.children:
                if child.dep_ in ['dobj', 'attr', 'nsubj']:
                    skills.add(child.text.lower())

    tech_skills = ['sql', 'python', 'r', 'qlikview', 'powerbi', 'google analytics',
                  'firebase', 'bi tools', 'analytics']

    for i in range(len(doc) - 1):
        if doc[i].text.lower() + ' ' + doc[i + 1].text.lower() in tech_skills:
            skills.add(doc[i].text.lower() + ' ' + doc[i + 1].text.lower())
        elif doc[i].text.lower() in tech_skills:
            skills.add(doc[i].text.lower())

    return skills

def calculate_keyword_score(resume_text, jd_text):
    """Calculate the keyword matching score between resume and job description."""
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)

    if not jd_keywords and not jd_skills:
        return 0.0

    matching_keywords = resume_keywords.intersection(jd_keywords)
    matching_skills = resume_skills.intersection(jd_skills)

    keyword_score = len(matching_keywords) / len(jd_keywords) if jd_keywords else 0
    skill_score = len(matching_skills) / len(jd_skills) if jd_skills else 0

    total_score = (0.3 * keyword_score + 0.7 * skill_score)
    return min(total_score, 1.0)

def extract_role_keywords(text):
    """Extract role-specific keywords from the text."""
    doc = nlp(text)
    role_keywords = set()

    role_indicators = ['developer', 'engineer', 'analyst', 'architect', 'manager',
                      'specialist', 'consultant', 'lead', 'director', 'expert',
                      'scientist', 'programmer', 'designer', 'administrator', 'coordinator']

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(role in chunk_text for role in role_indicators):
            role_keywords.add(chunk_text)

    for token in doc:
        token_text = token.text.lower()
        if token_text in role_indicators:
            role_keywords.add(token_text)

    return role_keywords

def calculate_role_score(resume_text, jd_text):
    """Calculate the role matching score between resume and job description."""
    resume_roles = extract_role_keywords(resume_text)
    jd_roles = extract_role_keywords(jd_text)

    if not jd_roles:
        return 0.0

    exact_matches = resume_roles.intersection(jd_roles)

    partial_matches = set()
    for jd_role in jd_roles:
        for resume_role in resume_roles:
            if jd_role in resume_role or resume_role in jd_role:
                partial_matches.add((jd_role, resume_role))

    exact_score = len(exact_matches) / len(jd_roles)
    partial_score = len(partial_matches) / (2 * len(jd_roles))

    return max(exact_score, partial_score)

def calculate_hybrid_score(resume_text, jd_text, weights=None):
    """Calculate hybrid score combining semantic similarity, keyword matching, and role matching."""
    if weights is None:
        weights = {
            "semantic_weight_with_role": 0.1,
            "keyword_weight_with_role": 0.7,
            "role_weight": 0.2,
            "semantic_weight_no_role": 0.2,
            "keyword_weight_no_role": 0.8
        }

    resume_embedding = model.encode([resume_text])[0]
    jd_embedding = model.encode([jd_text])[0]
    semantic_score = cosine_similarity([resume_embedding], [jd_embedding])[0][0]

    keyword_score = calculate_keyword_score(resume_text, jd_text)

    jd_roles = extract_role_keywords(jd_text)
    if jd_roles:
        role_score = calculate_role_score(resume_text, jd_text)
        final_score = (
            weights["semantic_weight_with_role"] * semantic_score +
            weights["keyword_weight_with_role"] * keyword_score +
            weights["role_weight"] * role_score
        )
    else:
        role_score = 0.0
        final_score = (
            weights["semantic_weight_no_role"] * semantic_score +
            weights["keyword_weight_no_role"] * keyword_score
        )

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_score,
        "role_score": role_score,
        "has_role_requirement": bool(jd_roles)
    }

def main():
    with st.sidebar:
        st.header("Settings")
        display_options = st.selectbox(
            "Number of resumes to display",
            ["Top 5", "Top 10", "All"]
        )

        jd_type = st.radio(
            "Job Description Input Type",
            ["Text", "PDF"],
            horizontal=True
        )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Job Description")

        if jd_type == "Text":
            jd_text = st.text_area("Paste job description here", height=300)
            jd_file = None
        else:
            jd_file = st.file_uploader("Upload job description PDF", type=['pdf'], key="jd_uploader")
            jd_text = ""

            if jd_file is not None:
                try:
                    parsed_text = extract_text_from_pdf(jd_file.getvalue())
                    st.subheader("Parsed Job Description")
                    st.text_area("Parsed text", value=parsed_text, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Error parsing PDF: {str(e)}")

    with col2:
        st.header("Resumes")
        resume_files = st.file_uploader(
            "Upload resumes",
            type=['pdf'],
            accept_multiple_files=True,
            key="resume_uploader"
        )

    if st.button("Process and Match"):
        if not jd_text and not jd_file:
            st.error("Please provide a job description either by pasting text or uploading a PDF")
            return

        if not resume_files:
            st.error("Please upload at least one resume")
            return

        jd_content = parse_jd(jd_text, is_text=True) if jd_text else parse_jd(jd_file.getvalue(), is_text=False)
        if not jd_content:
            st.error("Failed to extract job description text. Please ensure the file is a valid PDF or text is not empty.")
            return

        resume_texts = process_resumes(resume_files)
        if not resume_texts:
            st.error("No valid resumes found")
            return

        results = []
        for filename, resume_text in resume_texts.items():
            scores = calculate_hybrid_score(resume_text, jd_content)
            results.append({
                "filename": filename,
                "final_score": scores["final_score"],
                "semantic_score": scores["semantic_score"],
                "keyword_score": scores["keyword_score"],
                "role_score": scores["role_score"]
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)

        if display_options == "Top 5":
            results = results[:5]
        elif display_options == "Top 10":
            results = results[:10]

        st.header("Matching Results")
        if results:
            st.markdown("""
            <style>
            .download-btn {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border: none;
                border-radius: 4px;
                text-decoration: none;
                font-size: 12px;
                cursor: pointer;
            }
            .results-table th, .results-table td {
                padding: 8px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .results-table th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            .results-table tr:hover {background-color: #f5f5f5;}
            .stButton button {height: 36px; line-height: 1;}
            </style>
            """, unsafe_allow_html=True)

            file_data = {}
            for result in results:
                original_file = next((f for f in resume_files if f.name == result['filename']), None)
                if original_file:
                    file_data[result['filename']] = original_file.getvalue()

            if file_data:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for filename, file_content in file_data.items():
                        zip_file.writestr(filename, file_content)

                st.download_button(
                    label=f"Download All {len(results)} Resumes as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="top_resumes.zip",
                    mime="application/zip",
                    key="download_all_zip"
                )

            df_data = []
            for result in results:
                df_data.append({
                    "Filename": result["filename"],
                    "Final Score": f"{result['final_score']:.2%}"
                })

            df = pd.DataFrame(df_data)
            st.dataframe(df)

            st.markdown("### Download Options")
            st.markdown("*Individual Resume Downloads:*")

            button_cols = st.columns(3)
            for i, result in enumerate(results):
                col_idx = i % 3
                with button_cols[col_idx]:
                    if result['filename'] in file_data:
                        st.download_button(
                            label=f"Download {result['filename']}",
                            data=file_data[result['filename']],
                            file_name=result["filename"],
                            mime="application/pdf",
                            key=f"download_{result['filename']}",
                            use_container_width=True
                        )
        else:
            st.info("No matching resumes found")

if __name__ == "__main__":
    main()
