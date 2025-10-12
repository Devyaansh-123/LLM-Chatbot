import os
import time
import tempfile
import streamlit as st
import requests
import io
from dotenv import load_dotenv
from gtts import gTTS
from streamlit_lottie import st_lottie
import streamlit.components.v1 as components

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader  
