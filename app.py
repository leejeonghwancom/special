# app.py
import streamlit as st
import os
import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========================================
# 0. 설정 (환경에 맞게 수정 필요)
# ==========================================
# BASE_DIR을 현재 앱이 실행되는 디렉토리로 설정
BASE_DIR = os.path.dirname(__file__) # app.py 파일이 있는 디렉토리

# FAISS 및 CSV 파일이 저장된 하위 디렉토리 (app.py와 같은 위치)
# 이전 대화에서 "slowrecent" 폴더 아래에 저장한다고 하셨으므로,
# 만약 app.py와 같은 레벨에 slowrecent가 있다면:
DATA_DIR = os.path.join(BASE_DIR, "slowrecent")

# 또는 만약 app.py가 있는 폴더 자체가 데이터 폴더라면 (즉, slowbot/ 폴더 안에 데이터가 있다면):
# DATA_DIR = BASE_DIR

# FAISS 인덱스 파일 경로
# faiss_index_recent.bin, documents.pkl, metadata.pkl이 DATA_DIR 안에 있다고 가정
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index_recent.bin")
FAISS_DOCS_FILE = os.path.join(DATA_DIR, "documents.pkl")
FAISS_METADATAS_FILE = os.path.join(DATA_DIR, "metadata.pkl")

# CSV 파일 경로
CSV_DATA_FILE = os.path.join(DATA_DIR, "slowletter_data_recent.csv")

# ==========================================
# 1. 환경 변수 (API KEY)
# ... (나머지 코드는 동일)
