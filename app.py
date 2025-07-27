import os
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from datetime import datetime
from urllib.parse import quote
import streamlit as st

# ==========================================
# 1. 설정
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API Key가 없습니다. 환경 변수나 Streamlit secrets에 OPENAI_API_KEY를 설정하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

CSV_PATH = "/content/drive/MyDrive/slowproject/slowrecent/slowletter_data_recent.csv"
INDEX_PATH = "/content/drive/MyDrive/slowproject/slowrecent/faiss_index_recent.bin"
OUTPUT_DIR = "/content/drive/MyDrive/slowproject/slowrecent/special_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K = 20

# ==========================================
# 2. 데이터와 인덱스 로드
# ==========================================
@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(CSV_PATH)
    return index, df

index, df = load_resources()

# ==========================================
# 3. 임베딩
# ==========================================
def embed_text(text):
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding
    return np.array(emb, dtype="float32").reshape(1, -1)

# ==========================================
# 4. GPT 요약
# ==========================================
def gpt_summarize(text, query, mode="long"):
    if mode == "long":
        prompt = f"""
'{query}' 관련 상위 기사 내용을 기반으로,
주요 쟁점·흐름·인물·최근 동향을 7줄 내외로 요약하세요.
문체는 건조하고 간결하게.

{text}
"""
        max_tokens = 700
    else:
        prompt = f"다음 내용을 한 문장으로 짧게 요약:\n\n{text}"
        max_tokens = 150

    res = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=max_tokens,
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

# ==========================================
# 5. 스페셜 페이지 생성 함수
# ==========================================
def make_special_page(query):
    q_emb = embed_text(query)

    # 벡터 검색
    D, I = index.search(q_emb, 50)
    valid_indices = [i for i in I[0] if i < len(df)]
    rows = df.iloc[valid_indices].copy()

    # 가중치 기반 정렬 (벡터 + 최신성 + 엔티티 매칭)
    query_ns = query.replace(" ", "")
    scores = []
    for dist, (_, r) in zip(D[0], rows.iterrows()):
        score = -dist
        # 최신성
        try:
            date_str = str(r.get("date", ""))[:10]
            days = (datetime.now() - datetime.strptime(date_str, "%Y-%m-%d")).days
            score += max(0, 1 - days/365)
        except:
            pass
        # 엔티티 매칭
        ents = str(r.get("entities","")) + str(r.get("events",""))
        if query_ns in ents:
            score += 2
        scores.append(score)

    rows["score"] = scores
    rows = rows.sort_values("score", ascending=False).head(TOP_K)

    # 1차 요약용 context
    def safe_col(r, *cols):
        for c in cols:
            if c in r and isinstance(r[c], str):
                return r[c]
        return ""

    context = "\n\n".join(
        f"- {safe_col(r,'title','h3_title')}\n엔티티:{safe_col(r,'entities')}\n이벤트:{safe_col(r,'events')}\n내용:{safe_col(r,'content','h3_content_html')[:200]}"
        for _, r in rows.iterrows()
    )

    summary = gpt_summarize(context, query, mode="long")

    # 최신순 요약 리스트
    list_items = []
    for _, r in rows.sort_values("date", ascending=False).iterrows():
        short_context = f"{safe_col(r,'title','h3_title')} {safe_col(r,'entities')} {safe_col(r,'events')} {safe_col(r,'content','h3_content_html')[:200]}"
        short = gpt_summarize(short_context, query, mode="short")
        list_items.append((str(r.get("date",""))[:10], safe_col(r,'title','h3_title'), short))

    # HTML 저장
    html_path = os.path.join(OUTPUT_DIR, f"special_{quote(query)}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:800px;margin:2rem auto;} h1{color:#333} li{margin:0.5rem 0;} .date{color:#777;font-size:0.9em}</style></head><body>")
        f.write(f"<h1>{query} – 스페셜 리포트</h1>")
        f.write(f"<p>{summary}</p><hr><h2>관련 꼭지 ({TOP_K}건)</h2><ul>")
        for date, title, short in list_items:
            f.write(f"<li><span class='date'>{date}</span><br><b>{title}</b><br>{short}</li>")
        f.write("</ul></body></html>")

    return html_path

# ==========================================
# 6. Streamlit UI
# ==========================================
st.title("Slowletter 주제별 스페셜 페이지 생성기")

query = st.text_input("특집 페이지를 만들 주제어를 입력하세요:")

if st.button("생성"):
    if query.strip():
        with st.spinner(f"'{query}' 관련 페이지 생성 중..."):
            html_path = make_special_page(query)
            st.success(f"스페셜 페이지가 생성되었습니다: {html_path}")
