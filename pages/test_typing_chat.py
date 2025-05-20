import os
import sys
import json
import uuid
import asyncio
from datetime import datetime

import streamlit as st
import torch
import fitz  # PyMuPDF
import docx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from streamlit_js import st_js, st_js_blocking

from app_resources import mongo_client, pinecone_client, model

# ------------------------------------------------------------
# Environment & Globals
# ------------------------------------------------------------
load_dotenv()
DATABASE_NAME = os.getenv("DATABASE_NAME")
OPENAI_API_KEY = os.getenv("OPEN_AI")

client_async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
client_sync_openai  = OpenAI(api_key=OPENAI_API_KEY)

torch.classes.__path__ = []             # silence JIT warning in some envs
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------
# Pinecone & Mongo collections (shared)
# ------------------------------------------------------------
judgment_index = pinecone_client.Index("judgments-names")
law_index      = pinecone_client.Index("laws-names")

db                    = mongo_client[DATABASE_NAME]
judgment_collection   = db["judgments"]
law_collection        = db["laws"]
conversation_coll     = db["conversations"]

# ------------------------------------------------------------
# Streamlit Config & Global Styles
# ------------------------------------------------------------
st.set_page_config(page_title="Ask Mini Lawyer Suite", page_icon="⚖️", layout="wide")

COMMON_CSS = """
<style>
  .chat-container {background:#1E1E1E;padding:20px;border-radius:10px;}
  .chat-header   {color:#4CAF50;font-size:36px;font-weight:bold;text-align:center;}
  .user-message  {background:#4CAF50;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px;}
  .bot-message   {background:#44475a;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px;}
  .timestamp     {font-size:0.75em;color:#bbb;}

  .law-card{border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:15px;
            box-shadow:0 2px 4px rgba(0,0,0,0.1);background:#f9f9f9;}
  .law-title{font-size:20px;font-weight:bold;color:#333}
  .law-description{font-size:16px;color:#444;margin:10px 0}
  .law-meta{font-size:14px;color:#555}

  .stButton>button{background:#7ce38b;color:#fff;font-size:14px;border:none;padding:8px 16px;
                   border-radius:5px;cursor:pointer}
  .stButton>button:hover{background:#69d67a}
</style>
"""

st.markdown(COMMON_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar – module switcher
# ------------------------------------------------------------
app_mode = st.sidebar.selectbox("Choose module", ["Chat Assistant", "Legal Finder Assistant"])

# =============================================================================
# ---------------  SHARED UTILITY FUNCTIONS  ----------------------------------
# =============================================================================

def get_localstorage_value(key: str):
    return st_js_blocking(f"return localStorage.getItem('{key}');", key="get_" + key)

def set_localstorage_value(key: str, value: str):
    st_js(f"localStorage.setItem('{key}', '{value}');")

# ----------  File Readers ----------------------------------------------------

def read_pdf(file):
    return "".join(page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf"))

def read_docx(file):
    return "\n".join(p.text for p in docx.Document(file).paragraphs)

# ----------  Chat helpers ----------------------------------------------------

def show_typing_realtime(msg: str = "🧐 הבוט מקליד..."):
    ph = st.empty()
    ph.markdown(f"<div style='color:gray;'>{msg}</div>", unsafe_allow_html=True)
    return ph


def add_message(role: str, content: str):
    st.session_state.setdefault("messages", []).append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

# ----------  Display stored chat --------------------------------------------

def display_messages():
    for msg in st.session_state.get("messages", []):
        css_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(
            f"<div class='{css_class}'>{msg['content']}<div class='timestamp'>{msg['timestamp']}</div></div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# -------------------------  CHAT ASSISTANT  ----------------------------------
# =============================================================================

def chat_assistant():
    """Conversational Ask Mini Lawyer bot"""
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)

    # --- Chat/session init --------------------------------------------------
    if "current_chat_id" not in st.session_state:
        cid = get_localstorage_value("MiniLawyerChatId") or str(uuid.uuid4())
        set_localstorage_value("MiniLawyerChatId", cid)
        st.session_state.current_chat_id = cid
    chat_id = st.session_state.current_chat_id

    if "messages" not in st.session_state:
        convo = conversation_coll.find_one({"local_storage_id": chat_id})
        st.session_state["messages"] = convo.get("messages", []) if convo else []
    st.session_state.setdefault("user_name", None)

    # --- Name prompt --------------------------------------------------------
    if not st.session_state["user_name"]:
        with st.form("user_name_form"):
            name = st.text_input("הכנס שם להתחלת שיחה:")
            if st.form_submit_button("התחל שיחה") and name:
                st.session_state["user_name"] = name
                add_message("assistant", f"שלום {name}, איך אפשר לעזור?")
                conversation_coll.update_one(
                    {"local_storage_id": chat_id},
                    {"$set": {"user_name": name, "messages": st.session_state["messages"]}},
                    upsert=True,
                )
                st.rerun()
        return

    # --- Chat container -----------------------------------------------------
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_messages()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- File upload --------------------------------------------------------
    uploaded_file = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf", "docx"])
    if uploaded_file:
        full_text = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)
        st.session_state["uploaded_doc_text"] = full_text
        st.success("המסמך נטען בהצלחה!")

        # classify doc type
        with st.spinner("📑 מסווג את סוג המסמך..."):
            classify_prompt = f"""
            סווג את סוג המסמך הבא לאחת מהקטגוריות: חוזה, מכתב, תקנון, תביעה, פסק דין, אחר.
            החזר רק את שם הקטגוריה המתאימה ביותר.
            ---
            {full_text[:1500]}
            ---"""
            resp = asyncio.run(
                client_async_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": classify_prompt}],
                    temperature=0,
                    max_tokens=12,
                )
            )
            st.session_state["doc_type"] = resp.choices[0].message.content.strip()
            st.success(f"📄 סוג המסמך שזוהה: {st.session_state['doc_type']}")

    # --- Summarise button ---------------------------------------------------
    if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
        with st.spinner("GPT מסכם את המסמך..."):
            doc_type = st.session_state.get("doc_type", "מסמך")
            summary_prompt = f"""
אתה עורך-דין מומחה. סכּם את {doc_type} הבא בשלושה חלקים:\n\n1. **Executive Summary** – פסקה עד 120 מילים המסבירה את מטרת המסמך וההקשר.\n2. **נקודות עיקריות** – רשימת בולטים על סעיפים מחייבים, מועדים קריטיים, סכומים עיקריים, סיכונים וסנקציות.\n3. **השלכות והמלצות** – פעולות או החלטות שהקורא צריך לשקול.\n
---\n{st.session_state['uploaded_doc_text']}\n---"""
            resp = asyncio.run(
                client_async_openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )
            )
            st.session_state["doc_summary"] = resp.choices[0].message.content.strip()
            st.success("📃 המסמך סוכם בהצלחה.")
    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    # --- Question helpers ---------------------------------------------------
    async def generate_response_strict(question: str) -> str:
        q_emb = model.encode([question], normalize_embeddings=True)[0]

        async def _fetch(index, coll, id_key):
            res = index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True)
            docs = []
            for m in res.get("matches", []):
                meta = m.get("metadata", {})
                doc = coll.find_one({id_key: meta.get(id_key)})
                if doc:
                    docs.append(doc)
            return docs

        laws      = await _fetch(law_index, law_collection, "IsraelLawID")
        judgments = await _fetch(judgment_index, judgment_collection, "CaseNumber")

        law_snippets  = "\n\n".join(d.get("Description", "")[:800] for d in laws)
        judg_snippets = "\n\n".join(d.get("Description", "")[:800] for d in judgments)

        sys_prompt = (
            "ענית רק בהתבסס על החוקים ופסקי הדין הבאים. צטט מקור במפורש.\n\n" +
            "--- חוקים רלוונטיים ---\n" + law_snippets + "\n\n" +
            "--- פסקי דין רלוונטיים ---\n" + judg_snippets + "\n\n"
        )

        ans = await client_async_openai.chat
