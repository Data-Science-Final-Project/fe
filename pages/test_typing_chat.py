# unified_legal_finder.py

import os
import json
import uuid
import torch
import fitz
import docx
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI
from app_resources import mongo_client, pinecone_client, model
from streamlit_js import st_js, st_js_blocking
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load environment variables
load_dotenv()
DATABASE_NAME = os.getenv("DATABASE_NAME")
OPENAI_API_KEY = os.getenv("OPEN_AI")
client_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Collections & Indexes
judgment_index = pinecone_client.Index("judgments-names")
law_index = pinecone_client.Index("laws-names")
judgment_collection = mongo_client[DATABASE_NAME]["judgments"]
law_collection = mongo_client[DATABASE_NAME]["laws"]
conversation_collection = mongo_client[DATABASE_NAME]["conversations"]

torch.classes.__path__ = []
st.set_page_config(page_title="Ask Mini Lawyer", page_icon="\ud83d\udcac", layout="wide")

# ========== UI Styling ==========
st.markdown("""
<style>
    .chat-container {background-color: #1E1E1E; padding: 20px; border-radius: 10px;}
    .chat-header {color: #4CAF50; font-size: 36px; font-weight: bold; text-align: center;}
    .user-message {background-color: #4CAF50; color: #ecf2f8; padding: 10px; border-radius: 10px; margin: 10px;}
    .bot-message {background-color: #44475a; color: #ecf2f8; padding: 10px; border-radius: 10px; margin: 10px;}
    .timestamp {font-size: 0.8em; color: #bbb;}
</style>
""", unsafe_allow_html=True)

# ========== Helper Functions ==========
def get_localstorage_value(key):
    return st_js_blocking(f"return localStorage.getItem('{key}');", key="get_" + key)

def set_localstorage_value(key, value):
    st_js(f"localStorage.setItem('{key}', '{value}');")

def get_or_create_chat_id():
    if 'current_chat_id' not in st.session_state:
        chat_id = get_localstorage_value("MiniLawyerChatId")
        if not chat_id or chat_id == "null":
            chat_id = str(uuid.uuid4())
            set_localstorage_value("MiniLawyerChatId", chat_id)
        st.session_state.current_chat_id = chat_id
    return st.session_state.current_chat_id

def read_pdf(file):
    return "".join([page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf")])

def read_docx(file):
    return "\n".join([p.text for p in docx.Document(file).paragraphs])

def add_message(role, content):
    st.session_state['messages'].append({"role": role, "content": content, "timestamp": datetime.now().strftime("%H:%M:%S")})

def display_messages():
    for msg in st.session_state['messages']:
        role = "user-message" if msg['role'] == "user" else "bot-message"
        st.markdown(f"<div class='{role}'>{msg['content']}<div class='timestamp'>{msg['timestamp']}</div></div>", unsafe_allow_html=True)

def save_conversation(chat_id, user_name, messages):
    conversation_collection.update_one(
        {"local_storage_id": chat_id},
        {"$set": {"local_storage_id": chat_id, "user_name": user_name, "messages": messages}},
        upsert=True
    )

def load_conversation(chat_id):
    convo = conversation_collection.find_one({"local_storage_id": chat_id})
    return convo.get('messages', []) if convo else []

def delete_conversation(chat_id):
    conversation_collection.delete_one({"local_storage_id": chat_id})
    st_js("localStorage.clear();")
    st.session_state.current_chat_id = None

def show_typing_realtime(msg="\ud83e\uddd0 הבוט מקליד..."):
    ph = st.empty()
    ph.markdown(f"<div style='color:gray;'>{msg}</div>", unsafe_allow_html=True)
    return ph

# ========== Async Legal Document Search ==========

async def find_relevant_documents(text, index, collection, id_field, name_field, desc_field, label):
    question_embedding = model.encode([text], normalize_embeddings=True)[0]
    results = index.query(vector=question_embedding.tolist(), top_k=3, include_metadata=True)
    output = []

    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        doc_id = meta.get(id_field)
        if not doc_id:
            continue
        doc = collection.find_one({id_field: doc_id})
        if not doc:
            continue

        name = doc.get(name_field, "לא צויין")
        desc = doc.get(desc_field, "אין תיאור")
        segments = doc.get("Segments", [])

        section_texts = [
            f"סעיף {s.get('SectionNumber', '')}: {s.get('SectionDescription', '').strip()}\n{s.get('SectionContent', '').strip()}"
            for s in segments if s.get("SectionContent", "").strip()
        ]

        top_text = f"תיאור כללי: {desc}\n\n" + "\n\n".join(section_texts[:3])
        output.append((name, top_text))

    return output

async def generate_legal_response(user_input):
    laws = await find_relevant_documents(user_input, law_index, law_collection, "IsraelLawID", "Name", "Description", "חוק")
    judgments = await find_relevant_documents(user_input, judgment_index, judgment_collection, "CaseNumber", "Name", "Description", "פסק דין")

    doc_summary = st.session_state.get("doc_summary", "")
    doc_type = st.session_state.get("doc_type", "מסמך משפטי")

    prompt = f"""
אתה יועץ משפטי מקצועי המתמחה בדין הישראלי.

❓ שאלה:
{user_input}

📄 סיכום מסמך מצורף:
{doc_summary}

📗 חוקים:
{chr(10).join([f'\u2022 {name}\n{text}' for name, text in laws])}

📘 פסקי דין:
{chr(10).join([f'\u2022 {name}\n{text}' for name, text in judgments])}

---
ענה בצורה מקצועית, עם הפניות למסמכים בלבד.
"""

    response = await client_openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

# ========== App ==========
st.markdown('<div class="chat-header">\ud83d\udcac Ask Mini Lawyer</div>', unsafe_allow_html=True)
chat_id = get_or_create_chat_id()

if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = load_conversation(chat_id)

if not st.session_state["user_name"]:
    with st.form("user_name_form"):
        name = st.text_input("הכנס שם להתחלת שיחה:")
        if st.form_submit_button("התחל שיחה") and name:
            st.session_state["user_name"] = name
            add_message("assistant", f"שלום {name}, איך אפשר לעזור?")
            save_conversation(chat_id, name, st.session_state["messages"])
            st.rerun()
else:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    display_messages()
    st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("\ud83d\udcc4 העלה מסמך משפטי", type=["pdf", "docx"])
    if uploaded_file:
        st.session_state["uploaded_doc_text"] = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)
        st.success("\u05d4\u05de\u05e1\u05de\u05da \u05e0\u05d8\u05e2\u05df \u05d1\u05d4\u05e6\u05dc\u05d7\u05d4!")

        classify_prompt = f"""סווג את סוג המסמך הבא לאחת מהקטגוריות: חוזה, מכתב, תקנון, תביעה, פסק דין, אחר.
החזר רק את שם הקטגוריה.
---
{st.session_state['uploaded_doc_text'][:1500]}
---"""

        response = asyncio.run(client_openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classify_prompt}],
            temperature=0,
            max_tokens=10
        ))
        st.session_state["doc_type"] = response.choices[0].message.content.strip()
        st.success(f"\ud83d\udcc4 סוג המסמך: {st.session_state['doc_type']}")

    if "uploaded_doc_text" in st.session_state and st.button("\ud83d\udccb סכם את המסמך"):
        with st.spinner("GPT מסכם את המסמך..."):
            summary_prompt = f"""סכם את המסמך המשפטי הבא בקצרה:\n---\n{st.session_state['uploaded_doc_text']}"""
            summary_response = asyncio.run(client_openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5
            ))
            st.session_state["doc_summary"] = summary_response.choices[0].message.content.strip()
            st.success("\ud83d\udcc3 המסמך סוכם בהצלחה.")

    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    with st.form("chat_form"):
        user_input = st.text_area("\u05d4\u05db\u05e0\u05e1 \u05e9\u05d0\u05dc\u05d4 \u05de\u05e9\u05e4\u05d8\u05d9\u05ea", height=100)
        if st.form_submit_button("\u05e9\u05dc\u05d7 \u05e9\u05d0\u05dc\u05d4") and user_input.strip():
            add_message("user", user_input)
            save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
            st.rerun()

    if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "user":
        typing = show_typing_realtime()
        user_input = st.session_state["messages"][-1]["content"]
        response = asyncio.run(generate_legal_response(user_input))
        typing.empty()
        add_message("assistant", response)
        save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
        st.rerun()

    if st.button("\ud83d\uddd1 נקה שיחה"):
        delete_conversation(chat_id)
        st.session_state["messages"] = []
        st.session_state["user_name"] = None
        st.rerun()
