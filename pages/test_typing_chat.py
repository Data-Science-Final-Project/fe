import os
import streamlit as st
import torch
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime
from app_resources import mongo_client, pinecone_client, model
import uuid
from streamlit_js import st_js, st_js_blocking
import json
import fitz
import docx
import asyncio
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables early
load_dotenv()
DATABASE_NAME = os.getenv("DATABASE_NAME")
client_openai = AsyncOpenAI(api_key=os.getenv("OPEN_AI"))

# External sources
judgment_index = pinecone_client.Index("judgments-names")
law_index = pinecone_client.Index("laws-names")
judgment_collection = mongo_client[DATABASE_NAME]["judgments"]
law_collection = mongo_client[DATABASE_NAME]["laws"]
conversation_collection = mongo_client[DATABASE_NAME]["conversations"]

torch.classes.__path__ = []
st.set_page_config(page_title="Ask Mini Lawyer", page_icon="💬", layout="wide")

# ===== UI Style =====
st.markdown("""
<style>
    .chat-container {background-color: #1E1E1E; padding: 20px; border-radius: 10px;}
    .chat-header {color: #4CAF50; font-size: 36px; font-weight: bold; text-align: center;}
    .user-message {background-color: #4CAF50; color: #ecf2f8; padding: 10px; border-radius: 10px; margin: 10px;}
    .bot-message {background-color: #44475a; color: #ecf2f8; padding: 10px; border-radius: 10px; margin: 10px;}
    .timestamp {font-size: 0.8em; color: #bbb;}
</style>
""", unsafe_allow_html=True)

# ===== Functions =====
def get_localstorage_value(key): return st_js_blocking(f"return localStorage.getItem('{key}');", key="get_" + key)
def set_localstorage_value(key, value): st_js(f"localStorage.setItem('{key}', '{value}');")

def get_or_create_chat_id():
    if 'current_chat_id' not in st.session_state:
        chat_id = get_localstorage_value("MiniLawyerChatId")
        if not chat_id or chat_id == "null":
            chat_id = str(uuid.uuid4())
            set_localstorage_value("MiniLawyerChatId", chat_id)
        st.session_state.current_chat_id = chat_id
    return st.session_state.current_chat_id

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

def read_pdf(file):
    return "".join([page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf")])

def read_docx(file):
    return "\n".join([p.text for p in docx.Document(file).paragraphs])

def show_typing_realtime(msg="🧐 הבוט מקליד..."):
    ph = st.empty()
    ph.markdown(f"<div style='color:gray;'>{msg}</div>", unsafe_allow_html=True)
    return ph

def add_message(role, content):
    st.session_state['messages'].append({
        "role": role, "content": content, "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# ===== Async Document Retrieval Engine =====
async def find_relevant_documents_fulltext(
    text, index, mongo_collection,
    id_field, name_field, desc_field, label,
    top_k=3, max_sections=3, score_threshold=0.75
):
    try:
        question_embedding = model.encode([text], normalize_embeddings=True)[0]
        results = index.query(
            vector=question_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            metric="cosine"
        )

        output = []

        for match in results.get("matches", []):
            if match.get("score", 0) < score_threshold:
                continue

            meta = match.get("metadata", {})
            doc = mongo_collection.find_one({id_field: meta.get(id_field)})
            if not doc:
                continue

            name = doc.get(name_field, "לא צויין")
            desc = doc.get(desc_field, "אין תיאור")
            segments = doc.get("Segments", [])

            section_texts = []
            for seg in segments:
                number = seg.get("SectionNumber", "")
                title = seg.get("SectionDescription", "").strip()
                content = seg.get("SectionContent", "").strip()
                if content:
                    section_texts.append(f"סעיף {number}: {title}\n{content}")

            if section_texts:
                section_embeddings = model.encode(section_texts, normalize_embeddings=True)
                similarities = cosine_similarity([question_embedding], section_embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:max_sections]

                top_sections = [section_texts[i] for i in top_indices]
                top_text = f"תיאור כללי: {desc}\n\n" + "\n\n".join(top_sections)
                output.append((name, top_text))

        return output

    except Exception as e:
        return [(f"שגיאה באחזור {label.lower()}ים", str(e))]

async def find_relevant_judgments(text):
    return await find_relevant_documents_fulltext(
        text, judgment_index, judgment_collection,
        id_field="CaseNumber", name_field="Name", desc_field="Description", label="פסק דין"
    )

async def find_relevant_laws(text):
    return await find_relevant_documents_fulltext(
        text, law_index, law_collection,
        id_field="IsraelLawID", name_field="Name", desc_field="Description", label="חוק"
    )

# ===== פונקציה אחת לשליחת שאלה או שאלת המשך =====
async def handle_question_submission(user_input, chat_id, is_follow_up=False):
    # אם מדובר בשאלת המשך, נדאג להוסיף היסטוריית שיחה קודמת
    if is_follow_up:
        history = ""
        for msg in st.session_state["messages"][-6:]:  # עד 6 הודעות אחרונות
            role_label = "משתמש" if msg["role"] == "user" else "בוט"
            history += f"{role_label}: {msg['content']}\n"
    else:
        history = ""  # אין היסטוריה בשאלת פתיחה

    # הפקת תשובה עם חיפוש סמנטי
    response = await generate_response_strict(user_input)

    # הוספת ההודעה החדשה (שאלה או שאלת המשך)
    add_message("user", user_input)
    save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])

    # שליחה של התשובה מהבוט
    add_message("assistant", response)
    save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])

    st.rerun()

# ===== קוד הממשק הגרפי (UI) =====
st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)
chat_id = get_or_create_chat_id()

# טעינת שם משתמש ושיחה
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = load_conversation(chat_id)

# התחברות ראשונית
if not st.session_state["user_name"]:
    with st.form("user_name_form"):
        name = st.text_input("הכנס שם להתחלת שיחה:")
        if st.form_submit_button("התחל שיחה") and name:
            st.session_state["user_name"] = name
            add_message("assistant", f"שלום {name}, איך אפשר לעזור?")
            save_conversation(chat_id, name, st.session_state["messages"])
            st.rerun()

else:
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_messages()
        st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf", "docx"])

if uploaded_file:
    # קריאת הטקסט מהקובץ
    st.session_state["uploaded_doc_text"] = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)
    st.success("המסמך נטען בהצלחה!")

    # סיווג אוטומטי של סוג המסמך (חוזה, מכתב וכו')
    with st.spinner("📑 מסווג את סוג המסמך..."):
        classify_prompt = f"""
        סווג את סוג המסמך הבא לאחת מהקטגוריות: חוזה, מכתב, תקנון, תביעה, פסק דין, אחר.
        החזר רק את שם הקטגוריה המתאימה ביותר.

        ---
        {st.session_state["uploaded_doc_text"][:1500]}
        ---
        """
        classification_response = asyncio.run(client_openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": classify_prompt}],
            temperature=0,
            max_tokens=10
        ))
        doc_type = classification_response.choices[0].message.content.strip()
        st.session_state["doc_type"] = doc_type
        st.success(f"📄 סוג המסמך שזוהה: {doc_type}")

# כפתור סיכום מופיע רק אחרי טעינת הטקסט
if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
    with st.spinner("GPT מסכם את המסמך..."):
        summary_prompt = f"""סכם את המסמך המשפטי הבא בקצרה:\n---\n{st.session_state['uploaded_doc_text']}"""
        summary_response = asyncio.run(client_openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5
        ))
        st.session_state["doc_summary"] = summary_response.choices[0].message.content.strip()
        st.success("📃 המסמך סוכם בהצלחה.")


    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    with st.form("chat_form"):
        user_input = st.text_area("הכנס שאלה משפטית", height=100)
        if st.form_submit_button("שלח שאלה") and user_input.strip():
            # שליחה של השאלה הראשונה
            await handle_question_submission(user_input, chat_id, is_follow_up=False)

    if st.session_state['messages'] and st.session_state['messages'][-1]['role'] == "user":
        typing = show_typing_realtime()
        user_input = st.session_state['messages'][-1]['content']

        # הפקת תשובה לשאלה
        await handle_question_submission(user_input, chat_id, is_follow_up=False)

# ✅ שאלת המשך (follow-up)
if st.session_state.get("user_name") and st.session_state.get("messages"):
    with st.form("follow_up_form", clear_on_submit=True):
        follow_up = st.text_input("🔁 שאל שאלה נוספת על בסיס התשובה הקודמת:")
        if st.form_submit_button("שלח שאלה נוספת") and follow_up.strip():
            # שליחה של שאלת המשך
            await handle_question_submission(follow_up.strip(), chat_id, is_follow_up=True)

    if st.button("🗑 נקה שיחה"):
        delete_conversation(chat_id)
        st.session_state["messages"] = []
        st.session_state["user_name"] = None
        st.rerun()
