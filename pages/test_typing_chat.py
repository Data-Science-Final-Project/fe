
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
    top_k=3, max_sections=3, score_threshold=0.6
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


async def generate_response_strict(user_input, k=3):
    raw_judgments = await find_relevant_judgments(user_input)
    raw_laws = await find_relevant_laws(user_input)

    raw_judgments = raw_judgments[:k]
    raw_laws = raw_laws[:k]

    judgment_texts = "\n\n".join([
        f"📘 פסק דין: {name}\n{content}" for name, content in raw_judgments
    ]) if raw_judgments else "לא נמצאו פסקי דין."

    law_texts = "\n\n".join([
        f"📗 חוק: {name}\n{content}" for name, content in raw_laws
    ]) if raw_laws else "לא נמצאו חוקים."

    # ✨ סיכום וסוג המסמך המצורף
    uploaded_summary = st.session_state.get("doc_summary", "")
    doc_type = st.session_state.get("doc_type", "מסמך משפטי")
    document_text_block = (
        f"\n\n📄 סיכום ה־{doc_type} שצורף:\n{uploaded_summary}"
        if uploaded_summary else ""
    )

    # 🧠 היסטוריית שיחה קודמת (לשאלות המשך)
    history = ""
    for msg in st.session_state["messages"][-6:]:  # עד 6 הודעות אחרונות
        role_label = "משתמש" if msg["role"] == "user" else "בוט"
        history += f"{role_label}: {msg['content']}\n"

    # 📜 ניסוח פרומפט מקצועי
    prompt = f"""
אתה יועץ משפטי מקצועי המתמחה בדין הישראלי.

היסטוריית שיחה:
{history}

המטרה שלך היא לענות על השאלה המשפטית המופיעה מטה – אך ורק על בסיס המסמכים המצורפים. 
כל טענה משפטית חייבת להתבסס על אחד מהמסמכים: החוק, פסק הדין, או {doc_type.lower()} שצורף.
אין להשתמש בידע כללי, ואין להמציא מידע.

---

❓ שאלה:
{user_input}

{document_text_block}

---

📚 מסמכים משפטיים:

{law_texts}

{judgment_texts}

---

אנא השב תשובה משפטית מקצועית, ברורה, ממוקדת ומנומקת, עם הפניות מדויקות למקורות (למשל: "סעיף 7 ב־{doc_type.lower()}", "סעיף 3 לחוק", או "פסק הדין פלוני נגד אלמוני").
"""

    response = await client_openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content.strip()




def display_messages():
    for msg in st.session_state['messages']:
        role = "user-message" if msg['role'] == "user" else "bot-message"
        st.markdown(
            f"<div class='{role}'>{msg['content']}<div class='timestamp'>{msg['timestamp']}</div></div>",
            unsafe_allow_html=True
        )
# ===== App =====
# כותרת ראשית
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
        st.session_state["uploaded_doc_text"] = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)
        st.success("המסמך נטען בהצלחה!")

        # סיווג המסמך
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

    # סיכום המסמך
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

    # הצגת סיכום
    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    # טופס שאלה משפטית
    with st.form("chat_form"):
        user_input = st.text_area("הכנס שאלה משפטית", height=100)
        if st.form_submit_button("שלח שאלה") and user_input.strip():
            add_message("user", user_input)
            save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
            st.rerun()

    # הפקת תשובה אוטומטית לאחר שאלה
    if st.session_state.get("messages") and st.session_state["messages"][-1]["role"] == "user":
        typing = show_typing_realtime()
        user_input = st.session_state["messages"][-1]["content"]

        response = asyncio.run(generate_response_strict(user_input))

        typing.empty()
        add_message("assistant", response)
        save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
        st.rerun()

    # שאלת המשך
    if st.session_state.get("messages"):
        with st.form("follow_up_form", clear_on_submit=True):
            follow_up = st.text_input("🔁 שאל שאלה נוספת על בסיס התשובה הקודמת:")
            if st.form_submit_button("שלח שאלה נוספת") and follow_up.strip():
                add_message("user", follow_up.strip())
                save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
                st.rerun()

    # ניקוי שיחה
    if st.button("🗑 נקה שיחה"):
        delete_conversation(chat_id)
        st.session_state["messages"] = []
        st.session_state["user_name"] = None
        st.rerun()


