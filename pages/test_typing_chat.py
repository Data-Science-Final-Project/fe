
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
async def find_relevant_documents(
    text, index, mongo_collection,
    id_field, name_field, desc_field, label,
    top_k=3, score_threshold=0.75
):
    try:
        embedding = model.encode([text], normalize_embeddings=True)[0]
        results = index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            metric="cosine"
        )

        tasks = []
        names = []
        full_prompts = []

        for match in results.get("matches", []):
            if match.get("score", 0) < score_threshold:
                continue

            meta = match.get("metadata", {})
            doc = mongo_collection.find_one({id_field: meta.get(id_field)})
            if not doc:
                continue

            name = doc.get(name_field, "לא צויין")
            desc = doc.get(desc_field, "אין תיאור")

            # 🧩 שלב ראשון – רק 3 סעיפים ראשונים
            segments = doc.get("Segments", [])
            short_section_text = ""
            full_section_text = ""
            for i, seg in enumerate(segments):
                number = seg.get("SectionNumber", "")
                title = seg.get("SectionDescription", "").strip()
                content = seg.get("SectionContent", "").strip()
                if content:
                    block = f"\n\nסעיף {number}: {title}\n{content}"
                    if i < 3:
                        short_section_text += block
                    full_section_text += block

            base_prompt = f"""סצנה:\n{text}\n\n{label}:\nשם: {name}\nתיאור כללי: {desc}"""

            short_prompt = base_prompt + short_section_text + \
                f"\n\nמדוע {label.lower()} זה רלוונטי לסיטואציה? החזר JSON:\n{{\"advice\": \"הסבר\", \"score\": 8}}"
            full_prompt = base_prompt + full_section_text + \
                f"\n\nמדוע {label.lower()} זה רלוונטי לסיטואציה? החזר JSON:\n{{\"advice\": \"הסבר\", \"score\": 8}}"

            names.append(name)
            full_prompts.append(full_prompt)  # לשימוש אם צריך fallback
            tasks.append(client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": short_prompt}],
                temperature=0.5
            ))

        replies = await asyncio.gather(*tasks)

        results = []
        for i, r in enumerate(replies):
            try:
                content = r.choices[0].message.content.strip()
                parsed = json.loads(content)
                score = parsed.get("score", 0)

                # fallback אם הציון נמוך
                if score < 6:
                    retry = await client_openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": full_prompts[i]}],
                        temperature=0.5
                    )
                    content = retry.choices[0].message.content.strip()
                    parsed = json.loads(content)
                    score = parsed.get("score", 0)

                advice = parsed.get("advice", "לא התקבל הסבר")
                results.append(
                    f"#### {label}: {names[i]}\n"
                    f"**הסבר**: {advice}\n"
                    f"**ציון רלוונטיות**: {score}/10"
                )
            except Exception as e:
                results.append(f"⚠️ שגיאה בפיענוח תגובת GPT: {e}")

        return results

    except Exception as e:
        return [f"שגיאה באחזור {label.lower()}ים: {e}"]


async def find_relevant_judgments(text):
    return await find_relevant_documents(text, judgment_index, judgment_collection, "CaseNumber", "Name", "Description", "פסק דין")

async def find_relevant_laws(text):
    return await find_relevant_documents(text, law_index, law_collection, "IsraelLawID", "Name", "Description", "חוק")

async def generate_response(user_input):
    context = "אתה עוזר משפטי מקצועי בדין הישראלי. ענה בקצרה ובמדויק."
    if "doc_summary" in st.session_state:
        context += f"\nהמסמך מסוכם כך: {st.session_state['doc_summary']}"
    judgments, laws = await asyncio.gather(
        find_relevant_judgments(user_input),
        find_relevant_laws(user_input)
    )
    context += "\n\n📚 פסקי דין רלוונטיים:\n" + "\n".join(judgments)
    context += "\n\n⚖️ חוקים רלוונטיים:\n" + "\n".join(laws)
    messages = [{"role": "system", "content": context}]
    messages += [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"][-5:]]
    messages.append({"role": "user", "content": user_input})
    response = await client_openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=700,
        temperature=0.7
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
st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)
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
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_messages()
        st.markdown('</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf", "docx"])
    if uploaded_file:
        st.session_state["uploaded_doc_text"] = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_docx(uploaded_file)
        st.success("המסמך נטען בהצלחה!")

    if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
        with st.spinner("GPT מסכם את המסמך..."):
            summary_prompt = f"""סכם את המסמך המשפטי הבא בקצרה:\n---\n{st.session_state['uploaded_doc_text']}"""
            response = asyncio.run(client_openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.5
            ))
            st.session_state["doc_summary"] = response.choices[0].message.content.strip()

    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    with st.form("chat_form"):
        user_input = st.text_area("הכנס שאלה משפטית", height=100)
        if st.form_submit_button("שלח שאלה") and user_input.strip():
            add_message("user", user_input)
            save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
            st.rerun()

    if st.session_state['messages'] and st.session_state['messages'][-1]['role'] == "user":
        typing = show_typing_realtime()
        response = asyncio.run(generate_response(st.session_state['messages'][-1]['content']))
        typing.empty()
        add_message("assistant", response)
        save_conversation(chat_id, st.session_state["user_name"], st.session_state["messages"])
        st.rerun()

    if st.button("🗑 נקה שיחה"):
        delete_conversation(chat_id)
        st.session_state["messages"] = []
        st.session_state["user_name"] = None
        st.rerun()
