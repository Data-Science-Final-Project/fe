
import os, sys, json, uuid, asyncio, re
from datetime import datetime

import streamlit as st
import torch, fitz, docx, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------------------------------------------------
# Pinecone & Mongo
# ------------------------------------------------------------
judgment_index = pinecone_client.Index("judgments-names")
law_index      = pinecone_client.Index("laws-names")

db = mongo_client[DATABASE_NAME]
judgment_collection = db["judgments"]
law_collection      = db["laws"]
conversation_coll   = db["conversations"]

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="Ask Mini Lawyer Suite", page_icon="⚖️", layout="wide")
COMMON_CSS = """
<style>
.chat-container{background:#1E1E1E;padding:20px;border-radius:10px}
.chat-header{color:#4CAF50;font-size:36px;font-weight:bold;text-align:center}
.user-message{background:#4CAF50;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px}
.bot-message{background:#44475a;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px}
.timestamp{font-size:0.75em;color:#bbb}
.law-card{border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:15px;box-shadow:0 2px 4px rgba(0,0,0,0.1);background:#f9f9f9}
.law-title{font-size:20px;font-weight:bold;color:#333}
.law-description{font-size:16px;color:#444;margin:10px 0}
.law-meta{font-size:14px;color:#555}
.stButton>button{background:#7ce38b;color:#fff;font-size:14px;border:none;padding:8px 16px;border-radius:5px;cursor:pointer}
.stButton>button:hover{background:#69d67a}
</style>
"""
st.markdown(COMMON_CSS, unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Choose module", ["Chat Assistant", "Legal Finder Assistant"])

# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

def get_localstorage_value(key):
    return st_js_blocking(f"return localStorage.getItem('{key}');", key="get_"+key)

def set_localstorage_value(key, val):
    st_js(f"localStorage.setItem('{key}', '{val}');")

def read_pdf(f):
    return "".join(p.get_text() for p in fitz.open(stream=f.read(), filetype="pdf"))

def read_docx(f):
    return "\n".join(p.text for p in docx.Document(f).paragraphs)

def add_message(role, content):
    st.session_state.setdefault("messages", []).append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%H:%M:%S")})

def display_messages():
    for m in st.session_state.get("messages", []):
        cls = "user-message" if m["role"]=="user" else "bot-message"
        st.markdown(f"<div class='{cls}'>{m['content']}<div class='timestamp'>{m['timestamp']}</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Chunk helpers
# ------------------------------------------------------------

def chunk_text(txt, max_len=450):
    """Split text into ~max_len‑char chunks on sentence boundaries."""
    sentences = re.split(r'(?:\.|\?|!)\s+', txt)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur)+len(s) > max_len and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks[:20]  # cap to 20 chunks for performance

async def pinecone_top_matches(embeddings, index, top_k=1):
    tasks = [index.query(vector=emb.tolist(), top_k=top_k, include_metadata=True) for emb in embeddings]
    return await asyncio.gather(*tasks)

# ------------------------------------------------------------
# Chat Assistant
# ------------------------------------------------------------

def chat_assistant():
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)
    # session init
    if "current_chat_id" not in st.session_state:
        cid = get_localstorage_value("MiniLawyerChatId") or str(uuid.uuid4())
        set_localstorage_value("MiniLawyerChatId", cid)
        st.session_state.current_chat_id = cid
    chat_id = st.session_state.current_chat_id
    if "messages" not in st.session_state:
        convo = conversation_coll.find_one({"local_storage_id": chat_id})
        st.session_state["messages"] = convo.get("messages", []) if convo else []
    st.session_state.setdefault("user_name", None)

    # name prompt
    if not st.session_state["user_name"]:
        with st.form("user_name_form"):
            n = st.text_input("הכנס שם להתחלת שיחה:")
            if st.form_submit_button("התחל שיחה") and n:
                st.session_state["user_name"] = n
                add_message("assistant", f"שלום {n}, איך אפשר לעזור?")
                conversation_coll.update_one({"local_storage_id":chat_id},{"$set":{"user_name":n,"messages":st.session_state["messages"]}}, upsert=True)
                st.rerun(); return
        return

    # display chat
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_messages()
        st.markdown('</div>', unsafe_allow_html=True)

    # upload document
    uploaded = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf","docx"])
    if uploaded:
        txt = read_pdf(uploaded) if uploaded.type=="application/pdf" else read_docx(uploaded)
        st.session_state["uploaded_doc_text"] = txt
        st.success("המסמך נטען בהצלחה!")

        # classify
        prompt_cls = f"סווג את המסמך הבא: חוזה, מכתב, תקנון, תביעה, פסק דין, אחר.\n---\n{txt[:1200]}\n---"
        resp = asyncio.run(client_async_openai.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"user","content":prompt_cls}],temperature=0,max_tokens=12))
        st.session_state["doc_type"] = resp.choices[0].message.content.strip()
        st.success(f"📄 סוג המסמך: {st.session_state['doc_type']}")

    # summarise
    if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
        with st.spinner("GPT מסכם את המסמך..."):
            doc_type = st.session_state.get("doc_type", "מסמך")
            sum_prompt = f"""אתה עורך-דין מומחה. סכּם את {doc_type} בשלושה חלקים:\n1. Executive Summary (≤120 מילים)\n2. נקודות עיקריות (בולטים)\n3. השלכות והמלצות\n---\n{st.session_state['uploaded_doc_text']}\n---"""
            r = asyncio.run(client_async_openai.chat.completions.create(model="gpt-4",messages=[{"role":"user","content":sum_prompt}],temperature=0.3,max_tokens=800))
            st.session_state["doc_summary"] = r.choices[0].message.content.strip()
            st.success("📃 המסמך סוכם.")
    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    # ------------------- retrieval with chunk‑to‑chunk ---------------------
    async def retrieve_sources(question:str):
        # embedding לשאלה
        q_emb = model.encode([question], normalize_embeddings=True)[0]
        # סעיפים מהמסמך (אם קיים)
        section_embs, section_texts = [], []
        if "uploaded_doc_text" in st.session_state:
            for sec in chunk_text(st.session_state["uploaded_doc_text"]):
                section_texts.append(sec)
                section_embs.append(model.encode([sec], normalize_embeddings=True)[0])
        else:
            section_embs = []

        # ----- שאילתות לפיינקון -----
        candidates = {"law":{}, "judgment":{}}

        async def add_candidate(match, kind):
            meta = match.get("metadata", {})
            score = match.get("score",0)
            doc_id = meta.get("IsraelLawID" if kind=="law" else "CaseNumber")
            if not doc_id: return
            d = law_collection.find_one({"IsraelLawID":doc_id}) if kind=="law" else judgment_collection.find_one({"CaseNumber":doc_id})
            if not d: return
            candidates[kind].setdefault(doc_id, {"doc":d, "scores":[]})["scores"].append(score)

        async def process_section(section_emb):
            res_law, res_jud = await asyncio.gather(
                law_index.query(vector=section_emb.tolist(), top_k=1, include_metadata=True),
                judgment_index.query(vector=section_emb.tolist(), top_k=1, include_metadata=True))
            for m in res_law.get("matches",[]): await add_candidate(m,"law")
            for m in res_jud.get("matches",[]): await add_candidate(m,"judgment")

        tasks=[process_section(e) for e in section_embs]
        await asyncio.gather(*tasks)

        # always include question embedding candidates as well (fallback)
        base_law  = law_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True)
        base_judg = judgment_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True)
        for m in base_law.get("matches", []): await add_candidate(m,"law")
        for m in base_judg.get("matches", []): await add_candidate(m,"judgment")

        # Rerank by avg score
        top_laws      = sorted(candidates["law"].values(), key=lambda x:-np.mean(x["scores"]))[:3]
        top_judgments = sorted(candidates["judgment"].values(), key=lambda x:-np.mean(x["scores"]))[:3]
        return [d["doc"] for d in top_laws], [d["doc"] for d in top_judgments]

    async def generate_answer(question:str):
        laws, judgments = await retrieve_sources(question)
        law_snip  = "\n\n".join(d.get("Description","")[:800] for d in laws)
        jud_snip  = "\n\n".join(d.get("Description","")[:800] for d in judgments)
        sys_prompt = "ענית אך ורק על סמך החוקים ופסקי הדין הבאים. צטט מקור במפורש.\n\n--- חוקים ---\n"+law_snip+"\n\n--- פסקי דין ---\n"+jud_snip+"\n\n"
        r = await client_async_openai.chat.completions.create(model="gpt-4o-mini",messages=[{"role":"system","content":sys_prompt},{"role":"user","content":question}],temperature=0.2)
        return r.choices[0].message.content.strip()

    async def handle_question(q, follow=False):
        ans = await generate_answer(q)
        add_message("user", q)
        add_message("assistant", ans)
        conversation_coll.update_one({"local_storage_id":chat_id},{"$set":{"messages":st.session_state["messages"],"user_name":st.session_state["user_name"]}}, upsert=True)
        st.rerun()

    with st.form("chat_form"):
        q = st.text_area("הכנס שאלה משפטית", height=100)
        if st.form_submit_button("שלח שאלה") and q.strip():
            asyncio.run(handle_question(q))

    if st.session_state.get("messages"):
        with st.form("follow_form", clear_on_submit=True):
            q2 = st.text_input("🔁 שאל שאלה נוספת:")
            if st.form_submit_button("שלח") and q2.strip():
                asyncio.run(handle_question(q2, follow=True))

    if st.button("🗑 נקה שיחה"):
        conversation_coll.delete_one({"local_storage_id":chat_id}); st_js("localStorage.clear();")
        st.session_state.clear(); st.rerun()

# ------------------------------------------------------------
# Legal Finder Assistant (ללא שינוי לוגי)
# ------------------------------------------------------------

def load_document_details(kind, doc_id):
    coll = judgment_collection if kind=="Judgment" else law_collection
    key  = "CaseNumber" if kind=="Judgment" else "IsraelLawID"
    return coll.find_one({key:doc_id})

def get_explanation(scen, doc, kind):
    name, desc = doc.get("Name",""), doc.get("Description","")
    prompt = f"בהתבסס על הסצנריו:\n{scen}\n\nועל {('פסק הדין','החוק')[kind=='Law']} הבא:\n{name} – {desc}\n\nהסבר בקצרה מדוע רלוונטי ודרג 0‑10 (JSON)."
    try:
        r = client_sync_openai.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role":"user","content":prompt}],temperature=0.7)
        return json.loads(r.choices[0].message.content.strip())
    except Exception as e:
        return {"advice":f"Error: {e}","score":"N/A"}

def legal_finder_assistant():
    st.title("Legal Finder Assistant")

    kind = st.selectbox("Choose what to search", ["Judgment", "Law"])
    scen = st.text_area("Describe your scenario")

    if st.button("Find Suitable Results") and scen:
        q_emb = model.encode([scen], normalize_embeddings=True)[0]
        index = judgment_index if kind == "Judgment" else law_index
        id_key = "CaseNumber" if kind == "Judgment" else "IsraelLawID"

        res = index.query(vector=q_emb.tolist(), top_k=5, include_metadata=True)
        matches = res.get("matches", [])
        if not matches:
            st.info("No matches found.")
            return

        for m in matches:
            doc_id = m.get("metadata", {}).get(id_key)
            if not doc_id:
                continue
            doc = load_document_details(kind, doc_id)
            if not doc:
                continue

            name      = doc.get("Name", "No Name")
            desc      = doc.get("Description", "N/A")
            date_lbl  = "DecisionDate" if kind == "Judgment" else "PublicationDate"

            # optional: show procedure type for judgments
            extra_label = "ProcedureType" if kind == "Judgment" else None
            extra_html  = (
                f"<div class='law-meta'>Procedure Type: {doc.get(extra_label, 'N/A')}</div>"
                if extra_label else ""
            )

            st.markdown(
                f"<div class='law-card'>"
                f"<div class='law-title'>{name} (ID: {doc_id})</div>"
                f"<div class='law-description'>{desc}</div>"
                f"<div class='law-meta'>{date_lbl}: {doc.get(date_lbl, 'N/A')}</div>"
                f"{extra_html}"
                f"</div>",
                unsafe_allow_html=True  
            )
