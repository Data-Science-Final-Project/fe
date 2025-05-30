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
    /* Chat Container and Header */
    .chat-container {
        background: #1C1C2E;
        padding: 20px;
        border-radius: 10px;
    }

    .chat-header {
        color: #9F7AEA;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }

    /* Messages */
    .user-message {
        background: #9F7AEA;
        color: #ECECEC;
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
    }

    .bot-message {
        background: #2A2A40;
        color: #ECECEC;
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
    }

    /* Timestamp */
    .timestamp {
        font-size: 0.75em;
        color: #AAAAAA;
    }

    /* Law Card */
    .law-card {
        border: 1px solid #3B3B52;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        background: #2A2A40;
    }

    .law-title {
        font-size: 20px;
        font-weight: bold;
        color: #ECECEC;
    }

    .law-description {
        font-size: 16px;
        color: #CCCCCC;
        margin: 10px 0;
    }

    .law-meta {
        font-size: 14px;
        color: #AAAAAA;
    }

    /* Buttons */
    .stButton > button {
        background: #9F7AEA !important;
        color: #FFFFFF !important;
        font-size: 14px;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }

    .stButton > button:hover {
        background: #805AD5 !important;
    }
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
        st.markdown(
            f"<div class='{cls}'>{m['content']}<div class='timestamp'>{m['timestamp']}</div></div>",
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# Chunk helpers
# ------------------------------------------------------------
def chunk_text(txt, max_len=450):
    sentences = re.split(r'(?:\.|\?|!)\s+', txt)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > max_len and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks[:20]

async def pinecone_top_matches(embeddings, index, top_k=1):
    tasks = [index.query(vector=emb.tolist(), top_k=top_k, include_metadata=True) for emb in embeddings]
    return await asyncio.gather(*tasks)

# ------------------------------------------------------------
# Chat Assistant
# ------------------------------------------------------------
def chat_assistant():
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)

    # ---------- session & history ----------
    if "current_chat_id" not in st.session_state:
        cid = get_localstorage_value("MiniLawyerChatId") or str(uuid.uuid4())
        set_localstorage_value("MiniLawyerChatId", cid)
        st.session_state.current_chat_id = cid
    chat_id = st.session_state.current_chat_id

    if "messages" not in st.session_state:
        convo = conversation_coll.find_one({"local_storage_id": chat_id})
        st.session_state["messages"] = convo.get("messages", []) if convo else []
    st.session_state.setdefault("user_name", None)

    # ---------- user name ----------
    if not st.session_state["user_name"]:
        with st.form("user_name_form"):
            n = st.text_input("הכנס שם להתחלת שיחה:")
            if st.form_submit_button("התחל שיחה") and n:
                st.session_state["user_name"] = n
                add_message("assistant", f"שלום {n}, איך אפשר לעזור?")
                conversation_coll.update_one(
                    {"local_storage_id": chat_id},
                    {"$set": {"user_name": n, "messages": st.session_state["messages"]}},
                    upsert=True,
                )
                st.rerun()
        return

    # ---------- chat window ----------
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        display_messages()
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- helper to strip english ----------
    def strip_english_lines(text: str) -> str:
        return "\n".join([ln for ln in text.splitlines() if re.search(r"[א-ת]", ln)])

    # ---------- file upload ----------
    uploaded = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf", "docx"])
    if uploaded:
        raw_txt = read_pdf(uploaded) if uploaded.type == "application/pdf" else read_docx(uploaded)
        clean_txt = strip_english_lines(raw_txt)
        st.session_state["uploaded_doc_text"] = clean_txt
        st.success("המסמך נטען – שורות באנגלית סוננו!")

        cls_system = (
            "אתה מסווג מסמכים משפטיים. החזר *בדיוק* אחת מהקטגוריות:\n"
            "חוזה, מכתב, תקנון, תביעה, פסק דין"
        )
        resp = asyncio.run(
            client_async_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": cls_system},
                    {"role": "user", "content": clean_txt[:1500]},
                ],
                temperature=0,
                max_tokens=5,
            )
        )
        st.session_state["doc_type"] = resp.choices[0].message.content.strip()
        st.success(f"📄 סוג המסמך שזוהה: {st.session_state['doc_type']}")

    # ---------- summarise ----------
    if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
        with st.spinner("GPT מסכם את המסמך..."):
            doc_type = st.session_state.get("doc_type", "מסמך")
            sum_prompt = (
                f"אתה עורך-דין מומחה. סכם את {doc_type} בעברית בשלושה חלקים:\n"
                "1. תקציר מנהלים – עד 100 מילים.\n"
                "2. נקודות עיקריות – רשימת בולטים (מועדים, סכומים, סיכונים).\n"
                "3. השלכות והמלצות מעשיות.\n"
                "השתמש בעברית בלבד וללא מילים באנגלית.\n"
                "—\n" + st.session_state["uploaded_doc_text"] + "\n—"
            )
            r = asyncio.run(
                client_async_openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": sum_prompt}],
                    temperature=0.1,
                    max_tokens=700,
                )
            )
            st.session_state["doc_summary"] = r.choices[0].message.content.strip()
            st.success("📃 המסמך סוכם בהצלחה.")

    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])

    # ---------- retrieval ----------
    async def retrieve_sources(question: str):
        q_emb = model.encode([question], normalize_embeddings=True)[0]
        section_embs = (
            [model.encode([sec], normalize_embeddings=True)[0] for sec in chunk_text(st.session_state["uploaded_doc_text"])]
            if "uploaded_doc_text" in st.session_state else []
        )
        candidates = {"law": {}, "judgment": {}}

        async def add_candidate(match, kind):
            meta = match.get("metadata", {})
            score = match.get("score", 0)
            doc_id = meta.get("IsraelLawID" if kind == "law" else "CaseNumber")
            if not doc_id:
                return
            doc = (law_collection.find_one({"IsraelLawID": doc_id}) if kind == "law"
                   else judgment_collection.find_one({"CaseNumber": doc_id}))
            if not doc:
                return
            candidates[kind].setdefault(doc_id, {"doc": doc, "scores": []})["scores"].append(score)

        async def process_section(emb):
            res_law, res_jud = await asyncio.gather(
                asyncio.to_thread(law_index.query, vector=emb.tolist(), top_k=1, include_metadata=True),
                asyncio.to_thread(judgment_index.query, vector=emb.tolist(), top_k=1, include_metadata=True),
            )
            for m in res_law.get("matches", []):
                await add_candidate(m, "law")
            for m in res_jud.get("matches", []):
                await add_candidate(m, "judgment")

        await asyncio.gather(*(process_section(e) for e in section_embs))

        for m in law_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True).get("matches", []):
            await add_candidate(m, "law")
        for m in judgment_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True).get("matches", []):
            await add_candidate(m, "judgment")

        top_laws = sorted(candidates["law"].values(), key=lambda x: -np.mean(x["scores"]))[:3]
        top_judgments = sorted(candidates["judgment"].values(), key=lambda x: -np.mean(x["scores"]))[:3]

        return [d["doc"] for d in top_laws], [d["doc"] for d in top_judgments]

    # ---------- answer ----------
    async def generate_answer(question: str):
        laws, judgments = await retrieve_sources(question)
        doc_text = st.session_state.get("uploaded_doc_text", "")[:1500]

        if not laws and not judgments and not doc_text:
            return "לא נמצאו חוקים, פסקי-דין או מסמך רלוונטי למתן תשובה מוסמכת."

        law_snip = "\n\n".join(d.get("Description", "")[:800] for d in laws)
        jud_snip = "\n\n".join(d.get("Description", "")[:800] for d in judgments)

        sys_prompt = (
            "אתה עורך-דין ישראלי. עליך לנסח תשובה משפטית מקצועית ומנומקת בעברית בלבד.\n"
            "חובה להסתמך אך ורק על החומר המצוטט מטה: המסמך שהועלה, חוקים ופסקי-דין. "
            "אין להמציא מידע, ואין לשער.\n"
            "אם אין מספיק מידע – כתוב 'אין לי מידע מוסמך לענות'.\n"
            "יש לציין מקור ברור לכל טענה (שם חוק / פס״ד + מספר סעיף / עמוד).\n\n"
            "--- חלקים רלוונטיים מהמסמך שהועלה ---\n" + doc_text +
            "\n\n--- חוקים ---\n" + law_snip +
            "\n\n--- פסקי דין ---\n" + jud_snip + "\n\n"
        )

        r = await client_async_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=700,
        )
        return r.choices[0].message.content.strip()

    # ---------- handle question ----------
    async def handle_question(q):
        ans = await generate_answer(q)
        add_message("user", q)
        add_message("assistant", ans)
        conversation_coll.update_one(
            {"local_storage_id": chat_id},
            {"$set": {"messages": st.session_state["messages"], "user_name": st.session_state["user_name"]}},
            upsert=True,
        )
        st.rerun()

    # ---------- chat input ----------
    with st.form("chat_form", clear_on_submit=True):
        q = st.text_area("הקלד כאן שאלה משפטית (גם שאלות נוספות)", height=100)
        if st.form_submit_button("שלח") and q.strip():
            asyncio.run(handle_question(q.strip()))

    # ---------- clear ----------
    if st.button("🗑 נקה שיחה"):
        conversation_coll.delete_one({"local_storage_id": chat_id})
        st_js("localStorage.clear();")
        st.session_state.clear()
        st.rerun()

# ------------------------------------
# Legal Finder Assistant
# ------------------------------------
def load_document_details(kind, doc_id):
    coll = judgment_collection if kind == "Judgment" else law_collection
    key  = "CaseNumber" if kind == "Judgment" else "IsraelLawID"
    return coll.find_one({key: doc_id})

def get_explanation(scenario, doc, kind):
    name = doc.get("Name", "")
    desc = doc.get("Description", "")

    if kind == "Judgment":
        prompt = f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי פסק הדין הבא:
שם: {name}
תיאור: {desc}

אנא הסבר בצורה תמציתית ומקצועית מדוע פסק דין זה יכול לעזור למקרה זה,
והערך אותו בסולם 0-10 (0 = לא עוזר כלל, 10 = מתאים במדויק).
**אל תיתן לרוב המסמכים ציון 9 – היה מגוון!**
החזר JSON בלבד, לדוגמה:
{{
  "advice": "הסבר מקצועי בעברית",
  "score": 8
}}
אין להוסיף טקסט נוסף.
"""
    else:  # kind == "Law"
        prompt = f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי החוק הבא:
שם: {name}
תיאור: {desc}

אנא הסבר בצורה תמציתית ומקצועית מדוע חוק זה יכול לעזור למקרה זה,
והערך אותו בסולם 0-10 (0 = לא קשור, 10 = מתאים כמו כפפה).
**אל תיתן לרוב החוקים ציון 9 – היה מגוון!**
החזר JSON בלבד, לדוגמה:
{{
  "advice": "הסבר תמציתי ומקצועי בעברית",
  "score": 7
}}
אין להוסיף טקסט נוסף.
"""

    try:
        response = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Error from GPT: {e}")
        return {"advice": "לא ניתן לקבל הסבר בשלב זה.", "score": "N/A"}

def legal_finder_assistant():
    st.title("Legal Finder Assistant")

    kind = st.selectbox("Choose what to search", ["Judgment", "Law"])
    scen = st.text_area("Describe your scenario")

    if st.button("Find Suitable Results") and scen:
        # ---------- Pinecone similarity search ---------------------------------------
        q_emb  = model.encode([scen], normalize_embeddings=True)[0]
        index  = judgment_index if kind == "Judgment" else law_index
        id_key = "CaseNumber" if kind == "Judgment" else "IsraelLawID"

        res     = index.query(vector=q_emb.tolist(), top_k=5, include_metadata=True)
        matches = res.get("matches", [])
        if not matches:
            st.info("No matches found.")
            return

        # ---------- loop over matches ----------------------------------------
        for m in matches:
            doc_id = m.get("metadata", {}).get(id_key)
            if not doc_id:
                continue
            doc = load_document_details(kind, doc_id)
            if not doc:
                continue

            name     = doc.get("Name", "No Name")
            desc     = doc.get("Description", "N/A")
            date_lbl = "DecisionDate" if kind == "Judgment" else "PublicationDate"

            extra_html = (
                f"<div class='law-meta'>Procedure Type: {doc.get('ProcedureType','N/A')}</div>"
                if kind == "Judgment" else ""
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

            # ---------- GPT Advise (Hidden Score) -----------------------------
            with st.spinner("Getting explanation..."):
                result = get_explanation(scen, doc, kind)

            advice = result.get("advice", "")
            # score  = result.get("score", "N/A")

            st.markdown(
                f"<span style='color:red;'>עצת האתר: {advice}</span>",
                unsafe_allow_html=True
            )

            # ---------- full JSON toggle -------------------------------
            with st.expander(f"View Full Details for {doc_id}"):
                st.json(doc)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if app_mode == "Chat Assistant":
    chat_assistant()
else:
    legal_finder_assistant()
