import os, sys, json, uuid, asyncio, re
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="Ask Mini Lawyer Suite", page_icon="⚖️", layout="wide")

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
DATABASE_NAME  = os.getenv("DATABASE_NAME")
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

db                    = mongo_client[DATABASE_NAME]
judgment_collection   = db["judgments"]
law_collection        = db["laws"]
conversation_coll     = db["conversations"]

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

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

app_mode = "Chat Assistant"


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def get_localstorage_value(key: str):
    return st_js_blocking(f"return localStorage.getItem('{key}');", key="get_"+key)

def set_localstorage_value(key: str, val: str):
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
        cls = "user-message" if m["role"] == "user" else "bot-message"
        st.markdown(
            f"<div class='{cls}'>{m['content']}<div class='timestamp'>{m['timestamp']}</div></div>",
            unsafe_allow_html=True
        )

# ------------------------------------------------------------
# Text helpers
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

# ------------------------------------------------------------
# Robust document classifier
# ------------------------------------------------------------
CLS_PROMPT = """
אתה מסווג מסמכים משפטיים. החזר *מילה אחת בלבד* מתוך הרשימה:
מכתב_פיטורין, חוזה, תקנון, תביעה, פסק_דין, מכתב_אחר

• **מכתב_פיטורין** – הודעה על סיום העסקה (termination notice), כוללת תאריך סיום, פיצויי פיטורין, הודעה מוקדמת.
• **תביעה** – כתב תביעה לבית-משפט, עם תובע/נתבע וסעד מבוקש.
• **חוזה**  – הסכם בין צדדים עם סעיפים הדדיים.
• **תקנון** – כללים/נהלים כלליים (לרוב פורמט PDF של חברה/עמותה).
• **פסק_דין** – החלטה סופית של בית-משפט.
• **מכתב_אחר** – כל מכתב רשמי שלא מתאים לקטגוריות לעיל.

דוגמה:  
«הריני להודיעך על הפסקת עבודתך בחברה…» → מכתב_פיטורין  
«בית-הדין הנכבד מתבקש לחייב את הנתבע…» → תביעה

הטקסט:
"""
KEYWORD_OVERRIDES = {
    r"פיטור(ין|ים)|סיום העסק(ה|תך)|הודעה על סיום|חשבונ(ך|ו) ייערך|termination notice|פיטורים|שימוע": "מכתב_פיטורין",
    r"כתב\s+תביעה|הנתבע|התובע": "תביעה",
}

def classify_doc(clean_txt: str, debug_mode=False) -> str:
    sample = clean_txt[:800] + "\n\n---\n\n" + clean_txt[-800:]
    try:
        resp = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": CLS_PROMPT + sample}],
            temperature=0.0,
            max_tokens=5,
        )
        cat = resp.choices[0].message.content.strip()
    except Exception:
        cat = "מכתב_אחר"
    if debug_mode:
        st.write("GPT סיווג ראשוני:", cat)
    # override heuristic
    for pattern, override in KEYWORD_OVERRIDES.items():
        if re.search(pattern, clean_txt, re.I):
            if debug_mode:
                st.write(f"Override מופעל: {pattern} => {override}")
            cat = override
            break
    return cat


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
        return "\n".join([ln for ln in text.splitlines() if len(re.findall(r"[א-ת]", ln)) > 6])

    # ---------- file upload ----------
    uploaded = st.file_uploader("📄 העלה מסמך משפטי", type=["pdf", "docx"])
    if uploaded:
        raw_txt   = read_pdf(uploaded) if uploaded.type == "application/pdf" else read_docx(uploaded)
        clean_txt = strip_english_lines(raw_txt)
        st.session_state["uploaded_doc_text"] = clean_txt
        st.success("המסמך נטען – שורות באנגלית סוננו!")

        # === classification ===
        st.session_state["doc_type"] = classify_doc(clean_txt)
        st.success(f"📄 סוג המסמך שזוהה: {st.session_state['doc_type']}")

    # ---------- summarise ----------
    if "uploaded_doc_text" in st.session_state and st.button("📋 סכם את המסמך"):
        with st.spinner("MiniLawyer מסכם את המסמך..."):
            doc_type = st.session_state.get("doc_type", "מסמך")
            sum_prompt = (
                f"אתה עורך-דין מומחה. סכם את המסמך הבא בעברית בלבד ובמבנה קבוע:\n"
                f"כותרת: סיכום {doc_type}\n"
                f"1. תקציר מנהלים – עד 100 מילים.\n"
                f"2. נקודות עיקריות – רשימת בולטים (מועדים, סכומים, סיכונים).\n"
                f"3. השלכות והמלצות מעשיות.\n"
                f"עליך להשתמש בעברית בלבד. אין להכניס אף מילה באנגלית.\n"
                f"אם המסמך הוא מכתב פיטורין, הדגש זאת במפורש בכותרת.\n"
                f"—\n" + st.session_state["uploaded_doc_text"] + "\n—"
            )
            r = asyncio.run(
                client_async_openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": sum_prompt}],
                    temperature=0.1,
                    max_tokens=700,
                )
            )
            summary = r.choices[0].message.content.strip()
            # סינון שורות ללא עברית
            summary_hebrew = "\n".join([ln for ln in summary.splitlines() if re.search(r"[א-ת]", ln)])
            st.session_state["doc_summary"] = summary_hebrew
            st.success("📃 המסמך סוכם בהצלחה.")

    if "doc_summary" in st.session_state:
        st.markdown("### סיכום המסמך:")
        st.info(st.session_state["doc_summary"])


    # ---------- retrieval ----------
    async def retrieve_sources(question: str):
        # Generating embedding for the question and document sections
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

        # חיפוש לפי מקטעים מהמסמך
        
        async def process_section(emb):
            res_law, res_jud = await asyncio.gather(
                asyncio.to_thread(law_index.query,      vector=emb.tolist(), top_k=2, include_metadata=True),
                asyncio.to_thread(judgment_index.query, vector=emb.tolist(), top_k=2, include_metadata=True),
            )
            for m in res_law.get("matches", []):
                await add_candidate(m, "law")
            for m in res_jud.get("matches", []):
                await add_candidate(m, "judgment")

        await asyncio.gather(*(process_section(e) for e in section_embs))

        # חיפוש לפי השאלה עם top_k 
        for m in law_index.query(vector=q_emb.tolist(), top_k=7, include_metadata=True).get("matches", []):
            await add_candidate(m, "law")
        for m in judgment_index.query(vector=q_emb.tolist(), top_k=7, include_metadata=True).get("matches", []):
            await add_candidate(m, "judgment")

        # מיון דירוג לפי ממוצע score
        top_laws = sorted(candidates["law"].values(), key=lambda x: -np.mean(x["scores"]))[:3]
        top_judgments = sorted(candidates["judgment"].values(), key=lambda x: -np.mean(x["scores"]))[:3]
        return [d["doc"] for d in top_laws], [d["doc"] for d in top_judgments]

        # ---------- answer ----------
    async def generate_answer(question: str):
        laws, judgments = await retrieve_sources(question)
        doc_text = st.session_state.get("uploaded_doc_text", "")[:1500]

        if not laws and not judgments and not doc_text:
            return "לא נמצאו חוקים, פסקי-דין או מסמך רלוונטי למתן תשובה מוסמכת."

        # ניצור snippet מפורמט - כולל שם חוק/פס"ד והמספר
        def get_law_snip(law):
            name = law.get("Name", "חוק לא מזוהה")
            desc = law.get("Description", "")[:800]
            law_id = law.get("IsraelLawID", "")
            return f"שם החוק: {name} (מס' מזהה: {law_id})\n{desc}"

        def get_judgment_snip(judgment):
            name = judgment.get("Name", "פסק דין לא מזוהה")
            desc = judgment.get("Description", "")[:800]
            num = judgment.get("CaseNumber", "")
            return f"שם פסק הדין: {name} (מס' תיק: {num})\n{desc}"

        law_snip = "\n\n".join(get_law_snip(d) for d in laws)
        jud_snip = "\n\n".join(get_judgment_snip(d) for d in judgments)

        sys_prompt = (
            "אתה עורך-דין ישראלי מקצועי, אמין, קפדן ובלתי מתפשר על דיוק. תפקידך לנסח תשובה משפטית מקצועית, מנומקת, מפורטת, מעשית וברורה – אך ורק בעברית.\n"
            "עליך להתבסס אך ורק על המידע שמופיע למטה (המסמך, החוקים, פסקי הדין, ותשובות המשתמש מהשיחה הקודמת). **אין להמציא מידע, אין לנחש, ואין להניח הנחות – גם אם חסר מידע.** פעל כמו עורך דין מנוסה ומחמיר.\n"
            "לפני כל תשובה, עבור בקפדנות על כל ההודעות, העובדות והתשובות שנמסרו בשיחה עד כה, ופרט מהם כל הפרטים/העובדות/המסמכים הדרושים לפי הדין ולפי סוג הבעיה (לרבות: צדדים, מועדים, מסמכים, נסיבות, פרטי עסקה, או כל פרט קריטי לסוגיה)."
            "ציין ליד כל פרט אם ידוע ומה המקור, או שהוא חסר (ומה בדיוק חסר). סכם ברשימה מסודרת.\n"
            "אם חסר אפילו פרט מהותי אחד – שאל **בסבב אחד בלבד** את כל שאלות ההבהרה ההכרחיות, בקצרה, בבהירות, ובשום אופן לא לחזור עליהן בשיחה, גם לא בניסוח שונה. אין לשאול שאלות שוליות או לא קריטיות.\n"
            "לפני שלב הבהרות, סכם בקצרה את כל העובדות הידועות שנמסרו בשיחה, ואחריהן את השאלות החסרות.\n"
            "אם עדיין חסרים פרטים מהותיים – הדגש במפורש את המגבלות של כל תשובה שתספק.\n"
            "לאחר קבלת כל הפרטים החיוניים, עבור למתן תשובה מקצועית:\n"
            "- פרט את כל הזכויות, החובות, ההגנות, ההליכים האפשריים והסיכונים.\n"
            "- עבור כל טענה – הפנה במדויק למקור: שם החוק, מספר הסעיף (למשל 'סעיף 6 לחוק שעות עבודה ומנוחה, התשי\"א-1951'), או שם ומספר פסק הדין (למשל 'ע\"ע 1234/56 פלוני נ' אלמוני'). **לאחר כל מקור, הסבר במפורש וללא קיצורים מדוע הוא רלוונטי למקרה ולמה הוא תומך בטענה שלך.**\n"
            "- אין להשיב בתשובה כללית, תאורטית או לאקונית – יש להציג מענה קונקרטי, מפורט ומדויק לפי הנתונים שנמסרו.\n"
            "- אם חסר מידע קריטי – פרט מה חסר, מה לא ניתן לקבוע בלעדיו, והמלץ במדויק אילו צעדים או מסמכים לאסוף להמשך.\n"
            "- בסוף כל תשובה מקצועית, הנחה את המשתמש **בפעולה המעשית המדויקת ביותר**: מה כדאי לו לעשות, אילו מסמכים לשמור, למי לפנות, מה לבדוק, מה לדרוש, או מהם הסיכונים המרכזיים שעליו להקפיד עליהם.\n"
            "- אם מדובר במקרה רגיש, גבולי, מורכב או פלילי – המלץ במפורש לפנות לייעוץ פרטני אצל עורך דין אנושי, והסבר מדוע.\n"
            "- התשובות שלך חייבות להתאים **לכל תחום משפטי**: אזרחי, מסחרי, עבודה, משפחה, קניין, חוזים, נזיקין, ירושה, מנהלי, פלילי, הגבלים עסקיים, קניין רוחני וכל תחום נוסף.\n"
            "פתח כל שלב הבהרות ב-'להמשך ייעוץ אנא השב על השאלות הבאות:' ופרט את כל השאלות החסרות. לאחר מכן, כאשר מתקבלות התשובות, עבור לתשובה מלאה.\n"
            "אין לשאול שאלות כפולות או חוזרות – גם לא בניסוח אחר. אסור להתעלם ממידע שניתן, גם אם לא היה בפורמט מלא.\n"
            "במהלך הסקת המסקנות, חפש ראשית תשובות במסמך שהועלה: נסה לאתר בו עובדות, מועדים, תנאים או סעיפים שיכולים לספק מענה לשאלות לפני שאתה פונה למשתמש.\n"
            "\n--- חלקים רלוונטיים מהמסמך ---\n" + doc_text +
            "\n\n--- חוקים ---\n" + law_snip +
            "\n\n--- פסקי דין ---\n" + jud_snip + "\n\n"
        )





        
        messages = [{"role": "system", "content": sys_prompt}]
        for msg in st.session_state["messages"]:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        r = await client_async_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,   
            temperature=0,
            max_tokens=1200,
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

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
chat_assistant()
