# ───────────────────── imports ─────────────────────
import os, re, json, uuid, asyncio
from datetime import datetime

import streamlit as st
import torch, fitz, docx, numpy as np
from bidi.algorithm import get_display
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from streamlit_js import st_js, st_js_blocking

from app_resources import mongo_client, pinecone_client, model

# ─────────────────── env / clients ─────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_AI")
DATABASE_NAME  = os.getenv("DATABASE_NAME")

client_async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
client_sync_openai  = OpenAI(api_key=OPENAI_API_KEY)

judgment_index = pinecone_client.Index("judgments-names")
law_index      = pinecone_client.Index("laws-names")

db = mongo_client[DATABASE_NAME]
judgment_coll = db["judgments"]
law_coll      = db["laws"]
conv_coll     = db["conversations"]

torch.classes.__path__ = []
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ───────────────────── UI CSS ──────────────────────
st.set_page_config("Ask Mini Lawyer", "⚖️", layout="wide")
st.markdown("""
<style>
.chat-container{background:#1E1E1E;padding:20px;border-radius:10px}
.chat-header{color:#4CAF50;font-size:36px;font-weight:bold;text-align:center}
.user-message{background:#4CAF50;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px}
.bot-message{background:#44475a;color:#ecf2f8;padding:10px;border-radius:10px;margin:10px}
.timestamp{font-size:0.75em;color:#bbb}
.law-card{border:1px solid #e0e0e0;border-radius:10px;padding:20px;margin-bottom:15px;
          box-shadow:0 2px 4px rgba(0,0,0,0.1);background:#f9f9f9}
.law-title{font-size:20px;font-weight:bold;color:#333}
.law-description{font-size:16px;color:#444;margin:10px 0}
.law-meta{font-size:14px;color:#555}
.stButton>button{background:#7ce38b;color:#fff;font-size:14px;border:none;padding:8px 16px;border-radius:5px;cursor:pointer}
.stButton>button:hover{background:#69d67a}
</style>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Choose module", ["Chat Assistant", "Legal Finder"])

# ─────────────────── helpers ────────────────────
# localStorage
def ls_get(key: str) -> str | None:
    with st.container():
        st.markdown("<div style='display:none'>", unsafe_allow_html=True)
        value = st_js_blocking(f"return localStorage.getItem('{key}');", key=f"ls_{key}")
        st.markdown("</div>", unsafe_allow_html=True)
    return value

def ls_set(key: str, value: str) -> None:
    st_js(f"localStorage.setItem('{key}', '{value}');")

# RTL text normalization
heb = re.compile(r"[א-ת]")  
SALUT = re.compile(r"\b(לכבוד|מר|מר\.?|גב'?|גברת|ד\"ר|ד\"ר\.?)\b")

def rtl_norm(text: str) -> str:
    if not heb.search(text):
        return text
    try:
        text = get_display(text)
    except Exception:
        text = " ".join(word[::-1] if heb.search(word) else word for word in text.split())  
    text = SALUT.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# File readers
read_pdf = lambda f: "".join(rtl_norm(p.get_text()) for p in fitz.open(stream=f.read(), filetype="pdf"))
read_docx = lambda f: "\n".join(rtl_norm(p.text) for p in docx.Document(f).paragraphs if p.text.strip())

# Chat UI
def add_msg(role: str, text: str) -> None:
    st.session_state.setdefault("messages", []).append({
        "role": role,
        "content": text,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

def show_msgs() -> None:
    for msg in st.session_state.get("messages", []):
        css_class = "user-message" if msg["role"] == "user" else "bot-message"
        st.markdown(
            f"<div class='{css_class}'>{msg['content']}<div class='timestamp'>{msg['timestamp']}</div></div>",
            unsafe_allow_html=True,
        )

# Text processing
def chunk_text(text: str, max_len: int = 450) -> list[str]:
    if not isinstance(text, str):
        text = str(text)
    sentences = re.split(r"(?:\.|\?|!)\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_len and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks[:20]

contains_en = lambda s: bool(re.search(r"[A-Za-z]", s))

def ensure_he(text: str) -> str:
    english_word_count = sum(contains_en(word) for word in text.split())
    if english_word_count <= 3:
        return text
    response = client_sync_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "תרגם לעברית מלאה:\n" + text}],
        temperature=0,
        max_tokens=len(text) // 2,
    )
    return response.choices[0].message.content.strip()




# ───────────── classification ─────────────
DOC_LABELS = {
    "חוזה_עבודה":  "הסכם בין מעסיק לעובד או מועמד לעבודה",
    "חוזה_שכירות": "הסכם להשכרת דירה, משרד או נכס אחר",
    "חוזה_שירות":  "הסכם בין מזמין לספק שירות",
    "מכתב_פיטורין": "הודעה על סיום העסקה או הפסקת עבודה",
    "מכתב_התראה":  "מכתב דרישה או אזהרה לפני נקיטת הליכים",
    "תקנון":       "מסמך כללי חובות וזכויות",
    "NDA":         "הסכם סודיות ואי-גילוי",
    "כתב_תביעה":   "מסמך פתיחת הליך בבית-משפט",
    "כתב_הגנה":    "תגובה לכתב תביעה",
    "פסק_דין":     "הכרעת בית-משפט",
    "מסמך_אחר":    "כל מסמך משפטי אחר"
}

CLS_EXAMPLES = [
    ("הננו להודיעך בזאת כי העסקתך תסתיים בתאריך …", "מכתב_פיטורין"),
    ("העובד מתחייב לשמור בסוד כל מידע",              "חוזה_עבודה"),
    ("השוכר מתחייב להחזיר את הנכס כשהוא נקי",        "חוזה_שכירות"),
    ("הצדדים מסכימים שלא לגלות מידע סודי",           "NDA"),
    ("הנתבע ביצע רשלנות … לפיכך מתבקש ביהמ״ש",        "כתב_תביעה")
]

CLS_SYS = (
    "אתה מסווג מסמכים משפטיים. החזר JSON במבנה:\n"
    '{"label":"<LABEL>","confidence":0-100}\n'
    f"עליך לבחור רק מתוך: {', '.join(DOC_LABELS.keys())}."
)

def classify_doc(txt: str) -> str:
    # Ensure txt is a string to prevent crashes (e.g., from None or file objects)
    if not isinstance(txt, str):
        try:
            txt = txt.decode("utf-8")
        except Exception:
            txt = str(txt)

    chunks = chunk_text(txt, max_len=500)[:3] or [txt[:1500]]
    msgs = [{"role": "system", "content": CLS_SYS}]
    
    for eg, lbl in CLS_EXAMPLES:
        msgs += [
            {"role": "user", "content": eg},
            {"role": "assistant", "content": json.dumps({"label": lbl, "confidence": 95})}
        ]
    
    for c in chunks:
        msgs.append({"role": "user", "content": c})

    try:
        r = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            temperature=0,
            max_tokens=20
        )
        j = json.loads(r.choices[0].message.content)
        return j["label"] if j.get("label") in DOC_LABELS else "מסמך_אחר"
    except Exception:
        return "מסמך_אחר"


# ────────────── prompts ──────────────
H = lambda s: s + "  **ענה בעברית מלאה וללא מילים באנגלית. ציין מקור ממוספר (חוק או פס״ד) אחרי כל קביעה.**"
PROMPTS = {
    "מכתב_פיטורין": dict(
        summary=H("סכם מכתב פיטורין: 1. פרטי עובד ותאריכים, 2. זכויות ותשלומים, 3. צעדים מומלצים."),
        answer=H("אתה עו\"ד דיני-עבודה. השב רק על סמך המכתב וחוקי עבודה רלוונטיים.")
    ),
    "חוזה_עבודה": dict(
        summary=H("סכם חוזה עבודה: 1. תנאי העסקה, 2. סעיפי סודיות ואי-תחרות, 3. סיכונים והמלצות."),
        answer=H("אתה עו\"ד דיני-עבודה. נתח את סעיפי החוזה.")
    ),
    "תקנון": dict(
        summary=H("סכם תקנון/מדיניות: 1. מטרות, 2. זכויות/חובות, 3. סיכונים לאי-ציות."),
        answer=H("אתה עו\"ד חברות. הסבר תוקף סעיפי התקנון.")
    ),
    "כתב_תביעה": dict(
        summary=H("סכם כתב תביעה: 1. עילות, 2. סעדים, 3. לוח זמנים דיוני."),
        answer=H("אתה עו\"ד ליטיגציה. נתח את עילות התביעה.")
    ),
    "פסק_דין": dict(
        summary=H("סכם פסק-דין: 1. שאלה משפטית, 2. קביעות, 3. הלכה."),
        answer=H("אתה עו\"ד. הסבר את הלכת בית-המשפט.")
    ),
    "_": dict(
        summary=H("סכם את המסמך: תקציר, נקודות עיקריות, השלכות."),
        answer=H("אתה עו\"ד.")
    )
}
tmpl = lambda l, k: PROMPTS.get(l, PROMPTS["_"])[k]

# ───────────── retrieval (RAG) ─────────────
embed = lambda t: model.encode([t], normalize_embeddings=True)[0]

async def retrieve(q, doc):
    q_emb = embed(q)
    secs = [embed(c) for c in chunk_text(doc, 400)] if doc else []
    cand = {"law": {}, "judg": {}}

    async def add(match, kind):
        meta, score = match.get("metadata", {}), match.get("score", 0)
        key = "IsraelLawID" if kind == "law" else "CaseNumber"
        _id = meta.get(key)
        if not _id:
            return
        coll = law_coll if kind == "law" else judgment_coll
        d = coll.find_one({key: _id})
        if d:
            cand[kind].setdefault(_id, {"doc": d, "scores": []})["scores"].append(score)

    async def query(idx, vec):
        vec_list = vec.tolist() if isinstance(vec, np.ndarray) else vec
        return idx.query(
            vector=vec_list,
            top_k=5,
            include_metadata=True
        ).get("matches", [])

    async def scan(vec):
        for m in await query(law_index, vec):
            await add(m, "law")
        for m in await query(judgment_index, vec):
            await add(m, "judg")

    await asyncio.gather(scan(q_emb), *(scan(e) for e in secs[:10]))

    top = lambda d: sorted(d.values(), key=lambda x: -np.mean(x["scores"]))[:3]
    return [x["doc"] for x in top(cand["law"])], [x["doc"] for x in top(cand["judg"])]

async def citations_ok(ans: str) -> bool:
    pat = r'\[\d+\]|\(\d+\)'
    lines = [l.strip() for l in ans.splitlines() if l.strip()]
    return all(re.search(pat, l) for l in lines)

# ───────────── gen – תשובה משפטית מקצועית ─────────────
async def gen(q: str) -> str:
    laws, judg = await retrieve(q, st.session_state.get("doc", ""))

    SNIPPET = 400
    def fmt_sources(lst, tag):
        if not lst:
            return f"לא נמצאו {tag} רלוונטיים."
        return "\n".join(
            f"[{i}] {d.get('Name', 'ללא שם')} – {(d.get('Description', '') or '')[:SNIPPET].strip()}"
            for i, d in enumerate(lst, 1)
        )

    sys = (
        tmpl(st.session_state.get("doctype", "_"), "answer") +
        "\n\nהנחיות ניסוח (חובה):\n"
        "• כתוב בעברית מלאה בלבד – אין להשתמש באנגלית.\n"
        "• השתמש בלשון משפטית-מקצועית, פסקאות/סעיפים ממוספרים.\n"
        "• הסתמך אך ורק על המקורות שלמטה, וציין בסוגריים את מספר-המקור ליד כל קביעה.\n"
        f"\n--- מסמך ---\n{st.session_state.get('doc', '')[:1000]}" +
        f"\n\n--- חוקים ---\n{fmt_sources(laws, 'חוקים')}" +
        f"\n\n--- פסקי דין ---\n{fmt_sources(judg, 'פסקי דין')}"
    )

    r = await client_async_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": q}
        ],
        temperature=0,
        max_tokens=900
    )
    return r.choices[0].message.content.strip()

# ───────────── chat assistant ─────────────
def chat_assistant():
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)

    # ───── cid & messages ─────
    if "cid" not in st.session_state:
        cid = ls_get("AMLChatId") or str(uuid.uuid4())
        ls_set("AMLChatId", cid)
        st.session_state.cid = cid

    if "messages" not in st.session_state:
        conv = conv_coll.find_one({"local_storage_id": st.session_state.cid})
        st.session_state["messages"] = conv.get("messages", []) if conv else []

    # ───── user name ─────
    if "user_name" not in st.session_state:
        stored = ls_get("AMLUserName")
        if stored:
            st.session_state["user_name"] = stored

    if "user_name" not in st.session_state:
        with st.form("name"):
            st.text_input("הכנס שם להתחלת שיחה:", key="user_name_input")
            sub = st.form_submit_button("התחל")
        if sub and st.session_state.get("user_name_input"):
            st.session_state["user_name"] = st.session_state["user_name_input"]
            ls_set("AMLUserName", st.session_state["user_name"])
            add_msg("assistant", f"שלום {st.session_state['user_name']}, איך אפשר לעזור?")
            conv_coll.update_one(
                {"local_storage_id": st.session_state.cid},
                {"$set": {
                    "user_name": st.session_state["user_name"],
                    "messages": st.session_state["messages"]
                }},
                upsert=True
            )
            st.rerun()
        return  # ממתין להגדרת שם

    # ───── history ─────
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    show_msgs()
    st.markdown("</div>", unsafe_allow_html=True)

    # ───── upload ─────
    up = st.file_uploader("📄 העלה מסמך", type=["pdf", "docx"])
    if up:
        raw = read_pdf(up) if up.type == "application/pdf" else read_docx(up)
        st.session_state.doctype = classify_doc(raw)
        st.session_state.doc = "\n".join(l for l in raw.splitlines() if heb.search(l))
        st.success(f"סוג המסמך: {st.session_state.doctype}")

    # ───── summary ─────
    if hasattr(st.session_state, "doc") and st.button("📋 סיכום"):
        with st.spinner("סיכום..."):
            prompt = (
                tmpl(st.session_state.doctype, "summary") +
                "\n\nכתוב 4-6 Bullet-ים קצרים (עד 30 מילים כל אחד):\n" +
                st.session_state.doc[:2000]
            )
            r = asyncio.run(
                client_async_openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=350
                )
            )
            st.session_state.summary = ensure_he(
                r.choices[0].message.content.strip().replace("•", "–")
            )

    if st.session_state.get("summary"):
        st.markdown("### סיכום:")
        st.markdown(
            f"<div dir='rtl' style='text-align:right'>{st.session_state.summary}</div>",
            unsafe_allow_html=True
        )

    # ───── answer helpers ─────
    async def handle(q):
        ans = ensure_he(await gen(q))
        if await citations_ok(ans):
            return ans
        ans2 = ensure_he(
            await gen(q + "\nחובה לציין מקור ממוספר (חוק/פס\"ד) אחרי כל משפט.")
        )
        return ans2

    # ───── ask form ─────
    with st.form("ask", clear_on_submit=True):
        q = st.text_area("הקלד שאלה משפטית:", height=100)
        send = st.form_submit_button("שלח")

    if send and q.strip():
        ans = asyncio.run(handle(q.strip()))
        add_msg("user", q.strip())
        add_msg("assistant", ans)
        conv_coll.update_one(
            {"local_storage_id": st.session_state.cid},
            {"$set": {
                "messages": st.session_state["messages"],
                "user_name": st.session_state["user_name"]
            }},
            upsert=True
        )
        st.rerun()

    # ───── clear chat ─────
    if st.button("🗑 נקה שיחה"):
    
    try:
        conv_coll.delete_one({"local_storage_id": st.session_state.cid})
    except Exception as e:
        st.error(f"שגיאה בניקוי השיחה מבסיס הנתונים: {e}")
    st_js_blocking("""
        localStorage.removeItem('AMLUserName');
        localStorage.removeItem('AMLChatId');
    """)
    st.session_state.clear()
    st.rerun()

# ───────────── legal finder assistant ─────────────
def load_document_details(kind, doc_id):
    coll = judgment_coll if kind == "Judgment" else law_coll
    key = "CaseNumber" if kind == "Judgment" else "IsraelLawID"
    return coll.find_one({key: doc_id})

def get_explanation(scenario, doc, kind):
    name, desc = doc.get("Name", ""), doc.get("Description", "")
    if kind == "Judgment":
        prom = f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי פסק הדין הבא:
שם: {name}
תיאור: {desc}

הסבר בקצרה מדוע פסק דין זה מסייע ודרג 0-10.
החזר JSON כמו:
{{"advice":"הסבר","score":7}}"""
    else:
        prom = f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי החוק הבא:
שם: {name}
תיאור: {desc}

הסבר בקצרה מדוע החוק רלוונטי ודרג 0-10.
החזר JSON כמו:
{{"advice":"הסבר","score":6}}"""
    try:
        r = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prom}],
            temperature=0.7
        )
        return json.loads(r.choices[0].message.content.strip())
    except Exception:
        return {"advice": "שגיאה", "score": "N/A"}

def legal_finder_assistant():
    st.title("Legal Finder Assistant")
    kind = st.selectbox("Choose what to search", ["Judgment", "Law"])
    scen = st.text_area("Describe your scenario")
    if st.button("Find Suitable Results") and scen:
        q_emb = model.encode([scen], normalize_embeddings=True)[0]
        idx = judgment_index if kind == "Judgment" else law_index
        key = "CaseNumber" if kind == "Judgment" else "IsraelLawID"
        matches = idx.query(
            vector=q_emb.tolist(),
            top_k=5,
            include_metadata=True
        ).get("matches", [])
        if not matches:
            st.info("No matches found.")
            return
        for m in matches:
            _id = m.get("metadata", {}).get(key)
            doc = load_document_details(kind, _id)
            if not doc:
                continue
            name, desc = doc.get("Name", ""), doc.get("Description", "")
            date_lbl = "DecisionDate" if kind == "Judgment" else "PublicationDate"
            extra = (
                f"<div class='law-meta'>Procedure Type: {doc.get('ProcedureType','N/A')}</div>"
                if kind == "Judgment" else ""
            )
            st.markdown(
                f"<div class='law-card'><div class='law-title'>{name} (ID:{_id})</div>"
                f"<div class='law-description'>{desc}</div>"
                f"<div class='law-meta'>{date_lbl}: {doc.get(date_lbl,'N/A')}</div>{extra}</div>",
                unsafe_allow_html=True
            )
            with st.spinner("GPT explanation..."):
                res = get_explanation(scen, doc, kind)
            st.markdown(
                f"<span style='color:red;'>עצת האתר: {res.get('advice','')}</span>",
                unsafe_allow_html=True
            )
            with st.expander(f"View Full Details for {_id}"):
                st.json(doc)

# ─────────────────── main ───────────────────
if app_mode == "Chat Assistant":
    chat_assistant()
else:
    legal_finder_assistant()
