import os, sys, json, uuid, asyncio, re
from datetime import datetime

import streamlit as st
import torch, fitz, docx, numpy as np
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv
from streamlit_js import st_js, st_js_blocking

from app_resources import mongo_client, pinecone_client, model

# ─────────────────────── ENV & GLOBALS ──────────────────────
load_dotenv()
DATABASE_NAME  = os.getenv("DATABASE_NAME")
OPENAI_API_KEY = os.getenv("OPEN_AI")

client_async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
client_sync_openai  = OpenAI    (api_key=OPENAI_API_KEY)

torch.classes.__path__ = []      # Streamlit-Torch bug-workaround
os.environ["TOKENIZERS_PARALLELISM"] = "false"

judgment_index = pinecone_client.Index("judgments-names")
law_index      = pinecone_client.Index("laws-names")

db = mongo_client[DATABASE_NAME]
judgment_collection = db["judgments"]
law_collection      = db["laws"]
conversation_coll   = db["conversations"]

# ─────────────────────────── UI CSS ─────────────────────────
st.set_page_config(page_title="Ask Mini Lawyer", page_icon="⚖️", layout="wide")
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
.stButton>button{background:#7ce38b;color:#fff;font-size:14px;border:none;padding:8px 16px;
                 border-radius:5px;cursor:pointer}
.stButton>button:hover{background:#69d67a}
</style>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox("Choose module", ["Chat Assistant", "Legal Finder"])

# ──────────────────── GENERAL HELPERS ───────────────────────
ls_get = lambda k: st_js_blocking(f"return localStorage.getItem('{k}');", key="ls_"+k)
ls_set = lambda k,v: st_js(f"localStorage.setItem('{k}', '{v}');")

read_pdf  = lambda f: "".join(p.get_text() for p in fitz.open(stream=f.read(), filetype="pdf"))
read_docx = lambda f: "\n".join(p.text for p in docx.Document(f).paragraphs)

def add_msg(role, txt):
    st.session_state.setdefault("messages", []).append(
        {"role": role, "content": txt, "timestamp": datetime.now().strftime("%H:%M:%S")}
    )

def show_msgs():
    for m in st.session_state.get("messages", []):
        css = "user-message" if m["role"] == "user" else "bot-message"
        st.markdown(f"<div class='{css}'>{m['content']}<div class='timestamp'>{m['timestamp']}</div></div>",
                    unsafe_allow_html=True)

def chunk_text(txt, L=450):
    sent = re.split(r'(?:\.|\?|!)\s+', txt)
    out, cur = [], ""
    for s in sent:
        if len(cur)+len(s) > L and cur:
            out.append(cur.strip()); cur = s
        else:
            cur += " " + s
    if cur.strip(): out.append(cur.strip())
    return out[:20]

# ───────────── Hebrew-only enforcement ──────────────────────
contains_english = lambda t: bool(re.search(r"[A-Za-z]", t))
def ensure_hebrew(t):
    if not contains_english(t): return t
    prompt = "תרגם את הטקסט הבא לעברית מלאה וללא מילים באנגלית:\n" + t
    r = client_sync_openai.chat.completions.create(
        model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}],
        temperature=0, max_tokens=len(t)//2)
    return r.choices[0].message.content.strip()

# ───────────────────── CLASSIFICATION ───────────────────────
CATEGORIES = {
 "מכתב_פיטורין": {"regex":[r"פיטור(ין|ים)|termination notice"]},
 "חוזה_עבודה":   {"regex":[r"הסכם\s+עבודה|employment agreement"]},
 "NDA":          {"regex":[r"סודיות|confidentiality"]},
 "CEASE_DESIST": {"regex":[r"חדל|להפסיק|cease and desist"]},
 "תקנון":        {"regex":[r"תקנון|by.?law|policy"]},
 "כתב_תביעה":    {"regex":[r"כתב\s+תביעה|התובע|הנתבע"]},
 "פסק_דין":      {"regex":[r"פסק[-\s]?דין|בית.?משפט"]},
 "מכתב_אחר":     {"regex":[]}
}
CLS_SYSTEM = "אתה מסווג מסמכים משפטיים. החזר תווית אחת בלבד: " + ", ".join(CATEGORIES.keys())
def classify_doc(txt:str)->str:
    sample = txt[:800] + "\n---\n" + txt[-800:]
    try:
        resp = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"system","content":CLS_SYSTEM+"\n"+sample}],
            temperature=0, max_tokens=5)
        label = resp.choices[0].message.content.strip()
    except Exception:
        label = "מכתב_אחר"
    for lab,d in CATEGORIES.items():
        if any(re.search(p,txt,re.I) for p in d["regex"]): label = lab
    return label if label in CATEGORIES else "מכתב_אחר"

# ────────────────────── TEMPLATES ───────────────────────────
H = lambda s: s + "  **ענה בעברית מלאה וללא מילים באנגלית.**"
PROMPTS = {
 "מכתב_פיטורין":dict(summary=H("סכם מכתב פיטורין: 1. פרטי עובד ותאריכים, 2. זכויות ותשלומים, 3. צעדים מומלצים."),
                      answer =H("אתה עו\"ד דיני-עבודה. השב רק על סמך המכתב וחוקי עבודה רלוונטיים; ציין מקור לכל טענה.")),
 "חוזה_עבודה":  dict(summary=H("סכם חוזה עבודה: 1. תנאי העסקה, 2. סעיפי סודיות ואי-תחרות, 3. סיכונים והמלצות."),
                      answer =H("אתה עו\"ד דיני-עבודה. נתח את סעיפי החוזה וציין חוקים/פס״דים תומכים.")),
 "NDA":          dict(summary=H("סכם NDA: 1. הגדרות מידע חסוי, 2. תקופת חיסיון, 3. אמצעי אכיפה."),
                      answer =H("אתה עו\"ד קניין-רוחני. פרט השלכות משפטיות של הפרת ה-NDA.")),
 "CEASE_DESIST": dict(summary=H("סכם מכתב אזהרה: 1. טענות, 2. דרישות, 3. לוחות זמנים לאכיפה."),
                      answer =H("אתה עו\"ד ליטיגציה. הערך חוקיות וסיכון של האיומים במכתב.")),
 "תקנון":        dict(summary=H("סכם תקנון/מדיניות: 1. מטרות, 2. זכויות/חובות, 3. סיכונים לאי-ציות."),
                      answer =H("אתה עו\"ד חברות. הסבר תוקף סעיפי התקנון.")),
 "כתב_תביעה":    dict(summary=H("סכם כתב תביעה: 1. עילות, 2. סעדים, 3. לוח זמנים דיוני."),
                      answer =H("אתה עו\"ד. מנתח כתב התביעה בהתאם לתקנות סדר הדין האזרחי.")),
 "פסק_דין":      dict(summary=H("סכם פסק-דין: 1. שאלה משפטית, 2. קביעות, 3. הלכה."),
                      answer =H("אתה עו\"ד. הסבר את הלכת בית-המשפט ותוקפה.")),
 "_":            dict(summary=H("סכם את המסמך: תקציר, נקודות עיקריות, השלכות."),
                      answer =H("אתה עו\"ד. השב על סמך המסמך וחוקים רלוונטיים."))
}
tmpl = lambda lbl, kind: PROMPTS.get(lbl, PROMPTS["_"])[kind]

# ───────────────────── RETRIEVAL (RAG) ──────────────────────
def embed(t): return model.encode([t], normalize_embeddings=True)[0]

async def retrieve(query, doc):
    q_emb = embed(query)
    secs  = [embed(c) for c in chunk_text(doc)] if doc else []
    cand  = {"law":{}, "judg":{}}

    async def add(match, kind):
        meta, score = match.get("metadata",{}), match.get("score",0)
        key   = "IsraelLawID" if kind=="law" else "CaseNumber"
        _id   = meta.get(key); coll = law_collection if kind=="law" else judgment_collection
        if not _id: return
        d = coll.find_one({key:_id}); 
        if not d: return
        cand[kind].setdefault(_id, {"doc":d,"scores":[]})["scores"].append(score)

    # scan each section
    async def scan(e):
        rl, rj = await asyncio.gather(
            asyncio.to_thread(law_index.query,      vector=e.tolist(), top_k=1, include_metadata=True),
            asyncio.to_thread(judgment_index.query, vector=e.tolist(), top_k=1, include_metadata=True))
        [await add(m,"law")  for m in rl.get("matches",[])]
        [await add(m,"judg") for m in rj.get("matches",[])]

    await asyncio.gather(*(scan(e) for e in secs))
    # add global matches
    for m in law_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True).get("matches",[]):   await add(m,"law")
    for m in judgment_index.query(vector=q_emb.tolist(), top_k=3, include_metadata=True).get("matches",[]): await add(m,"judg")

    top = lambda d: sorted(d.values(), key=lambda x:-np.mean(x["scores"]))[:3]
    return [x["doc"] for x in top(cand["law"])], [x["doc"] for x in top(cand["judg"])]

# ──────────────── SELF-CHECK  (citations + he) ──────────────
async def citations_ok(ans:str)->bool:
    if contains_english(ans): return False
    probe = "Does every claim have an explicit citation? Answer Yes/No."
    try:
        r = await client_async_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content": probe + "\n" + ans}],
            temperature=0, max_tokens=3)
        return "yes" in r.choices[0].message.content.lower()
    except: return True   # fail-open

# ─────────────────── CHAT ASSISTANT ─────────────────────────
def chat_assistant():
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)

    # session state bootstrap
    if "cid" not in st.session_state:
        cid = ls_get("AMLChatId") or str(uuid.uuid4())
        ls_set("AMLChatId", cid)
        st.session_state.cid = cid
    if "messages" not in st.session_state:
        conv = conversation_coll.find_one({"local_storage_id": st.session_state.cid})
        st.session_state["messages"] = conv.get("messages", []) if conv else []
    st.session_state.setdefault("name", None)

    # first-run name prompt
    if not st.session_state.get("name"):
        with st.form("name_form"):
            n = st.text_input("הכנס שם להתחלת שיחה:", key="user_name_field")
            submitted = st.form_submit_button("התחל")
        if submitted and n:
            st.session_state["name"] = n
            add_msg("assistant", f"שלום {n}, איך אפשר לעזור?")
            conversation_coll.update_one(
                {"local_storage_id": st.session_state.cid},
                {"$set": {"user_name": n,
                          "messages": st.session_state["messages"]}},
                upsert=True)
            st.rerun()
        return   

    # show chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    show_msgs()
    st.markdown("</div>", unsafe_allow_html=True)

    # file upload
    up = st.file_uploader("📄 העלה מסמך", type=["pdf","docx"])
    if up:
        raw = read_pdf(up) if up.type=="application/pdf" else read_docx(up)
        st.session_state.doc     = "\n".join(l for l in raw.splitlines() if re.search(r"[א-ת]", l))
        st.session_state.doctype = classify_doc(st.session_state.doc)
        st.success(f"סוג המסמך: {st.session_state.doctype}")

    # summary button
    if hasattr(st.session_state, "doc") and st.button("📋 סיכום"):
        with st.spinner("סיכום..."):
            prompt = tmpl(st.session_state.doctype,"summary") + "\n" + st.session_state.doc
            r = asyncio.run(client_async_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=700))
            st.session_state.summary = ensure_hebrew(r.choices[0].message.content.strip())
    if st.session_state.get("summary"):
        st.markdown("### סיכום:"); st.info(st.session_state.summary)

    # async handler
    async def handle(q):
        laws, judg = await retrieve(q, st.session_state.get("doc",""))
        doc_sn = st.session_state.get("doc","")[:1500]
        law_txt = "\n\n".join(d.get("Description","")[:800] for d in laws)
        jud_txt = "\n\n".join(d.get("Description","")[:800] for d in judg)

        sys = tmpl(st.session_state.get("doctype","_"),"answer") + \
              "\n\n--- מסמך ---\n" + doc_sn + \
              "\n\n--- חוקים ---\n" + law_txt + \
              "\n\n--- פסקי דין ---\n" + jud_txt
        r = await client_async_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":q}],
            temperature=0, max_tokens=700)
        ans = ensure_hebrew(r.choices[0].message.content.strip())
        if not await citations_ok(ans):
            ans = "⚠️ התשובה לא עמדה בקריטריונים (עברית מלאה + אסמכתאות). נסה לשאול בצורה מדויקת יותר."
        return ans

    # ask form
    with st.form("ask", clear_on_submit=True):
        q = st.text_area("הקלד שאלה משפטית:", height=100)
        send = st.form_submit_button("שלח")
    if send and q.strip():
        ans = asyncio.run(handle(q.strip()))
        add_msg("user", q.strip()); add_msg("assistant", ans)
        conversation_coll.update_one(
            {"local_storage_id": st.session_state.cid},
            {"$set":{"messages": st.session_state["messages"],
                     "user_name": st.session_state["name"]}},
            upsert=True)
        st.rerun()

    # clear chat
    if st.button("🗑 נקה"):
        conversation_coll.delete_one({"local_storage_id": st.session_state.cid})
        st_js("localStorage.clear();")
        st.session_state.clear()
        st.rerun()

# ───────────────────── LEGAL FINDER ─────────────────────────
def legal_finder():
    st.title("Legal Finder")
    kind = st.selectbox("Search for", ["Judgment","Law"])
    scen = st.text_area("Scenario")
    if st.button("Find") and scen:
        emb  = embed(scen)
        idx  = judgment_index if kind=="Judgment" else law_index
        key  = "CaseNumber" if kind=="Judgment" else "IsraelLawID"
        res  = idx.query(vector=emb.tolist(), top_k=5, include_metadata=True)
        for m in res.get("matches", []):
            _id = m.get("metadata", {}).get(key); 
            coll= judgment_collection if kind=="Judgment" else law_collection
            doc = coll.find_one({key:_id}); 
            if not doc: continue
            st.markdown(
                f"<div class='law-card'><div class='law-title'>{doc.get('Name','')} (ID:{_id})</div>"
                f"<div class='law-description'>{doc.get('Description','')[:600]}...</div></div>",
                unsafe_allow_html=True)

# ───────────────────────── MAIN ─────────────────────────────
if app_mode == "Chat Assistant":
    chat_assistant()
else:
    legal_finder()
