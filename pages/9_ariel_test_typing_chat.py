
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
def ls_get(key: str):
    
    with st.container():
        st.markdown("<div style='display:none'>", unsafe_allow_html=True)
        val = st_js_blocking(
            f"return localStorage.getItem('{key}');",
            key="ls_"+key
        )
        st.markdown("</div>", unsafe_allow_html=True)
    return val

def ls_set(key: str, val: str):
    st_js(f"localStorage.setItem('{key}', '{val}');")


heb = re.compile(r'[א-ת]')
SALUT = r'\b(לכבוד|מר|מר\.?|גב\'?|גברת|ד"ר|ד"ר\.?)\b'
def rtl_norm(t:str)->str:
    if not heb.search(t): return t
    try: t = get_display(t)
    except: t = " ".join(w[::-1] if heb.search(w) else w for w in t.split())
    return re.sub(r'\s{2,}', ' ', re.sub(SALUT, '', t)).strip()

read_pdf  = lambda f: "".join(rtl_norm(p.get_text()) for p in fitz.open(stream=f.read(), filetype="pdf"))
read_docx = lambda f: "\n".join(rtl_norm(p.text) for p in docx.Document(f).paragraphs if p.text.strip())

def add_msg(role, txt):
    st.session_state.setdefault("messages", []).append(
        {"role":role, "content":txt, "timestamp":datetime.now().strftime("%H:%M:%S")})

def show_msgs():
    for m in st.session_state.get("messages", []):
        css="user-message" if m["role"]=="user" else "bot-message"
        st.markdown(f"<div class='{css}'>{m['content']}<div class='timestamp'>{m['timestamp']}</div></div>", unsafe_allow_html=True)

def chunk_text(t,L=450):
    sent=re.split(r'(?:\.|\?|!)\s+',t); out,cur=[], ""
    for s in sent:
        if len(cur)+len(s)>L and cur: out.append(cur.strip()); cur=s
        else: cur+=" "+s
    if cur.strip(): out.append(cur.strip())
    return out[:20]

contains_en=lambda t: bool(re.search(r'[A-Za-z]',t))
def ensure_he(t):
    if sum(contains_en(w) for w in t.split())<=3: return t
    r=client_sync_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":"תרגם לעברית מלאה:\n"+t}],
        temperature=0,max_tokens=len(t)//2)
    return r.choices[0].message.content.strip()

# ───────────── classification ─────────────
DOC_LABELS = {
    "חוזה_עבודה":      "הסכם בין מעסיק לעובד או מועמד לעבודה",
    "חוזה_שכירות":     "הסכם להשכרת דירה, משרד או נכס אחר",
    "חוזה_שירות":      "הסכם בין מזמין לספק שירות",
    "מכתב_פיטורין":    "הודעה על סיום העסקה או הפסקת עבודה",
    "מכתב_התראה":      "מכתב דרישה או אזהרה לפני נקיטת הליכים",
    "תקנון":           "מסמך כללי חובות וזכויות (לדוגמה: תקנון חברה, אתר, עמותה)",
    "NDA":             "הסכם סודיות ואי-גילוי",
    "כתב_תביעה":       "מסמך פתיחת הליך בבית-משפט",
    "כתב_הגנה":        "תגובה לכתב תביעה",
    "פסק_דין":         "הכרעת בית-משפט",
    "מסמך_אחר":        "כל מסמך משפטי אחר שאינו נכנס לאף קטגוריה"
}

CLS_SYS = (
    "אתה מסווג מסמכים משפטיים בעברית. "
    "קרא את הטקסט המצורף והחזר *רק* את שם התווית המתאימה מבין:\n"
    + ", ".join(DOC_LABELS.keys()) +
    "\nאין לכתוב שום טקסט נוסף."
)

def classify_doc(txt: str) -> str:
    sample = txt[:1500]   # קטע מייצג, מקטין עלות
    try:
        resp = client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": CLS_SYS},
                {"role": "user",   "content": sample}
            ],
            temperature=0,
            max_tokens=5
        )
        label = resp.choices[0].message.content.strip()
        return label if label in DOC_LABELS else "מסמך_אחר"
    except Exception:
        return "מסמך_אחר"



# ────────────── prompts ──────────────
H=lambda s: s+"  **ענה בעברית מלאה וללא מילים באנגלית. ציין מקור ממוספר (חוק או פס״ד) אחרי כל קביעה.**"
PROMPTS={
 "מכתב_פיטורין":dict(summary=H("סכם מכתב פיטורין: 1. פרטי עובד ותאריכים, 2. זכויות ותשלומים, 3. צעדים מומלצים."),
                      answer =H("אתה עו\"ד דיני-עבודה. השב רק על סמך המכתב וחוקי עבודה רלוונטיים.")),
 "חוזה_עבודה":  dict(summary=H("סכם חוזה עבודה: 1. תנאי העסקה, 2. סעיפי סודיות ואי-תחרות, 3. סיכונים והמלצות."),
                      answer =H("אתה עו\"ד דיני-עבודה. נתח את סעיפי החוזה.")),
 "תקנון":        dict(summary=H("סכם תקנון/מדיניות: 1. מטרות, 2. זכויות/חובות, 3. סיכונים לאי-ציות."),
                      answer =H("אתה עו\"ד חברות. הסבר תוקף סעיפי התקנון.")),
 "כתב_תביעה":    dict(summary=H("סכם כתב תביעה: 1. עילות, 2. סעדים, 3. לוח זמנים דיוני."),
                      answer =H("אתה עו\"ד ליטיגציה. נתח את עילות התביעה.")),
 "פסק_דין":      dict(summary=H("סכם פסק-דין: 1. שאלה משפטית, 2. קביעות, 3. הלכה."),
                      answer =H("אתה עו\"ד. הסבר את הלכת בית-המשפט.")),
 "_":             dict(summary=H("סכם את המסמך: תקציר, נקודות עיקריות, השלכות."),
                      answer =H("אתה עו\"ד."))
}
tmpl=lambda l,k: PROMPTS.get(l,PROMPTS["_"])[k]

# ───────────── retrieval (RAG) ─────────────
embed=lambda t: model.encode([t],normalize_embeddings=True)[0]

async def retrieve(q,doc):
    q_emb=embed(q); secs=[embed(c) for c in chunk_text(doc)] if doc else []
    cand={"law":{}, "judg":{}}
    async def add(m,kind):
        meta,score=m.get("metadata",{}), m.get("score",0)
        key="IsraelLawID" if kind=="law" else "CaseNumber"; _id=meta.get(key)
        if not _id: return
        coll=law_coll if kind=="law" else judgment_coll
        d=coll.find_one({key:_id}); 
        if d: cand[kind].setdefault(_id,{"doc":d,"scores":[]})["scores"].append(score)
    async def scan(e):
        rl,rj=await asyncio.gather(
            asyncio.to_thread(law_index.query,vector=e.tolist(),top_k=1,include_metadata=True),
            asyncio.to_thread(judgment_index.query,vector=e.tolist(),top_k=1,include_metadata=True))
        [await add(m,"law") for m in rl.get("matches",[])]
        [await add(m,"judg")for m in rj.get("matches",[])]
    await asyncio.gather(*(scan(e) for e in secs))
    for m in law_index.query(vector=q_emb.tolist(),top_k=3,include_metadata=True).get("matches",[]):  await add(m,"law")
    for m in judgment_index.query(vector=q_emb.tolist(),top_k=3,include_metadata=True).get("matches",[]): await add(m,"judg")
    top=lambda d: sorted(d.values(), key=lambda x:-np.mean(x["scores"]))[:3]
    return [x["doc"] for x in top(cand["law"])], [x["doc"] for x in top(cand["judg"])]

async def citations_ok(ans:str)->bool:
    if contains_en(ans): return False
    try:
        r=await client_async_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":"Does every claim have an explicit citation? Answer Yes/No.\n"+ans}],
            temperature=0,max_tokens=3)
        return "yes" in r.choices[0].message.content.lower()
    except: return True

# ───────────── chat assistant ─────────────
def chat_assistant():
    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)
    # cid & messages
    if "cid" not in st.session_state:
        cid=ls_get("AMLChatId") or str(uuid.uuid4()); ls_set("AMLChatId",cid); st.session_state.cid=cid
    if "messages" not in st.session_state:
        conv=conv_coll.find_one({"local_storage_id":st.session_state.cid})
        st.session_state["messages"]=conv.get("messages",[]) if conv else []

    # name persistence
    if "user_name" not in st.session_state:
        stored=ls_get("AMLUserName")
        if stored: st.session_state["user_name"]=stored
    if "user_name" not in st.session_state:
        with st.form("name"): st.text_input("הכנס שם להתחלת שיחה:", key="user_name_input"); sub=st.form_submit_button("התחל")
        if sub and st.session_state.get("user_name_input"):
            st.session_state["user_name"]=st.session_state["user_name_input"]; ls_set("AMLUserName",st.session_state["user_name"])
            add_msg("assistant",f"שלום {st.session_state['user_name']}, איך אפשר לעזור?")
            conv_coll.update_one({"local_storage_id":st.session_state.cid},
                                 {"$set":{"user_name":st.session_state["user_name"],
                                          "messages":st.session_state["messages"]}}, upsert=True)
            st.rerun()
        return

    # history
    st.markdown('<div class="chat-container">',unsafe_allow_html=True); show_msgs(); st.markdown("</div>",unsafe_allow_html=True)

    # upload
    up=st.file_uploader("📄 העלה מסמך",type=["pdf","docx"])
    if up:
        raw=read_pdf(up) if up.type=="application/pdf" else read_docx(up)
        st.session_state.doctype=classify_doc(raw)
        st.session_state.doc="\n".join(l for l in raw.splitlines() if heb.search(l))
        st.success(f"סוג המסמך: {st.session_state.doctype}")

    # summary
    if hasattr(st.session_state,"doc") and st.button("📋 סיכום"):
        with st.spinner("סיכום..."):
            prompt=tmpl(st.session_state.doctype,"summary")+"\n"+st.session_state.doc
            r=asyncio.run(client_async_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0,max_tokens=700))
            st.session_state.summary=ensure_he(r.choices[0].message.content.strip())
    if st.session_state.get("summary"):
        st.markdown("### סיכום:"); st.markdown(f"<div dir='rtl' style='text-align:right'>{st.session_state.summary}</div>",unsafe_allow_html=True)

    # answer helpers
    async def gen(q):
        laws,judg=await retrieve(q,st.session_state.get("doc",""))
        law_txt="\n\n".join(d.get("Description","")[:800] for d in laws) or "לא נמצאו חוקים רלוונטיים."
        jud_txt="\n\n".join(d.get("Description","")[:800] for d in judg) or "לא נמצאו פסקי דין רלוונטיים."
        sys=tmpl(st.session_state.get("doctype","_"),"answer")+\
            f"\n\n--- מסמך ---\n{st.session_state.get('doc','')[:1500]}" +\
            f"\n\n--- חוקים ---\n{law_txt}" +\
            f"\n\n--- פסקי דין ---\n{jud_txt}"
        r=await client_async_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},{"role":"user","content":q}],
            temperature=0,max_tokens=700)
        return r.choices[0].message.content.strip()

    async def handle(q):
        ans=ensure_he(await gen(q))
        if await citations_ok(ans): return ans
        ans2=ensure_he(await gen(q+"\nחובה לציין מקור ממוספר (חוק/פס\"ד) אחרי כל משפט."))
        return ans2

    with st.form("ask",clear_on_submit=True):
        q=st.text_area("הקלד שאלה משפטית:",height=100); send=st.form_submit_button("שלח")
    if send and q.strip():
        ans=asyncio.run(handle(q.strip())); add_msg("user",q.strip()); add_msg("assistant",ans)
        conv_coll.update_one({"local_storage_id":st.session_state.cid},
                             {"$set":{"messages":st.session_state["messages"],
                                      "user_name":st.session_state["user_name"]}}, upsert=True)
        st.rerun()

    if st.button("🗑 נקה"):
        conv_coll.delete_one({"local_storage_id":st.session_state.cid})
        st_js("localStorage.clear()"); st.session_state.clear(); st.rerun()

# ───────────── legal finder assistant ─────────────
def load_document_details(kind, doc_id):
    coll = judgment_coll if kind=="Judgment" else law_coll
    key  = "CaseNumber" if kind=="Judgment" else "IsraelLawID"
    return coll.find_one({key:doc_id})

def get_explanation(scenario, doc, kind):
    name, desc = doc.get("Name",""), doc.get("Description","")
    if kind=="Judgment":
        prom=f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי פסק הדין הבא:
שם: {name}
תיאור: {desc}

הסבר בקצרה מדוע פסק דין זה מסייע ודרג 0-10.
החזר JSON כמו:
{{"advice":"הסבר","score":7}}"""
    else:
        prom=f"""בהתבסס על הסצנריו הבא:
{scenario}

וכן על פרטי החוק הבא:
שם: {name}
תיאור: {desc}

הסבר בקצרה מדוע החוק רלוונטי ודרג 0-10.
החזר JSON כמו:
{{"advice":"הסבר","score":6}}"""
    try:
        r=client_sync_openai.chat.completions.create(
            model="gpt-3.5-turbo",messages=[{"role":"user","content":prom}],temperature=0.7)
        return json.loads(r.choices[0].message.content.strip())
    except: return {"advice":"שגיאה","score":"N/A"}

def legal_finder_assistant():
    st.title("Legal Finder Assistant")
    kind=st.selectbox("Choose what to search",["Judgment","Law"])
    scen=st.text_area("Describe your scenario")
    if st.button("Find Suitable Results") and scen:
        q_emb=model.encode([scen],normalize_embeddings=True)[0]
        idx=judgment_index if kind=="Judgment" else law_index
        key="CaseNumber" if kind=="Judgment" else "IsraelLawID"
        matches=idx.query(vector=q_emb.tolist(),top_k=5,include_metadata=True).get("matches",[])
        if not matches: st.info("No matches found."); return
        for m in matches:
            _id=m.get("metadata",{}).get(key); doc=load_document_details(kind,_id); 
            if not doc: continue
            name,desc=doc.get("Name",""),doc.get("Description","")
            date_lbl="DecisionDate" if kind=="Judgment" else "PublicationDate"
            extra=f"<div class='law-meta'>Procedure Type: {doc.get('ProcedureType','N/A')}</div>" if kind=="Judgment" else ""
            st.markdown(f"<div class='law-card'><div class='law-title'>{name} (ID:{_id})</div>"
                        f"<div class='law-description'>{desc}</div>"
                        f"<div class='law-meta'>{date_lbl}: {doc.get(date_lbl,'N/A')}</div>{extra}</div>",unsafe_allow_html=True)
            with st.spinner("GPT explanation..."): res=get_explanation(scen,doc,kind)
            st.markdown(f"<span style='color:red;'>עצת האתר: {res.get('advice','')}</span>",unsafe_allow_html=True)
            with st.expander(f"View Full Details for {_id}"): st.json(doc)

# ─────────────────── main ───────────────────
if app_mode=="Chat Assistant":
    chat_assistant()
else:
    legal_finder_assistant()
