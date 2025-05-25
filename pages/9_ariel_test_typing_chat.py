
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

DATABASE_NAME = os.getenv("DATABASE_NAME")



if not OPENAI_API_KEY:

    st.error("OPEN_AI API key not found in environment variables. Please set it.")

    st.stop() # Stop execution if key is missing



client_async_openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

client_sync_openai = OpenAI(api_key=OPENAI_API_KEY)



try:

    judgment_index = pinecone_client.Index("judgments-names")

    law_index = pinecone_client.Index("laws-names")

except Exception as e:

    st.warning(f"Could not connect to Pinecone indexes: {e}. RAG functionality might be limited.")

    judgment_index = None # Set to None if connection fails

    law_index = None # Set to None if connection fails



db = mongo_client[DATABASE_NAME]

judgment_coll = db["judgments"]

law_coll = db["laws"]

conv_coll = db["conversations"]



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

def ls_get(key: str) -> str | None:

    with st.container():

        st.markdown("<div style='display:none'>", unsafe_allow_html=True)

        value = st_js_blocking(f"return localStorage.getItem('{key}');", key=f"ls_{key}")

        st.markdown("</div>", unsafe_allow_html=True)

    return value



def ls_set(key: str, value: str) -> None:

    st_js(f"localStorage.setItem('{key}', '{value}');")



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



read_pdf = lambda f: "".join(rtl_norm(p.get_text()) for p in fitz.open(stream=f.read(), filetype="pdf"))

read_docx = lambda f: "\n".join(rtl_norm(p.text) for p in docx.Document(f).paragraphs if p.text.strip())



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

    english_word_count = sum(1 for word in text.split() if contains_en(word))

    total_words = len(text.split())

    if total_words > 0 and (english_word_count / total_words) < 0.2:

        return text

    try:

        response = client_sync_openai.chat.completions.create(

            model="gpt-3.5-turbo",

            messages=[{"role": "user", "content": "תרגם לעברית מלאה:\n" + text}],

            temperature=0,

            max_tokens=len(text) * 2

        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        st.warning(f"Translation failed: {e}. Returning original text.")

        return text



# ───────────── classification ─────────────

DOC_LABELS = {

    "חוזה_עבודה": "הסכם בין מעסיק לעובד או מועמד לעבודה",

    "חוזה_שכירות": "הסכם להשכרת דירה, משרד או נכס אחר",

    "חוזה_שירות": "הסכם בין מזמין לספק שירות",

    "מכתב_פיטורין": "הודעה על סיום העסקה או הפסקת עבודה",

    "מכתב_התראה": "מכתב דרישה או אזהרה לפני נקיטת הליכים",

    "תקנון": "מסמך כללי חובות וזכויות",

    "NDA": "הסכם סודיות ואי-גילוי",

    "כתב_תביעה": "מסמך פתיחת הליך בבית-משפט",

    "כתב_הגנה": "תגובה לכתב תביעה",

    "פסק_דין": "הכרעת בית-משפט",

    "מסמך_אחר": "כל מסמך משפטי אחר"

}



CLS_EXAMPLES = [

    ("הננו להודיעך בזאת כי העסקתך תסתיים בתאריך …", "מכתב_פיטורין"),

    ("העובד מתחייב לשמור בסוד כל מידע", "חוזה_עבודה"),

    ("השוכר מתחייב להחזיר את הנכס כשהוא נקי", "חוזה_שכירות"),

    ("הצדדים מסכימים שלא לגלות מידע סודי", "NDA"),

    ("הנתבע ביצע רשלנות … לפיכך מתבקש ביהמ״ש", "כתב_תביעה"),

    ("המשכיר משכיר בזה לשוכר והשוכר שוכר בזה מהמשכיר את הדירה המצויה ברחוב", "חוזה_שכירות"),

    ("תקופת השכירות תחל ביום 1.1.2024 ותסתיים ביום 31.12.2024. דמי השכירות יעמדו על סך", "חוזה_שכירות"),

    ("העובד מתחייב לבצע את תפקידו במסירות ובהתאם להוראות המעסיק", "חוזה_עבודה"),

    ("החברה תספק שירותי ייעוץ ותמיכה טכנית ללקוח במשך 12 חודשים", "חוזה_שירות"),

    ("אנו דורשים את תשלום החוב בסך 10,000 ש\"ח תוך 7 ימים ממועד מכתב זה.", "מכתב_התראה"),

    ("החלטת בית המשפט קובעת כי על הנתבע לשלם פיצויים לתובע", "פסק_דין"),

    ("תקנון זה מגדיר את כללי ההתנהלות והזכויות של חברי העמותה", "תקנון"),

    ("הסכמת הצדדים לביצוע העבודה על ידי הקבלן בהתאם למפרט טכני", "חוזה_שירות"),

    ("לפי החלטת בית הדין האזורי לעבודה בחיפה מיום 1.1.2023", "פסק_דין")

]



CLS_SYS = (

    "אתה מסווג מסמכים משפטיים בעברית באופן מדויק ואמין. "

    "החזר JSON במבנה:\n"

    '{"label":"<LABEL>","confidence":0-100}\n'

    f"עליך לבחור *רק* מתוך התוויות הבאות: {', '.join(DOC_LABELS.keys())}. "

    "חשוב מאוד: אם אינך בטוח בסיווג המסמך, בחר 'מסמך_אחר'. "

    "הימנע מניחושים. ההחלטה חייבת להיות מבוססת על תוכן המסמך בלבד."

)



def classify_doc(txt: str) -> str:

    if not isinstance(txt, str):

        try:

            txt = txt.decode("utf-8")

        except Exception:

            txt = str(txt)



    chunks_for_cls = []

    text_len = len(txt)

    if text_len > 1500:

        chunks_for_cls.append(txt[:500])

        chunks_for_cls.append(txt[text_len//4 - 250 : text_len//4 + 250])

        chunks_for_cls.append(txt[text_len//2 - 250 : text_len//2 + 250])

        chunks_for_cls.append(txt[text_len*3//4 - 250 : text_len*3//4 + 250])

        chunks_for_cls.append(txt[-500:])

    elif text_len > 500:

        chunks_for_cls.append(txt[:500])

        chunks_for_cls.append(txt[-500:])

    else:

        chunks_for_cls.append(txt)



    msgs = [{"role": "system", "content": CLS_SYS}]



    for eg, lbl in CLS_EXAMPLES:

        msgs += [

            {"role": "user", "content": eg},

            {"role": "assistant", "content": json.dumps({"label": lbl, "confidence": 95})}

        ]



    for c in chunks_for_cls:

        msgs.append({"role": "user", "content": c})



    try:

        r = client_sync_openai.chat.completions.create(

            model="gpt-4o-mini",

            messages=msgs,

            temperature=0,

            max_tokens=30

        )

        response_content = r.choices[0].message.content

        j = json.loads(response_content)

        if j.get("label") in DOC_LABELS:

            return j["label"]

        else:

            return "מסמך_אחר"

    except json.JSONDecodeError:

        return "מסמך_אחר"

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

    "חוזה_שכירות": dict(

        summary=H(

            "סכם את חוזה השכירות לנקודות קריטיות: "

            "1. מיהם הצדדים (משכיר, שוכר, ערבים) ותאריכים עיקריים (תקופת שכירות, אופציות הארכה). "

            "2. מהם דמי השכירות, מועדי התשלום ודרכי הצמדה. "

            "3. מהן חובות וזכויות עיקריות של השוכר (לדוגמה, אחזקה, תיקונים קטנים, תשלום חשבונות, שימוש בנכס) "

            "   ושל המשכיר (לדוגמה, תיקונים גדולים, מתן שירותים). "

            "4. מהם הביטחונות והערבויות הנדרשים (שיק ביטחון, שטר חוב, ערבות בנקאית/צד ג'). "

            "5. מהם תנאי סיום ההסכם או הפרתו (לדוגמה, אפשרות יציאה מוקדמת, סעיפי פיצוי מוסכם, סעיפי הפרה יסודית)."

            "\n\nכתוב 4-6 Bullet-ים קצרים (עד 40 מילים כל אחד):"

        ),

        answer=H("אתה עו\"ד המתמחה בדיני חוזים ומקרקעין. נתח את סעיפי החוזה ומשמעותם המשפטית בהתאם לשאלת המשתמש.")

    ),

    "חוזה_שירות": dict(

        summary=H("סכם חוזה שירות: 1. מהות השירות, 2. תנאי תשלום, 3. התחייבויות הצדדים, 4. תנאי סיום."),

        answer=H("אתה עו\"ד מסחרי. נתח את תנאי ההתקשרות.")

    ),

    "מכתב_התראה": dict(

        summary=H("סכם מכתב התראה: 1. זהות השולח והנמען, 2. מהות הדרישה/טענה, 3. הסעד הנדרש ומועד לתגובה, 4. השלכות אי-תגובה."),

        answer=H("אתה עו\"ד ליטיגציה. הסבר את המשמעות המשפטית של המכתב ואת הצעדים הנדרשים.")

    ),

    "תקנון": dict(

        summary=H("סכם תקנון/מדיניות: 1. מטרות, 2. זכויות/חובות, 3. סיכונים לאי-ציות."),

        answer=H("אתה עו\"ד חברות. הסבר תוקף סעיפי התקנון.")

    ),

    "NDA": dict(

        summary=H("סכם NDA: 1. הצדדים, 2. מהו מידע סודי, 3. התחייבויות סודיות, 4. חריגים ותוקף."),

        answer=H("אתה עו\"ד מסחרי. נתח את סעיפי הסודיות והשלכותיהם.")

    ),

    "כתב_תביעה": dict(

        summary=H("סכם כתב תביעה: 1. צדדים, 2. עילות התביעה, 3. הסעדים הנדרשים, 4. לוח זמנים דיוני."),

        answer=H("אתה עו\"ד ליטיגציה. נתח את עילות התביעה והצעדים האפשריים להגנה.")

    ),

    "כתב_הגנה": dict(

        summary=H("סכם כתב הגנה: 1. צדדים, 2. טענות ההגנה המרכזיות, 3. עובדות נטענות, 4. סעדים נדרשים."),

        answer=H("אתה עו\"ד ליטיגציה. נתח את טענות ההגנה המועלות.")

    ),

    "פסק_דין": dict(

        summary=H("סכם פסק-דין: 1. שאלה משפטית מרכזית, 2. עובדות רלוונטיות, 3. הכרעות וקביעות בית-המשפט, 4. הלכה משפטית שנקבעה."),

        answer=H("אתה עו\"ד. הסבר את הלכת בית-המשפט והשלכותיה.")

    ),

    "מסמך_אחר": dict(

        summary=H("סכם את המסמך בקצרה: תקציר כללי, נקודות עיקריות, השלכות אפשריות."),

        answer=H("אתה עו\"ד. נתח את המסמך באופן כללי.")

    )

}

tmpl = lambda l, k: PROMPTS.get(l, PROMPTS["מסמך_אחר"])[k]



# ───────────── retrieval (RAG) ─────────────

embed = lambda t: model.encode([t], normalize_embeddings=True)[0] if model else np.array([0.0])



async def retrieve(q, doc):

    if not model or not judgment_index or not law_index:

        return [], []



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

        try:

            d = coll.find_one({key: _id})

            if d:

                cand[kind].setdefault(_id, {"doc": d, "scores": []})["scores"].append(score)

        except Exception:

            pass



    async def query(idx, vec):

        if not idx: return []

        vec_list = vec.tolist() if isinstance(vec, np.ndarray) else vec

        try:

            return idx.query(

                vector=vec_list,

                top_k=5,

                include_metadata=True

            ).get("matches", [])

        except Exception:

            return []



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

    cited_lines = sum(1 for l in lines if re.search(pat, l))

    return len(lines) > 0 and (cited_lines / len(lines)) >= 0.5



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



    doc_for_llm = st.session_state.get('doc', '')[:4000]



    sys = (

        tmpl(st.session_state.get("doctype", "מסמך_אחר"), "answer") +

        "\n\nהנחיות ניסוח (חובה):\n"

        "• כתוב בעברית מלאה בלבד – אין להשתמש באנגלית.\n"

        "• השתמש בלשון משפטית-מקצועית, פסקאות/סעיפים ממוספרים (לדוגמה: 1.2.3). במידת הצורך צור רשימות בנקודות (בולטים).\n"

        "• הסתמך אך ורק על המקורות שלמטה, וציין בסוגריים מרובעים את מספר-המקור ליד כל קביעה או פיסקה רלוונטית, לדוגמה: [1] או [2,3].\n"

        "• אל תציין מידע שלא נמצא במקורות. אם מידע חסר, ציין זאת במפורש.\n"

        f"\n--- מסמך שהועלה (חלק ראשוני) ---\n{doc_for_llm}" +

        f"\n\n--- חוקים רלוונטיים ---\n{fmt_sources(laws, 'חוקים')}" +

        f"\n\n--- פסקי דין רלוונטיים ---\n{fmt_sources(judg, 'פסקי דין')}"

    )



    try:

        r = await client_async_openai.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {"role": "system", "content": sys},

                {"role": "user", "content": q}

            ],

            temperature=0.2,

            max_tokens=1200

        )

        return r.choices[0].message.content.strip()

    except Exception:

        return "אירעה שגיאה בתשובה. אנא נסה שוב מאוחר יותר."



# ───────────── chat assistant ─────────────

def chat_assistant():

    st.markdown('<div class="chat-header">💬 Ask Mini Lawyer</div>', unsafe_allow_html=True)



    if "cid" not in st.session_state:

        cid = ls_get("AMLChatId") or str(uuid.uuid4())

        ls_set("AMLChatId", cid)

        st.session_state.cid = cid



    if "messages" not in st.session_state:

        try:

            conv = conv_coll.find_one({"local_storage_id": st.session_state.cid})

            st.session_state["messages"] = conv.get("messages", []) if conv else []

        except Exception:

            st.session_state["messages"] = []



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

            try:

                conv_coll.update_one(

                    {"local_storage_id": st.session_state.cid},

                    {"$set": {

                        "user_name": st.session_state["user_name"],

                        "messages": st.session_state["messages"]

                    }},

                    upsert=True

                )

            except Exception:

                pass

            st.rerun()

        return



    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    show_msgs()

    st.markdown("</div>", unsafe_allow_html=True)



    up = st.file_uploader("📄 העלה מסמך (PDF או DOCX)", type=["pdf", "docx"])

    if up:

        with st.spinner("מעבד מסמך ומסווג..."):

            try:

                raw = read_pdf(up) if up.type == "application/pdf/test" else read_docx(up)

                st.session_state.doc = "\n".join(l for l in raw.splitlines() if heb.search(l))

                if not st.session_state.doc.strip():

                    st.warning("המסמך ריק או לא מכיל תוכן עברי רלוונטי.")

                    if "doc" in st.session_state: del st.session_state.doc

                    if "doctype" in st.session_state: del st.session_state.doctype

                    if "summary" in st.session_state: del st.session_state.summary

                else:

                    st.session_state.doctype = classify_doc(st.session_state.doc)

                    st.success(f"סוג המסמך שסווג: **{DOC_LABELS.get(st.session_state.doctype, st.session_state.doctype)}**")

                    if "summary" in st.session_state:

                        del st.session_state.summary

            except Exception:

                st.error("שגיאה בעיבוד או סיווג המסמך.")

                if "doc" in st.session_state: del st.session_state.doc

                if "doctype" in st.session_state: del st.session_state.doctype

                if "summary" in st.session_state: del st.session_state.summary



    if hasattr(st.session_state, "doc") and st.session_state.doc.strip():

        if st.button("📋 סכם את המסמך"):

            with st.spinner("מסכם את המסמך..."):

                doc_for_summary = st.session_state.doc[:5000]

                prompt = tmpl(st.session_state.doctype, "summary") + "\n" + doc_for_summary

                try:

                    r = asyncio.run(

                        client_async_openai.chat.completions.create(

                            model="gpt-4o-mini",

                            messages=[{"role": "user", "content": prompt}],

                            temperature=0.1,

                            max_tokens=450

                        )

                    )

                    st.session_state.summary = ensure_he(

                        r.choices[0].message.content.strip().replace("•", "–")

                    )

                except Exception:

                    st.error("שגיאה בסיכום המסמך.")

                    st.session_state.summary = "לא ניתן לסכם את המסמך."



    if st.session_state.get("summary"):

        st.markdown("### סיכום המסמך:")

        st.markdown(

            f"<div dir='rtl' style='text-align:right'>{st.session_state.summary}</div>",

            unsafe_allow_html=True

        )

        st.markdown("---")



    async def handle(q):

        ans = ensure_he(await gen(q))

        if not await citations_ok(ans):

            ans2 = ensure_he(

                await gen(q + "\nחובה לציין מקור ממוספר (חוק/פס\"ד) אחרי כל משפט או פיסקה רלוונטית.")

            )

            if await citations_ok(ans2):

                return ans2

            else:

                return ans

        return ans



    with st.form("ask", clear_on_submit=True):

        q = st.text_area("הקלד שאלה משפטית:", height=100, key="user_question_input")

        send = st.form_submit_button("שלח")



    if send and q.strip():

        add_msg("user", q.strip())

        with st.spinner("מנסח תשובה משפטית..."):

            ans = asyncio.run(handle(q.strip()))

        add_msg("assistant", ans)

        try:

            conv_coll.update_one(

                {"local_storage_id": st.session_state.cid},

                {"$set": {

                    "messages": st.session_state["messages"],

                    "user_name": st.session_state["user_name"]

                }},

                upsert=True

            )

        except Exception:

            pass

        st.rerun()



    if st.button("🗑 נקה שיחה"):

        try:

            conv_coll.delete_one({"local_storage_id": st.session_state.cid})

        except Exception:

            pass

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

    try:

        return coll.find_one({key: doc_id})

    except Exception:

        return None



def get_explanation(scenario, doc, kind):

    name, desc = doc.get("Name", ""), doc.get("Description", "")

    if kind == "Judgment":

        prom = f"""בהתבסס על הסצנריו הבא:

{scenario}



וכן על פרטי פסק הדין הבא:

שם: {name}

תיאור: {desc}



הסבר בקצרה (עד 100 מילים) מדוע פסק דין זה רלוונטי ויכול לסייע לסצנריו, ודרג את מידת הרלוונטיות בציון מ-0 עד 10.

החזר JSON במבנה:

{{"advice":"הסבר קצר על הרלוונטיות","score":7}}"""

    else:

        prom = f"""בהתבסס על הסצנריו הבא:

{scenario}



וכן על פרטי החוק הבא:

שם: {name}

תיאור: {desc}



הסבר בקצרה (עד 100 מילים) מדוע חוק זה רלוונטי ויכול לסייע לסצנריו, ודרג את מידת הרלוונטיות בציון מ-0 עד 10.

החזר JSON במבנה:

{{"advice":"הסבר קצר על הרלוונטיות","score":6}}"""

    try:

        r = client_sync_openai.chat.completions.create(

            model="gpt-3.5-turbo",

            messages=[{"role": "user", "content": prom}],

            temperature=0.7,

            max_tokens=200

        )

        response_content = r.choices[0].message.content.strip()

        return json.loads(response_content)

    except json.JSONDecodeError:

        return {"advice": "שגיאה בפרשנות התשובה מה-AI.", "score": "N/A"}

    except Exception:

        return {"advice": "שגיאה באחזור הסבר.", "score": "N/A"}



def legal_finder_assistant():

    st.title("⚖️ Legal Finder Assistant")

    kind = st.selectbox("בחר סוג מסמך לחיפוש", ["Judgment", "Law"])

    scen = st.text_area("תאר את התרחיש המשפטי שלך (עד 500 מילים) לצורך חיפוש רלוונטי:")

    

    if st.button("🔍 מצא תוצאות רלוונטיות") and scen:

        if not model or (kind == "Judgment" and not judgment_index) or (kind == "Law" and not law_index):

            st.error("רכיבי חיפוש לא זמינים. אנא וודא שהמודל ואינדקסי ה-Pinecone מוגדרים.")

            return



        with st.spinner("מחפש מסמכים רלוונטיים..."):

            try:

                q_emb = model.encode([scen], normalize_embeddings=True)[0]

                idx = judgment_index if kind == "Judgment" else law_index

                key = "CaseNumber" if kind == "Judgment" else "IsraelLawID"

                

                matches = idx.query(

                    vector=q_emb.tolist(),

                    top_k=5,

                    include_metadata=True

                ).get("matches", [])



                if not matches:

                    st.info("לא נמצאו תוצאות רלוונטיות עבור התרחיש שתואר.")

                    return



                for m in matches:

                    _id = m.get("metadata", {}).get(key)

                    if not _id: continue



                    doc = load_document_details(kind, _id)

                    if not doc:

                        continue



                    name = doc.get("Name", f"ללא שם (ID: {_id})")

                    desc = doc.get("Description", "")

                    date_lbl = "DecisionDate" if kind == "Judgment" else "PublicationDate"

                    extra = (

                        f"<div class='law-meta'>סוג הליך: {doc.get('ProcedureType','N/A')}</div>"

                        if kind == "Judgment" else ""

                    )

                    st.markdown(

                        f"<div class='law-card'><div class='law-title'>{name} (ID:{_id})</div>"

                        f"<div class='law-description'>{desc}</div>"

                        f"<div class='law-meta'>{date_lbl}: {doc.get(date_lbl,'N/A')}</div>{extra}</div>",

                        unsafe_allow_html=True

                    )

                    with st.spinner("מנתח רלוונטיות עם AI..."):

                        res = get_explanation(scen, doc, kind)

                    st.markdown(

                        f"<span style='color:red;'>**עצת המערכת (רלוונטיות {res.get('score','N/A')}/10):** {res.get('advice','')}</span>",

                        unsafe_allow_html=True

                    )

                    with st.expander(f"הצג פרטים מלאים עבור {name}"):

                        st.json(doc)

            except Exception:

                st.error("אירעה שגיאה בחיפוש. אנא וודא שכל המערכות פועלות כהלכה.")



# ─────────────────── main ───────────────────

if app_mode == "Chat Assistant":

    chat_assistant()

else:

    legal_finder_assistant()
