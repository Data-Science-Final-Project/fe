import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os

# 📄 הגדרת הדף
rcParams['font.family'] = 'Arial'
st.set_page_config(page_title="דשבורד עורכי דין", layout="wide")

# 🎯 טעינת הדאטה
@st.cache_data
def load_data():
    # df = pd.read_csv("lawyers_with_case_info.csv")
    current_dir = os.path.dirname(__file__)  # Get current script directory
    file_path = os.path.join(current_dir, "lawyers_with_case_info.csv")
    df = pd.read_csv(file_path)
    return df
    return df

def reverse_hebrew(s):
    try:
        return s[::-1]
    except:
        return s

df = load_data()
df = df.dropna(subset=["ProcedureType", "CourtType", "LawyerName"])

# 🏛️ כותרת עמוד ראשית
st.title("⚖️ דשבורד עורכי דין ממסמכים משפטיים")
st.markdown(
    "🔍 *הנתונים מבוססים על מאגר חלקי מתוך מסמכים משפטיים שנשלפו אוטומטית. "
    "ייתכן שעורכי הדין פעילים גם בתיקים שלא נכללו במערכת.*"
)

# ---------------------------------------------------
# 🎯 סינון כללי
# ---------------------------------------------------
st.markdown("---")
st.header("🔝 טופ 5 עורכי דין לפי תחום וערכאה")

procedure_options = ["הכל"] + sorted(df["ProcedureType"].dropna().unique())
court_options = ["הכל"] + sorted(df["CourtType"].dropna().unique())

col_filters = st.columns(2)
selected_proc = col_filters[0].selectbox("בחר תחום משפטי:", procedure_options)
selected_court = col_filters[1].selectbox("בחר ערכאה:", court_options)

filtered = df.copy()

if selected_proc != "הכל":
    filtered = filtered[filtered["ProcedureType"] == selected_proc]

if selected_court != "הכל":
    filtered = filtered[filtered["CourtType"] == selected_court]

top_lawyers = (
    filtered.groupby("LawyerName")
    .size()
    .reset_index(name="Count")
    .sort_values("Count", ascending=False)
    .head(5)
)

# ---------------------------------------------------
# 🖼️ תצוגה ב-Card לטופ 5 עורכי דין
# ---------------------------------------------------
if top_lawyers.empty:
    st.warning("⚠️ אין נתונים עבור הקטגוריות שנבחרו.")
else:
    with st.container():
        card_cols = st.columns([1, 1])

        with card_cols[0]:
            st.subheader("🏆 טופ 5 עורכי דין")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_lawyers)))
            ax.bar([reverse_hebrew(name) for name in top_lawyers["LawyerName"]], top_lawyers["Count"], color=colors)
            # ax.bar(top_lawyers["LawyerName"], top_lawyers["Count"], color=colors)
            ax.set_ylabel(reverse_hebrew("מספר תיקים"), fontsize=9, ha='right')
            ax.set_xlabel(reverse_hebrew("שם עורך דין"), fontsize=9, ha='right')
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.xticks(rotation=45, ha='right', fontsize=8)
            st.pyplot(fig)

        with card_cols[1]:
            st.subheader("📋 טבלת טופ 5")
            st.dataframe(top_lawyers.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------
# 👤 סטטיסטיקה לפי עורך דין
# ---------------------------------------------------
st.markdown("---")
st.header("👤 פרופיל אישי - סטטיסטיקה לכל עורך דין")

lawyer_names = sorted(df["LawyerName"].dropna().unique())
selected_lawyer = st.selectbox("בחר עורך דין:", lawyer_names)

lawyer_df = df[df["LawyerName"] == selected_lawyer]

total_cases = lawyer_df.shape[0]
unique_fields = lawyer_df["ProcedureType"].nunique()
unique_courts = lawyer_df["CourtType"].nunique()

# ---------------------------------------------------
# 🧮 תצוגת נתונים מספריים
# ---------------------------------------------------
with st.container():
    stats_cols = st.columns(3)
    stats_cols[0].metric("📁 מספר תיקים", total_cases)
    stats_cols[1].metric("📚 תחומי עיסוק שונים", unique_fields)
    stats_cols[2].metric("🏛 ערכאות שונות", unique_courts)

# ---------------------------------------------------
# 📊 גרפים אישיים
# ---------------------------------------------------
with st.container():
    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.subheader("📊 התפלגות לפי תחום משפטי")
        pie_data = lawyer_df["ProcedureType"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 2.5))
        colors_pie = plt.cm.Paired(np.linspace(0, 1, len(pie_data)))
        ax1.pie(pie_data, labels=[reverse_hebrew(label) for label in pie_data.index], autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 8})
        # ax1.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 8})
        ax1.axis('equal')
        st.pyplot(fig1)

    with chart_cols[1]:
        st.subheader("🏛 התפלגות לפי ערכאה")
        bar_data = lawyer_df["CourtType"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        colors_bar = plt.cm.Set2(np.linspace(0.2, 0.8, len(bar_data)))
        ax2.bar([reverse_hebrew(label) for label in bar_data.index], bar_data.values, color=colors_bar)
        # ax2.bar(bar_data.index, bar_data.values, color=colors_bar)
        ax2.set_ylabel(reverse_hebrew("מספר תיקים"), fontsize=9, ha='right')
        ax2.set_xlabel(reverse_hebrew("סוג ערכאה"), fontsize=9, ha='right')
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.xticks(rotation=45, ha='right', fontsize=8)
        st.pyplot(fig2)

# ---------------------------------------------------
# 🗂️ הצגת רשימת תיקים לעורך הדין
# ---------------------------------------------------
st.markdown("---")
st.subheader("📄 רשימת תיקים של עורך הדין")

if "CaseName" in lawyer_df.columns and "CaseURL" in lawyer_df.columns:

    def split_case_name(full_name):
        """
        פיצול שם תיק לשם בית משפט + שם תיק לפי הפסיק הראשון
        """
        parts = full_name.split(",", 1)
        if len(parts) == 2:
            court = parts[0].strip()
            case_name = parts[1].strip()
            return court, case_name
        else:
            return "", full_name

    # יצירת עמודות "בית משפט" ו"שם תיק"
    lawyer_df[["בית משפט", "שם תיק"]] = lawyer_df["CaseName"].apply(lambda x: pd.Series(split_case_name(x)))

    # פונקציה לבניית כפתור לינק מעוצב
    def make_button(url):
        clean_url = str(url).strip().replace('\n', '')
        return f"""
        <a href="{clean_url}" target="_blank" style="text-decoration: none;">
            <button style="
                padding:6px 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                cursor: pointer;
                transition: background-color 0.3s;
            " 
            onmouseover="this.style.backgroundColor='#45a049'"
            onmouseout="this.style.backgroundColor='#4CAF50'"
            >מעבר לפסק הדין</button>
        </a>
        """

    lawyer_df["מעבר לפסק הדין"] = lawyer_df["CaseURL"].apply(make_button)

    final_table = lawyer_df[["שם תיק", "בית משפט", "מעבר לפסק הדין"]]

    st.write(
        final_table.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

else:
    st.info("אין מידע על תיקים לעורך הדין שנבחר.")