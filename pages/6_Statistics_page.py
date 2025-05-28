from app_resources import mongo_client
import torch
from matplotlib import rcParams
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from streamlit_option_menu import option_menu
import os
import streamlit as st

# st.set_page_config must be the first Streamlit command
st.set_page_config(page_title="Statistics & Lawyers Dashboard",
                   page_icon="📊", layout="wide")


torch.classes.__path__ = []

# --- PAGE CONFIG ---
rcParams['font.family'] = 'DejaVu Sans'

DATABASE_NAME = os.getenv('DATABASE_NAME')
LAWS_COLLECTION = "laws"
JUDGMENTS_COLLECTION = "judgments"

# --- TOGGLE ---
selected_dashboard = option_menu(
    None,
    ["General Statistics", "Lawyers Statistics"],
    icons=["bar-chart", "person-badge"],
    orientation="horizontal",
    styles={
            "container": {
                "padding": "0!important",
                "background-color": "#2A2A40",
                "border-radius": "10px",
                "margin-bottom": "20px"
            },
            "icon": {
                "color": "#FFFFFF",
                "font-size": "20px"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "color": "#CCCCCC",
                "padding": "10px 25px",
                "border-radius": "10px",
                "transition": "0.3s ease",
            },
            "nav-link-selected": {
                "background-color": "#9F7AEA",
                "color": "white",
                "font-weight": "bold",
                "border-radius": "10px",
            }
        }
)

# --- STATISTICS DASHBOARD ---
if selected_dashboard == "General Statistics":
    @st.cache_data(show_spinner=False)
    def load_laws_data():
        db = mongo_client[DATABASE_NAME]
        docs = list(db[LAWS_COLLECTION].find(
            {}, {"PublicationDate": 1, "IsBasicLaw": 1, "IsFavoriteLaw": 1}
        ))
        df = pd.DataFrame(docs)
        if not df.empty:
            df = df.drop(columns=["_id"], errors="ignore")
            df["PublicationDate"] = pd.to_datetime(
                df["PublicationDate"], errors="coerce")
        return df

    @st.cache_data(show_spinner=False)
    def load_judgments_data():
        db = mongo_client[DATABASE_NAME]
        docs = list(db[JUDGMENTS_COLLECTION].find(
            {}, {"DecisionDate": 1, "CourtType": 1, "ProcedureType": 1, "District": 1}
        ))
        df = pd.DataFrame(docs)
        if not df.empty:
            df = df.drop(columns=["_id"], errors="ignore")
            df["DecisionDate"] = pd.to_datetime(
                df["DecisionDate"], errors="coerce")
        return df

    st.title("General Statistics")
    st.info("Loading data...")
    df_judgments = load_judgments_data()
    df_laws = load_laws_data()
    st.write("Data loaded.")

    st.header("Judgments Statistics")
    if not df_judgments.empty:
        timeline_chart = alt.Chart(df_judgments).mark_bar().encode(
            x=alt.X("year(DecisionDate):O", title="Year"),
            y=alt.Y("count()", title="Number of Judgments")
        ).properties(title="Judgments Timeline (Decision Date)")
        st.altair_chart(timeline_chart, use_container_width=True)

        if "CourtType" in df_judgments.columns:
            court_chart = alt.Chart(df_judgments).mark_arc().encode(
                theta=alt.Theta("count()", stack=True),
                color=alt.Color(
                    "CourtType:N", legend=alt.Legend(title="Court Type"))
            ).properties(title="Distribution of Court Type")
            st.altair_chart(court_chart, use_container_width=True)

        if "ProcedureType" in df_judgments.columns:
            procedure_chart = alt.Chart(df_judgments).mark_arc().encode(
                theta=alt.Theta("count()", stack=True),
                color=alt.Color("ProcedureType:N",
                                legend=alt.Legend(title="Procedure Type"))
            ).properties(title="Distribution of Procedure Type")
            st.altair_chart(procedure_chart, use_container_width=True)

        if "District" in df_judgments.columns:
            district_chart = alt.Chart(df_judgments).mark_arc().encode(
                theta=alt.Theta("count()", stack=True),
                color=alt.Color(
                    "District:N", legend=alt.Legend(title="District"))
            ).properties(title="Distribution of District")
            st.altair_chart(district_chart, use_container_width=True)
    else:
        st.info("No judgments data available.")

    st.markdown("---")
    st.header("Laws Statistics")
    if not df_laws.empty:
        timeline_laws = alt.Chart(df_laws).mark_bar().encode(
            x=alt.X("year(PublicationDate):O", title="Year"),
            y=alt.Y("count()", title="Number of Laws")
        ).properties(title="Laws Timeline (Publication Date)")
        st.altair_chart(timeline_laws, use_container_width=True)

        if "IsBasicLaw" in df_laws.columns:
            basic_chart = alt.Chart(df_laws).mark_bar().encode(
                x=alt.X("IsBasicLaw:N", title="Is Basic Law (True/False)"),
                y=alt.Y("count()", title="Count")
            ).properties(title="Distribution of IsBasicLaw")
            st.altair_chart(basic_chart, use_container_width=True)

        if "IsFavoriteLaw" in df_laws.columns:
            favorite_chart = alt.Chart(df_laws).mark_bar().encode(
                x=alt.X("IsFavoriteLaw:N", title="Is Favorite Law (True/False)"),
                y=alt.Y("count()", title="Count")
            ).properties(title="Distribution of IsFavoriteLaw")
            st.altair_chart(favorite_chart, use_container_width=True)
    else:
        st.info("No laws data available.")
    st.info("Statistics page loaded successfully.")

# --- LAWYERS DASHBOARD ---
if selected_dashboard == "Lawyers Statistics":
    @st.cache_data
    def load_data():
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, "data",
                                 "lawyers_with_case_info.csv")
        df = pd.read_csv(file_path)
        return df

    def reverse_hebrew(s):
        try:
            return s[::-1]
        except:
            return s

    df = load_data()
    df = df.dropna(subset=["ProcedureType", "CourtType", "LawyerName"])

    st.title("⚖️ דשבורד עורכי דין ממסמכים משפטיים")
    st.markdown(
        "🔍 *הנתונים מבוססים על מאגר חלקי מתוך מסמכים משפטיים שנשלפו אוטומטית. "
        "ייתכן שעורכי הדין פעילים גם בתיקים שלא נכללו במערכת.*"
    )
    st.markdown("---")
    st.header("🔝 טופ 5 עורכי דין לפי תחום וערכאה")

    procedure_options = ["הכל"] + sorted(df["ProcedureType"].dropna().unique())
    court_options = ["הכל"] + sorted(df["CourtType"].dropna().unique())

    col_filters = st.columns(2)
    selected_proc = col_filters[0].selectbox(
        "בחר תחום משפטי:", procedure_options)
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

    if top_lawyers.empty:
        st.warning("⚠️ אין נתונים עבור הקטגוריות שנבחרו.")
    else:
        with st.container():
            card_cols = st.columns([1, 1])
            with card_cols[0]:
                st.subheader("🏆 טופ 5 עורכי דין")
                fig, ax = plt.subplots(figsize=(4, 2.5))
                colors = plt.cm.viridis(
                    np.linspace(0.2, 0.8, len(top_lawyers)))
                ax.bar([reverse_hebrew(name) for name in top_lawyers["LawyerName"]],
                       top_lawyers["Count"], color=colors)
                ax.set_ylabel(reverse_hebrew("מספר תיקים"),
                              fontsize=9, ha='right')
                ax.set_xlabel(reverse_hebrew("שם עורך דין"),
                              fontsize=9, ha='right')
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                plt.xticks(rotation=45, ha='right', fontsize=8)
                st.pyplot(fig)
            with card_cols[1]:
                st.subheader("📋 טבלת טופ 5")
                st.dataframe(top_lawyers.reset_index(
                    drop=True), use_container_width=True)

    st.markdown("---")
    st.header("👤 פרופיל אישי - סטטיסטיקה לכל עורך דין")
    lawyer_names = sorted(df["LawyerName"].dropna().unique())
    selected_lawyer = st.selectbox("בחר עורך דין:", lawyer_names)
    lawyer_df = df[df["LawyerName"] == selected_lawyer]
    total_cases = lawyer_df.shape[0]
    unique_fields = lawyer_df["ProcedureType"].nunique()
    unique_courts = lawyer_df["CourtType"].nunique()

    with st.container():
        stats_cols = st.columns(3)
        stats_cols[0].metric("📁 מספר תיקים", total_cases)
        stats_cols[1].metric("📚 תחומי עיסוק שונים", unique_fields)
        stats_cols[2].metric("🏛 ערכאות שונות", unique_courts)

    with st.container():
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.subheader("📊 התפלגות לפי תחום משפטי")
            pie_data = lawyer_df["ProcedureType"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(4, 2.5))
            colors_pie = plt.cm.Paired(np.linspace(0, 1, len(pie_data)))
            ax1.pie(pie_data, labels=[reverse_hebrew(label) for label in pie_data.index],
                    autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 8})
            ax1.axis('equal')
            st.pyplot(fig1)
        with chart_cols[1]:
            st.subheader("🏛 התפלגות לפי ערכאה")
            bar_data = lawyer_df["CourtType"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(4, 2.5))
            colors_bar = plt.cm.Set2(np.linspace(0.2, 0.8, len(bar_data)))
            ax2.bar([reverse_hebrew(label) for label in bar_data.index],
                    bar_data.values, color=colors_bar)
            ax2.set_ylabel(reverse_hebrew("מספר תיקים"),
                           fontsize=9, ha='right')
            ax2.set_xlabel(reverse_hebrew("סוג ערכאה"), fontsize=9, ha='right')
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.xticks(rotation=45, ha='right', fontsize=8)
            st.pyplot(fig2)

    st.markdown("---")
    st.subheader("📄 רשימת תיקים של עורך הדין")
    if "CaseName" in lawyer_df.columns and "CaseURL" in lawyer_df.columns:
        def split_case_name(full_name):
            parts = full_name.split(",", 1)
            if len(parts) == 2:
                court = parts[0].strip()
                case_name = parts[1].strip()
                return court, case_name
            else:
                return "", full_name
        lawyer_df.loc[:, ["בית משפט", "שם תיק"]] = lawyer_df["CaseName"].apply(
            lambda x: pd.Series(split_case_name(x)))

        def make_button(url):
            clean_url = str(url).strip().replace('\n', '')
            return f"""
            <a href=\"{clean_url}\" target=\"_blank\" style=\"text-decoration: none;\">
                <button style=\"
                    padding:6px 10px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    font-size: 14px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                \" 
                onmouseover=\"this.style.backgroundColor='#45a049'\"
                onmouseout=\"this.style.backgroundColor='#4CAF50'\"
                >מעבר לפסק הדין</button>
            </a>
            """
        lawyer_df.loc[:, "מעבר לפסק הדין"] = lawyer_df["CaseURL"].apply(
            make_button)
        final_table = lawyer_df[["שם תיק", "בית משפט", "מעבר לפסק הדין"]]
        st.write(
            final_table.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )
    else:
        st.info("אין מידע על תיקים לעורך הדין שנבחר.")
