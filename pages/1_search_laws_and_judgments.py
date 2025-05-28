import streamlit as st
from app_resources import mongo_client
from dotenv import load_dotenv
import os
from datetime import datetime
from streamlit_option_menu import option_menu

# Load environment variables
load_dotenv()

# Database details
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')

# Custom CSS for Styling
st.markdown("""
<style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #1C1C2E;
        color: #ECECEC;
    }

    .law-card {
        background-color: #2A2A40;
        border-left: 6px solid #9F7AEA;
        border-radius: 10px;
        padding: 20px 25px;
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        direction:rtl;
    }

    .law-title {
        font-size: 22px;
        font-weight: 600;
        color: #ECECEC;
        margin-bottom: 8px;
        direction:rtl;
    }

    .law-description {
        font-size: 16px;
        color: #CCCCCC;
        margin: 8px 0 12px 0;
        line-height: 1.6;
    }

    .law-meta {
        font-size: 14px;
        color: #AAAAAA;
    }

    .stButton>button {
        background-color: #9F7AEA !important;
        color: white !important;
        font-weight: 500;
        padding: 8px 18px;
        border: none;
        border-radius: 5px;
        transition: background-color 0.2s ease;
    }

    .stButton>button:hover {
        background-color: #805AD5 !important;
    }

    .pagination-controls {
        margin: 30px 0 0 0;
        text-align: center;
    }

    .filters-section {
        background-color: #2A2A40;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
        box-shadow: inset 0 0 3px rgba(255,255,255,0.05);
    }

    .toggle-button {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for search type
if 'search_type' not in st.session_state:
    st.session_state.search_type = 'Laws'


def toggle_expansion(key):
    st.session_state[key] = not st.session_state.get(key, False)

def is_expanded(key):
    return st.session_state.get(key, False)


# Functions for Laws


def query_laws(client, filters=None, skip=0, limit=10):
    try:
        db = client[DATABASE_NAME]
        collection = db["laws"]
        pipeline = []
        if filters:
            pipeline.append({"$match": filters})
        pipeline.append({"$sort": {"IsraelLawID": 1}})
        pipeline.append({"$skip": skip})
        pipeline.append({"$limit": limit})
        pipeline.append({"$project": {"Segments": 0}})
        laws = list(collection.aggregate(pipeline))
        return laws
    except Exception as e:
        st.error(f"Error querying laws: {str(e)}")
        return []


def count_laws(client, filters=None):
    try:
        db = client[DATABASE_NAME]
        collection = db["laws"]
        if filters:
            return collection.count_documents(filters)
        return collection.estimated_document_count()
    except Exception as e:
        st.error(f"Error counting laws: {str(e)}")
        return 0


def load_full_law_details(client, law_id):
    try:
        db = client[DATABASE_NAME]
        collection = db["laws"]
        law = collection.find_one({"IsraelLawID": law_id})
        return law
    except Exception as e:
        st.error(f"Error fetching full details for law ID {law_id}: {str(e)}")
        return None

# Functions for Judgments

def get_procedure_types(client):
    try:
        db = client[DATABASE_NAME]
        collection = db["judgments"]
        procedure_types = collection.distinct("ProcedureType")
        return sorted(procedure_types)
    except Exception as e:
        st.error(f"Error fetching ProcedureType values: {str(e)}")
        return []


def query_judgments(client, filters=None, skip=0, limit=10):
    try:
        db = client[DATABASE_NAME]
        collection = db["judgments"]
        pipeline = []
        if filters:
            pipeline.append({"$match": filters})
        pipeline.append({"$sort": {"CaseNumber": 1}})
        pipeline.append({"$skip": skip})
        pipeline.append({"$limit": limit})
        judgments = list(collection.aggregate(pipeline))
        return judgments
    except Exception as e:
        st.error(f"Error querying judgments: {str(e)}")
        return []


def count_judgments(client, filters=None):
    try:
        db = client[DATABASE_NAME]
        collection = db["judgments"]
        if filters:
            return collection.count_documents(filters)
        return collection.estimated_document_count()
    except Exception as e:
        st.error(f"Error counting judgments: {str(e)}")
        return 0


def reset_page():
    st.session_state["page"] = 1


def main():
    st.title("📜 Legal Search")

    # Modern animated toggle using streamlit-option-menu
    search_type = option_menu(
        None,
        ["Laws", "Judgments"],
        icons=["book", "briefcase"],
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
    st.session_state.search_type = search_type

    # Initialize pagination state
    if "page" not in st.session_state:
        st.session_state["page"] = 1

    client = mongo_client

    if search_type == "Laws":
        
        # Laws filters
        with st.expander("Filters"):
            israel_law_id = st.number_input(
                "Filter by IsraelLawID (Exact Match)",
                min_value=0,
                step=1,
                value=0,
                key="law_id_filter",
                on_change=reset_page
            )
            law_name = st.text_input(
                "Filter by Name (Regex)",
                key="law_name_filter",
                on_change=reset_page
            )
            date_range = st.date_input(
                "Filter by Publication Date Range",
                [],
                key="date_filter",
                on_change=reset_page
            )

        # Build filters for laws
        filters = {}
        if israel_law_id > 0:
            filters["IsraelLawID"] = israel_law_id
        if law_name:
            filters["Name"] = {"$regex": law_name, "$options": "i"}
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filters["PublicationDate"] = {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }

        # Query laws
        with st.spinner("Loading laws..."):
            laws = query_laws(
                client, filters, (st.session_state["page"] - 1) * 10, 10)
            total_items = count_laws(client, filters)

            if laws:
                st.markdown(
                    f"### Page {st.session_state['page']} (Showing {len(laws)} of {total_items} laws)")
                for law in laws:
                    law_description = law.get(
                        "Description", "").strip() or "אין תיאור לחוק זה"
                    with st.container():
                        st.markdown(f"""
                            <div class="law-card">
                                <div class="law-title">{law['Name']} (ID: {law['IsraelLawID']})</div>
                                <div class="law-description">{law_description}</div>
                                <div class="law-meta">Publication Date: {law.get('PublicationDate', 'N/A')}</div>
                            </div>
                        """, unsafe_allow_html=True)

                        law_id = law['IsraelLawID']
                        button_key = f"law_button_{law_id}"
                        state_key = f"law_expanded_{law_id}"

                        if state_key not in st.session_state:
                            st.session_state[state_key] = False

                        st.button(
                            "🔍 View Details" if not st.session_state.get(state_key, False) else "🙈 Hide Details",
                            key=button_key,
                            on_click=toggle_expansion,
                            args=(state_key,)
                        )

                        if st.session_state[state_key]:
                            full_law = load_full_law_details(client, law_id)
                            if not full_law:
                                st.error(f"Could not load full details for law ID {law_id}")
                            else:
                                segments_html = ''.join([
                                f"""
                                <div style='margin-top:12px;'>
                                    <p><strong>📑 Section {s.get('SectionNumber', '')}</strong>: {s.get('SectionDescription', '')}</p>
                                    <p style='color:#BBBBBB; margin-right:10px;'>{s.get('SectionContent', '')}</p>
                                </div>
                                """ for s in full_law.get("Segments", [])
                            ])

                            st.markdown(f"""
                                <div style='padding:15px; background:#1C1C2E; border:1px solid #9F7AEA; border-radius:10px; margin-top:10px; direction:rtl; text-align:right;'>
                                    <p><strong>📘 Law ID:</strong> {full_law.get("IsraelLawID")}</p>
                                    <p><strong>📄 Name:</strong> {full_law.get("Name")}</p>
                                    <p><strong>📌 Basic Law:</strong> {"✅" if full_law.get("IsBasicLaw", False) else "❌"}</p>
                                    <hr style='border:1px solid #444;' />
                                    {segments_html}
                                </div>
                            """, unsafe_allow_html=True)




    else:  # Judgments
        with st.expander("Filters"):
            case_number = st.text_input(
                "Filter by Case Number (Regex)",
                key="case_number_filter",
                on_change=reset_page
            )
            judgments_name = st.text_input(
                "Filter by Name (Regex)",
                key="judgments_name_filter",
                on_change=reset_page
            )
            procedure_types = get_procedure_types(client)
            procedure_types = [x for x in procedure_types if x not in {
                '', ', , , , ', ' ,בג"ץ', 'בג"ץ, '}]
            procedure_type = st.selectbox(
                "Filter by Procedure Type",
                options=["All"] + procedure_types,
                key="procedure_type_filter",
                on_change=reset_page
            )
            date_range = st.date_input(
                "Filter by Publication Date Range",
                key="judgment_date_range",
                on_change=reset_page
            )

        # Build filters
        filters = {}
        case_number = st.session_state.get("case_number_filter", "")
        judgments_name = st.session_state.get("judgments_name_filter", "")
        procedure_type = st.session_state.get("procedure_type_filter", "All")
        date_range = st.session_state.get("judgment_date_range", [])

        if case_number:
            filters["CaseNumber"] = {"$regex": case_number, "$options": "i"}
        if judgments_name:
            filters["Name"] = {"$regex": judgments_name, "$options": "i"}
        if procedure_type != "All":
            filters["ProcedureType"] = procedure_type
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filters["PublicationDate"] = {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }


        # Initial load trigger
        with st.spinner("Loading Judgments..."):
            judgments = query_judgments(
                client, filters, (st.session_state["page"] - 1) * 10, 10
            )
            total_items = count_judgments(client, filters)

        if judgments:
            st.markdown(
                f"### Page {st.session_state['page']} (Showing {len(judgments)} of {total_items} judgments)")
            for judgment in judgments:
                judgment_description = judgment.get(
                    "Description", "").strip() or "אין תיאור לפסק הדין זה"
                with st.container():
                    st.markdown(f"""
                        <div class="law-card">
                            <div class="law-title">{judgment['Name']} (ID: {judgment['CaseNumber']})</div>
                            <div class="law-description">{judgment_description}</div>
                            <div class="law-meta">Publication Date: {judgment.get('DecisionDate', 'N/A')}</div>
                            <div class="law-meta">Procedure Type: {judgment.get('ProcedureType', 'N/A')}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    case_number = judgment['CaseNumber']
                    button_key = f"judgment_button_{case_number}"
                    state_key = f"judgment_expanded_{case_number}"

                    if state_key not in st.session_state:
                        st.session_state[state_key] = False

                    st.button(
                        "🔍 View Details" if not st.session_state.get(state_key, False) else "🙈 Hide Details",
                        key=button_key,
                        on_click=toggle_expansion,
                        args=(state_key,)
                    )

                    if st.session_state[state_key]:
                        st.markdown(f"""
                            <div style='padding:15px; background:#1C1C2E; border:1px solid #9F7AEA; border-radius:10px; margin-top:10px; direction:rtl; text-align:right;'>
                                <p><strong>📄 שם:</strong> {judgment.get("Name", "N/A")}</p>
                                <p><strong>📁 מספר תיק:</strong> {judgment.get("CaseNumber", "N/A")}</p>
                                <p><strong>🏛️ סוג בית משפט:</strong> {judgment.get("CourtType", "N/A")}</p>
                                <p><strong>⚖️ סוג הליך:</strong> {judgment.get("ProcedureType", "N/A")}</p>
                                <p><strong>👩‍⚖️ שופט:</strong> {judgment.get("Judge", "N/A")}</p>
                                <p><strong>🌍 מחוז:</strong> {judgment.get("District", "N/A")}</p>
                                <p><strong>📅 תאריך החלטה:</strong> {judgment.get("DecisionDate", "N/A")}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        documents = judgment.get('Documents', [])
                        if documents and isinstance(documents, list) and 'url' in documents[0]:
                            document_url = documents[0]['url']
                            st.markdown(f"""
                                <a href="{document_url}" target="_blank">
                                    <button style='background-color:#9F7AEA; color:white; padding:8px 16px; border:none; border-radius:5px; cursor:pointer; margin-top:10px;'>
                                        📥 Download
                                    </button>
                                </a>
                            """, unsafe_allow_html=True)

    # Pagination controls
    if total_items > 0:
        total_pages = (total_items + 9) // 10
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Previous Page") and st.session_state["page"] > 1:
                st.session_state["page"] -= 1
        with col2:
            st.write(f"Page {st.session_state['page']} of {total_pages}")
        with col3:
            if st.button("Next Page") and st.session_state["page"] < total_pages:
                st.session_state["page"] += 1
    else:
        st.warning(f"No {search_type.lower()} found with the applied filters.")


if __name__ == "__main__":
    main()
