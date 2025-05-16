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
        .law-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
            background-color: #f9f9f9;
        }
        .law-title {
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }
        .law-description {
            font-size: 16px;
            color: #444;
            margin: 10px 0;
        }
        .law-meta {
            font-size: 14px;
            color: #555;
        }
        .pagination-controls {
            margin-top: 20px;
            text-align: center;
        }
        .stButton>button {
            background-color: #7ce38b;
            color: white;
            font-size: 14px;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #7ce38b;
        }
        .toggle-container {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }
        .toggle-button {
            background-color: #f0f0f0;
            border: 2px solid #7ce38b;
            padding: 10px 20px;
            margin: 0 10px;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .toggle-button.active {
            background-color: #7ce38b;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for search type
if 'search_type' not in st.session_state:
    st.session_state.search_type = 'Laws'

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
        icons=["book", "gavel"],
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#7ce38b", "font-size": "20px"},
            "nav-link": {
                "font-size": "18px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {
                "background-color": "#7ce38b",
                "color": "white",
                "font-weight": "bold",
            },
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
        if len(date_range) == 2:
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

                        if st.button(f"View Full Details for {law['IsraelLawID']}", key=f"details_{law['IsraelLawID']}"):
                            with st.spinner("Loading full details..."):
                                full_law = load_full_law_details(
                                    client, law['IsraelLawID'])
                                if full_law:
                                    st.json(full_law)
                                else:
                                    st.error(
                                        f"Unable to load full details for law ID {law['IsraelLawID']}")

    else:  # Judgments
        # Judgments filters
        with st.expander("Filters"):
            case_number = st.text_input(
                "Filter by Case Number (Regex)", key="case_number_filter")
            judgments_name = st.text_input(
                "Filter by Name (Regex)", key="judgments_name_filter")
            procedure_types = get_procedure_types(client)
            procedure_types = [x for x in procedure_types if x not in {
                '', ', , , , ', ' ,בג"ץ', 'בג"ץ, '}]
            procedure_type = st.selectbox("Filter by Procedure Type", options=["All"] + procedure_types,
                                          key="procedure_type_filter")
            date_range = st.date_input("Filter by Publication Date Range", [])

        # Build filters for judgments
        filters = {}
        if case_number:
            filters["CaseNumber"] = {"$regex": case_number, "$options": "i"}
        if judgments_name:
            filters["Name"] = {"$regex": judgments_name, "$options": "i"}
        if procedure_type != "All":
            filters["ProcedureType"] = procedure_type
        if len(date_range) == 2:
            start_date, end_date = date_range
            filters["PublicationDate"] = {
                "$gte": datetime.combine(start_date, datetime.min.time()),
                "$lte": datetime.combine(end_date, datetime.max.time())
            }

        # Query judgments
        with st.spinner("Loading Judgments..."):
            judgments = query_judgments(
                client, filters, (st.session_state["page"] - 1) * 10, 10)
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

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button(f"View Full Details for {judgment['CaseNumber']}",
                                         key=f"details_{judgment['CaseNumber']}"):
                                st.json(judgment)
                        with col2:
                            documents = judgment.get('Documents', [])
                            if documents and isinstance(documents, list) and 'url' in documents[0]:
                                document_url = documents[0]['url']
                                st.markdown(
                                    f"""
                                    <a href="{document_url}" target="_blank" style="text-decoration:none;">
                                        <button style="
                                            background-color:#7ce38b;
                                            color:white;
                                            border:none;
                                            padding:8px 16px;
                                            border-radius:5px;
                                            cursor:pointer;
                                            font-size:14px;">
                                            Download Judgment
                                        </button>
                                    </a>
                                    """,
                                    unsafe_allow_html=True
                                )

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
