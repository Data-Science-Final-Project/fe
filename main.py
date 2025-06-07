import streamlit as st
from streamlit_lottie import st_lottie
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


st.set_page_config(page_title="Mini Lawyer", page_icon="⚖️", layout="wide")


# Custom CSS for styling
st.markdown("""
<style>
    .title-text {
        font-size: 48px;
        font-weight: bold;
        color: #9F7AEA;
        text-align: center;
        margin-bottom: 20px;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #FFFFFF;
        margin-top: -10px;
        margin-bottom: 30px;
    }
            
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        background-color: #1E1E1E;
        padding: 10px 0;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
    }
            
    .clickable-card {
        background-color: #2A2A40;
        border-radius: 15px;
        padding: 25px 20px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.2s ease;
        height: auto;
        cursor: pointer;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .clickable-card:hover {
        transform: translateY(-5px);
        background-color: #363654;
    }
    .card-title {
        font-size: 18px;
        font-weight: 600;
        margin-top: 12px;
        color: #FFFFFF;
        text-align: center;
    }

    .card-desc {
        font-size: 14px;
        color: #BBBBBB;
        margin-top: 6px;
        text-align: center;
    }

    .card:hover {
        transform: translateY(-5px);
        background-color: #363654;
        cursor: pointer;
        box-shadow: 0 0 10px #9F7AEA;
    }

    .dashboard {
        margin-top: 40px;
    }
            
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px 0;
        gap: 40px;
    }

</style>
""", unsafe_allow_html=True)


def render_dashboard_card(title):
    card_link = f"/{routes[title]}"
    
    st.markdown(f"""
    <a href="{card_link}" target="_self" style="text-decoration: none;">
        <div class="clickable-card">
            {st_lottie(icons[title], height=100, key=title)}
            <div class="card-title">{title}</div>
            <div class="card-desc">{descriptions[title]}</div>
        </div>
    </a>
    """, unsafe_allow_html=True)


# --- Lottie Loader ---
@st.cache_data
def load_lottie_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# --- Lottie Animations ---
icons = {
    "Case & Law Finder": load_lottie_file("animations/search.json"),
    "Ask for Legal Advice": load_lottie_file("animations/advice.json"),
    "Similar Case Finder": load_lottie_file("animations/similar.json"),
    "Similar Law Finder": load_lottie_file("animations/law.json"),      
    "Statistics & Insights": load_lottie_file("animations/stats.json"),
    "Legal Relationship Graph": load_lottie_file("animations/graph.json"),
    "About": load_lottie_file("animations/info.json")               
}

# --- Page Routing Dictionary ---
routes = {
    "Case & Law Finder": "search_laws_and_judgments",
    "Ask for Legal Advice": "Ask_Mini_Lawyer",
    "Similar Case Finder": "Finding_Suitable_Judgments",
    "Similar Law Finder": "Finding_Suitable_Law",
    "Statistics & Insights": "Statistics_page",
    "Legal Relationship Graph": "Graphs_Integration",
    "About": "About"
}

descriptions = {
    "Case & Law Finder": "Search for relevant cases and laws",
    "Ask for Legal Advice": "Get advisory answers on legal matters",
    "Similar Case Finder": "Compare your case with similar cases",
    "Similar Law Finder": "Find laws similar to a reference",
    "Statistics & Insights": "Explore case types and lawyer statistics",
    "Legal Relationship Graph": "Visualize connections between cases and laws",
    "About": "Learn more about Mini Lawyer"
}

def main():
    # Title
    st.markdown('<div class="title-text">⚖️ Mini Lawyer</div>', unsafe_allow_html=True)

    # Subtitle
    st.markdown('<div class="subtitle">Welcome to Mini Lawyer — your AI-powered legal assistant.<br> ' \
    'Choose a tool below to start exploring relevant laws, analyzing similar cases,'
    ' or delving into legal insights.</div>', unsafe_allow_html=True)

    # --- Dashboard Cards ---
    st.markdown('<div class="dashboard">', unsafe_allow_html=True)

    titles = list(icons.keys())
    total = len(titles)

    for row in range(0, total + 3, 3):
        remaining = total - row

        # Detect if this is the extra row reserved for the image
        if row >= total:
            cols = st.columns([1, 1, 1])
            with cols[1]:
                st.image("images/college_logo.png", width=400)
        elif remaining >= 3:
            cols = st.columns(3)
            for i in range(3):
                title = titles[row + i]
                with cols[i]:
                    render_dashboard_card(title)
        elif remaining == 2:
            cols = st.columns(3)
            for i in range(2):
                title = titles[row + i]
                with cols[i]:
                    render_dashboard_card(title)
        elif remaining == 1:
            cols = st.columns([1, 2, 1])
            title = titles[row]
            with cols[1]:
                render_dashboard_card(title)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
