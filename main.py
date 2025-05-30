import streamlit as st
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
        .content {
            font-size: 18px;
            color: #DDDDDD;
            line-height: 1.6;
            background-color: #1E1E1E;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
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
    </style>
""", unsafe_allow_html=True)

def main():
    # Title
    st.markdown('<div class="title-text">⚖️ Mini Lawyer</div>', unsafe_allow_html=True)

    # Subtitle
    st.markdown('<div class="subtitle">Your Interactive Legal Assistant</div>', unsafe_allow_html=True)

    # Content section
    st.markdown("""
        <div class="content">

        #### Welcome to Mini Lawyer
        An innovative legal assistance platform that simplifies access to legal information.
        Our system provides a user-friendly and intelligent interface to search for laws, judgments, and relevant legal cases.

        ### What is Mini Lawyer?
        Mini Lawyer leverages advanced natural language processing and machine learning technologies to:
        - Search the **entire database** of Israeli laws and judgments, ensuring comprehensive legal coverage.
        - Identify relevant laws, judgments, and legal articles based on your case.
        - Analyze cases and offer statistical insights to evaluate success probabilities.
        - Guide you through the legal process, helping build strong arguments for your needs.

        Mini Lawyer is primarily designed for **lawyers** and **law students**, providing them with a professional-grade tool to streamline research and legal analysis.

        ### Technology Behind the Scenes
        **Mini Lawyer** uses a robust and modern technology stack to deliver fast and accurate results:
        - **MongoDB**: Stores the full database of Israeli **laws** and **judgments**, including complete legal text, metadata, and related references.
        - **Vector Database**: Enables semantic search using advanced embeddings to locate the most relevant laws and judgments based on user queries.
        - **Voyage Law-2 Embedding**: A specialized embedding model tailored for legal language to ensure accurate keyword extraction and contextual search.
        - **LLM Model**: An advanced **Large Language Model** processes user input, provides legal insights, and generates recommendations.

        ### How the Pipeline Works:
        1. **User Input**: The user shares their legal query or case scenario through the chat interface.
        2. **Keyword Extraction**: The input is analyzed using **Voyage Law-2 embeddings** to extract contextual keywords for precise search.
        3. **Data Retrieval**:
            - The **Vector Database** finds the most relevant laws and judgments based on the extracted keywords.
            - The **MongoDB** returns the full legal texts of laws and judgments in structured JSON format or as downloadable documents.
        4. **LLM Analysis**: The LLM model processes the retrieved content and generates insights, including summaries, evidence, and recommendations.
        5. **Response to User**: The system delivers accurate, structured, and actionable legal information to the user.

        ### Project Architecture
        The Mini Lawyer architecture is designed for scalability and efficiency:
        - **Frontend (FE)**: A clean and interactive Streamlit interface that allows users to search, filter, and retrieve legal information.
        - **Backend (BE)**: The System Manager coordinates the GPT model, **Voyage Law-2 embeddings**, MongoDB, and VectorDB for smooth data processing.
        - **Databases**:
            - **MongoDB**: Stores laws, judgments, and user chat history.
            - **VectorDB**: Facilitates fast, semantic search using legal embeddings.
        - **LLM Pipeline**: Combines GPT-Model capabilities with data retrieval to deliver precise and context-aware legal recommendations.

        This pipeline ensures that lawyers and law students can access the most relevant legal documents quickly, efficiently, and in an actionable format.
        </div>
    """, unsafe_allow_html=True)

    # Image section at the bottom
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image("images/college_logo.png", width=400)  # Adjust path and size as needed
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
