import streamlit as st

st.set_page_config(page_title="About - Mini Lawyer", page_icon="ℹ️", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            text-align: center;
            font-size: 22px;
            color: #FFFFFF;
        }
        .content {
            font-size: 18px;
            color: #DDDDDD;
            margin-top: 30px;
            line-height: 1.6;
        }
        .container {
            background-color: #1E1E1E;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Main Page Content
st.markdown('<div class="title">ℹ️ About Mini Lawyer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Meet the Creators Behind the Innovation</div>', unsafe_allow_html=True)

# About Section
st.markdown("""
    <div class="container content">
    
    **Mini Lawyer** was created by a dedicated team of developers and data scientists,
     to provide accessible, efficient, and accurate legal assistance.  
    Our mission is to bridge the gap between advanced technologies and the legal system,
     enabling legal professionals to streamline their research and decision-making processes.

    ### Project Creators
    - **Idan Chen** 
    - **Ronen Senin** 
    - **Liav Ermias** 
    - **Ariel Politi** 
    - **Ron Elishar**
      
    ### Acknowledgments
    Special thanks to everyone who contributed to the success of this project, including open-source contributors, mentors, and the legal tech community.

    ### Contact Us
    If you'd like to collaborate or learn more about Mini Lawyer, feel free to reach out at:  
    📧 **contact@minilawyer.com**
    </div>
""", unsafe_allow_html=True)
