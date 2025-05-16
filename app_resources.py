import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st

load_dotenv()


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("intfloat/multilingual-e5-large")


@st.cache_resource
def init_pinecone_client():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    return pinecone.Pinecone(api_key=pinecone_api_key)


@st.cache_resource
def get_mongo_client():
    mongo_uri = os.getenv("MONGO_URI")
    return MongoClient(mongo_uri)


# EXPORT CACHED INSTANCES
model = load_embedding_model()
pinecone_client = init_pinecone_client()
mongo_client = get_mongo_client()
