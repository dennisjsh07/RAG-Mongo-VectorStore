# imports

import os
from dotenv import load_dotenv
import streamlit as st
import re
from pymongo import MongoClient
from langchain_community import document_loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI