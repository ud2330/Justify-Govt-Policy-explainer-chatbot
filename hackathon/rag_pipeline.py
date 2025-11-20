# rag_pipeline.py

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import re
import random
import spacy

# ------------------- Caching Models and Components -------------------
@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm_model():
    pipe = pipeline("text2text-generation", model="google/flan-t5-small", max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# ------------------- Load & Split Documents -------------------
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = file_path.split("/")[-1]
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ------------------- Create Conversational RAG Chain -------------------
def create_temp_qa_chain(documents):
    embeddings = get_embeddings_model()
    llm = get_llm_model()
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain, llm

# ------------------- Generate Contextual Suggestions -------------------
def generate_suggestions(docs, llm):
    all_questions = []

    for i, doc in enumerate(docs):
        context_text = doc.page_content[:1500]

        prompt = f"""
        You are given part of a legal/official/government document.

        Task: Generate **exactly 5 unique FAQ-style questions** based only on this text.

        Rules:
        - Each question must be **under 12 words**.
        - Cover purpose, scope, authority, penalties, rights, dates, definitions.
        - Output format must be ONLY a numbered list (1–5). No extra text.

        Text:
        {context_text}

        Questions:
        """

        raw_output = str(llm.invoke(prompt)).strip()
        print(f"\n--- DEBUG RAW SUGGESTIONS (Doc {i+1}) ---")
        print(raw_output)
        print("----------------------------------------\n")

        questions = re.findall(r'^\s*\d+[\.\)]\s*(.+)', raw_output, flags=re.MULTILINE)
        if not questions:
            questions = [line.strip("-•*1234567890. ") 
                         for line in raw_output.splitlines() if line.strip()]

        questions = [q.strip() for q in questions if len(q.split()) <= 12 and q.endswith("?")]

        all_questions.extend(questions[:5])  # add up to 5 from this doc

    # ✅ Keep only 8–10 questions total
    if len(all_questions) > 10:
        all_questions = random.sample(all_questions, 10)
    elif len(all_questions) < 8:
        fallback = [
            "What is the main purpose of this Act?",
            "Who enforces this Act?",
            "When was it enacted?",
            "Who benefits from it?",
            "What penalties are included?",
            "What rights are guaranteed?",
            "Which authority oversees compliance?",
            "What is the scope of the law?",
            "Are there any exceptions?",
            "What is the definition of key terms?"
        ]
        all_questions += fallback[: 8 - len(all_questions)]

    return all_questions

# ------------------- Legal Glossary via NER -------------------
@st.cache_resource
def get_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

def extract_legal_glossary(documents):
    """
    Returns a list of tuples: (source_file, glossary_dict)
    glossary_dict = { 'Acts/Laws': [], 'Authorities': [], 'Dates': [], 'Sections': [], 'Concepts': [] }
    """
    nlp = get_spacy_model()
    results = []

    for doc in documents:
        text = doc.page_content
        source = doc.metadata.get("source", "Unknown")
        glossary = {
            'Acts/Laws': [],
            'Authorities': [],
            'Dates': [],
            'Sections': [],
            'Concepts': []
        }

        spacy_doc = nlp(text)

        for ent in spacy_doc.ents:
            label = ent.label_
            token_text = ent.text.strip()

            # Map spaCy entity labels to glossary categories
            if label in ['LAW']:
                glossary['Acts/Laws'].append(token_text)
            elif label in ['ORG', 'GPE']:
                glossary['Authorities'].append(token_text)
            elif label in ['DATE']:
                glossary['Dates'].append(token_text)
            elif re.match(r'(Section|Sec\.?)\s*\d+', token_text, re.IGNORECASE):
                glossary['Sections'].append(token_text)
            else:
                # treat proper nouns not already captured as Concepts
                if ent.root.pos_ in ['PROPN', 'NOUN'] and len(token_text.split()) <= 4:
                    glossary['Concepts'].append(token_text)

        # Deduplicate lists
        for k in glossary.keys():
            glossary[k] = list(set(glossary[k]))

        results.append((source, glossary))

    return results
