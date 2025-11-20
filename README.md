JUSTIFY — Government Laws & Policies Explainer Chatbot

A Python-based legal document assistant built using Retrieval-Augmented Generation (RAG).
Justify processes legal documents, chunks and embeds them, retrieves relevant sections, and provides accurate legal explanations using an LLM.
It also generates summaries, suggestions, and an auto-extracted legal glossary.

---

## Features

* Upload and process legal documents (PDF, DOCX, TXT)
* Chunking and embedding using Sentence Transformers
* Vector search using FAISS
* RAG-based question answering
* Summaries and insights
* Legal glossary extraction
* Suggestion generation for drafts and reports
* Clean Flask-based user interface

---

## Tech Stack

* Python
* Flask
* LangChain
* Sentence Transformers
* FAISS Vector Store
* PyPDF2, python-docx
* ReportLab (PDF generation)

---

## Project Structure

```
app.py
rag_pipeline.py
templates/
static/
uploads/
requirements.txt
README.md
```

---

## Setup

### 1. Clone the repository

```
git clone https://github.com/yourusername/justify.git
cd justify
```

### 2. Create virtual environment

```
python -m venv venv
```

### 3. Activate environment

Windows PowerShell:

```
venv\Scripts\Activate
```

Windows CMD:

```
venv\Scripts\activate.bat
```

Linux/Mac:

```
source venv/bin/activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

### 5. Start the Flask app

```
python app.py
```

Visit:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Environment Variables

Create a `.env` file:

```
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=uploads
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## Core Endpoints

* POST /upload — Upload and process document
* POST /ask — RAG-based question answering
* POST /summarize — Generate summaries
* POST /suggestions — Draft suggestions
* POST /glossary — Extract legal glossary

---

## File Support

* .pdf, .docx, .txt
* Max file size: 20MB

---

## Security Notes

* Use strong SECRET_KEY
* Validate file types
* Prefer HTTPS for deployment
* Add optional rate limiting
* Restrict maximum query length
