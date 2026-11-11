# 📄 Document Chatbot using Retrieval-Augmented Generation (RAG)

### Chat with Your Documents. Retrieve Knowledge. Get Intelligent Answers.

An AI-powered document question-answering application that allows users to upload PDF and TXT files and interact with their content through natural language conversations.

Built using **Python, Streamlit, Sentence Transformers, Scikit-learn and Groq LLM API**, this project implements Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate context-aware responses.

---

## 🚀 Project Overview

Finding specific information in lengthy documents can be time-consuming.

The Document Chatbot solves this problem by combining semantic search with Large Language Models (LLMs).

Instead of manually searching through pages, users can upload documents, ask questions and receive relevant answers based on the uploaded content.

The application retrieves the most relevant text segments and provides them as context to an AI language model to generate responses.

## 🎯 Problem Statement

Traditional document-reading methods require users to manually search, identify and interpret information from lengthy files.

This creates difficulties when working with research papers, academic notes, reports and business documents.

This project addresses the problem through an intelligent document retrieval and conversational AI system.

## ✨ Key Features

- 📁 **PDF and TXT Upload:** Upload documents for AI-powered analysis.
- 🧠 **Semantic Search:** Understands the meaning and context of user questions.
- 🔍 **Intelligent Retrieval:** Identifies relevant document sections using vector embeddings.
- 🤖 **AI-Powered Responses:** Generates contextual answers using Groq LLM.
- 💬 **Conversational Interface:** Ask multiple questions about uploaded documents.
- 📚 **Source References:** Identify the retrieved document sections supporting responses.
- ⚡ **Text Chunking:** Divides large documents into smaller manageable sections.
- 📊 **Similarity Matching:** Uses cosine similarity to identify relevant content.
- 📝 **Chat History:** Maintains conversation history during the session.

## 🏗️ System Architecture

The application follows a Retrieval-Augmented Generation workflow.

**Workflow:**

1. User uploads a PDF or TXT document.
2. The application extracts document text.
3. Extracted text is divided into smaller chunks.
4. Sentence Transformers generate semantic embeddings.
5. User submits a natural language question.
6. The question is converted into an embedding.
7. Cosine similarity retrieves the most relevant chunks.
8. Retrieved context is sent to the Groq LLM.
9. The LLM generates a context-aware response.
10. The answer and relevant source references are displayed.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Sentence Transformers | Semantic embedding generation |
| Scikit-learn | Cosine similarity and retrieval |
| NumPy | Numerical operations |
| Groq API | LLM-powered answer generation |
| PyPDF | PDF text extraction |
| Python-dotenv | Environment variable management |

## 🧠 RAG Implementation

The project implements three major RAG components:

**1. Document Processing**

Extracts text from uploaded documents and divides it into smaller chunks for efficient retrieval.

**2. Retrieval System**

Uses Sentence Transformers to generate embeddings and cosine similarity to identify document sections relevant to the user's query.

**3. Generation System**

Uses the Groq API with the `openai/gpt-oss-120b` model to generate answers based on the retrieved document context.

## 📂 Project Structure

```text
Document-Chatbot-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .env (not uploaded to GitHub)
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or above
- Git
- Groq API key

### Step 1: Clone the Repository

```bash
git clone https://github.com/pallavirpallavir154-stack/Document-Chatbot-RAG.git
```

### Step 2: Navigate to the Project

```bash
cd Document-Chatbot-RAG
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project directory.

Add:

```env
GROQ_API_KEY=your_actual_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Never upload your API key to GitHub.

### Step 5: Run the Application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal.

## 🧪 Example Use Cases

- Summarizing academic research papers.
- Extracting important information from reports.
- Understanding technical documentation.
- Asking questions about study materials.
- Retrieving information from lengthy PDF documents.

## 🔮 Future Enhancements

- Support for multiple document uploads.
- Persistent vector database integration.
- Advanced conversation memory.
- Document summarization.
- Multi-format document support.
- Improved retrieval using hybrid search.
- Deployment through Streamlit Community Cloud.

## 🎓 Learning Outcomes

Through this project, the implementation demonstrates practical understanding of:

- Retrieval-Augmented Generation (RAG).
- Natural Language Processing (NLP).
- Semantic embeddings.
- Vector similarity search.
- Large Language Model integration.
- API integration.
- Document processing.
- AI application development.

## 👩‍💻 Developer

**Pallavi R**

B.E. Computer Science and Engineering – Artificial Intelligence and Machine Learning

Sapthagiri NPS University

**Areas of Interest:** Artificial Intelligence | Machine Learning | Python | NLP | Generative AI | Data Analytics

GitHub: [pallavirpallavir154-stack](https://github.com/pallavirpallavir154-stack)

LinkedIn: [Pallavi R](https://www.linkedin.com/in/pallavi-r-pallavi-r/)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, AI and curiosity.**
