import os
import streamlit as st
import numpy as np

from io import BytesIO
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv


# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="Document Chatbot",
    page_icon="📄",
    layout="wide"
)

API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

st.title("📄 Document Chatbot")
st.caption(
    "Upload your documents and ask questions "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==========================================
# 2. SESSION STATE
# ==========================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 3. LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ==========================================
# 4. DOCUMENT EXTRACTION
# ==========================================

def extract_text(file):

    documents = []

    if file.name.lower().endswith(".pdf"):

        pdf = PdfReader(
            BytesIO(file.getvalue())
        )

        if pdf.is_encrypted:
            raise ValueError(
                f"{file.name} is password protected."
            )

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            text = page.extract_text() or ""

            if text.strip():

                documents.append({

                    "text": text,

                    "source": file.name,

                    "page": page_number

                })

    elif file.name.lower().endswith(".txt"):

        text = file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

        if text.strip():

            documents.append({

                "text": text,

                "source": file.name,

                "page": None

            })

    return documents


# ==========================================
# 5. TEXT CHUNKING
# ==========================================

def create_chunks(
    documents,
    chunk_size=500,
    overlap=100
):

    chunks = []

    for document in documents:

        words = document["text"].split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append({

                    "text": chunk_text,

                    "source": document["source"],

                    "page": document["page"]

                })

            start += chunk_size - overlap

    return chunks


# ==========================================
# 6. GENERATE EMBEDDINGS
# ==========================================

def create_vector_database(chunks, model):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(

        texts,

        convert_to_numpy=True,

        normalize_embeddings=True,

        show_progress_bar=False

    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


# ==========================================
# 7. SEMANTIC RETRIEVAL
# ==========================================

def retrieve_documents(
    question,
    model,
    embeddings,
    chunks,
    top_k=4
):

    query_embedding = model.encode(

        [question],

        convert_to_numpy=True,

        normalize_embeddings=True

    )

    similarities = cosine_similarity(

        query_embedding,

        embeddings

    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({

            "text": chunks[idx]["text"],

            "source": chunks[idx]["source"],

            "page": chunks[idx]["page"],

            "score": float(similarities[idx])

        })

    return results


# ==========================================
# 8. GENERATE AI ANSWER
# ==========================================

def generate_answer(
    question,
    retrieved_chunks,
    chat_history
):

    if not API_KEY:

        return (
            "Groq API key is missing. "
            "Please configure your .env file."
        )

    if not retrieved_chunks:

        return (
            "No relevant information was found "
            "in your uploaded documents."
        )

    client = Groq(
        api_key=API_KEY
    )

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        page_info = (

            f"Page {chunk['page']}"

            if chunk["page"] is not None

            else "Page unavailable"

        )

        context += f"""

SOURCE {i}

Document: {chunk['source']}

{page_info}

Content:
{chunk['text']}

"""

    system_prompt = """
You are a document-based AI assistant.

Your job is to answer questions using the
information retrieved from uploaded documents.

RULES:

1. Use the provided document context as your
   factual source.

2. Do not invent facts or information.

3. If the answer is not available in the
   supplied context, clearly say that the
   uploaded documents do not contain enough
   information.

4. Give clear and understandable answers.

5. Use previous conversation messages to
   understand follow-up questions, but do not
   treat previous AI responses as factual evidence.

6. Never claim that information comes from a
   document unless it appears in the supplied context.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using the provided
document context.
"""

    history = []

    for message in chat_history[-6:]:

        history.append({

            "role": message["role"],

            "content": message["content"]

        })

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[

                {
                    "role": "system",
                    "content": system_prompt
                },

                *history,

                {
                    "role": "user",
                    "content": user_prompt
                }

            ],

            temperature=0.2,

            max_tokens=1000

        )

        return response.choices[0].message.content

    except Exception as e:

        return f"Unable to generate answer: {str(e)}"


# ==========================================
# 9. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Document Upload")

    uploaded_files = st.file_uploader(

        "Upload PDF or TXT documents",

        type=["pdf", "txt"],

        accept_multiple_files=True

    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            try:

                with st.spinner(
                    "Processing documents..."
                ):

                    model = load_embedding_model()

                    all_documents = []

                    for file in uploaded_files:

                        extracted = extract_text(file)

                        all_documents.extend(
                            extracted
                        )

                    if not all_documents:

                        st.error(
                            "No readable text found in documents."
                        )

                    else:

                        chunks = create_chunks(
                            all_documents
                        )

                        if not chunks:

                            st.error(
                                "No text chunks generated."
                            )

                        else:

                            embeddings = create_vector_database(
                                chunks,
                                model
                            )

                            st.session_state.chunks = chunks

                            st.session_state.embeddings = embeddings

                            st.session_state.processed = True

                            st.session_state.messages = []

                            st.success(
                                "Documents processed successfully!"
                            )

            except Exception as e:

                st.error(
                    f"Document processing error: {str(e)}"
                )

    st.divider()

    st.subheader("Document Information")

    if st.session_state.processed:

        st.success("Documents ready")

        st.write(
            f"Total chunks: {len(st.session_state.chunks)}"
        )

        st.write(
            f"Total embeddings: {len(st.session_state.embeddings)}"
        )

    else:

        st.info(
            "No documents processed yet."
        )

    st.divider()

    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# 10. MAIN CHAT INTERFACE
# ==========================================

st.subheader("Chat with your documents")

if not st.session_state.processed:

    st.info(
        "Upload PDF or TXT documents from the sidebar "
        "and click Process Documents to get started."
    )

else:

    # Display chat history

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if message["role"] == "assistant":

                with st.expander("View Sources"):

                    for source in message.get(
                        "sources",
                        []
                    ):

                        page = source["page"]

                        page_info = (

                            f"Page {page}"

                            if page is not None

                            else "Page unavailable"

                        )

                        st.write(
                            f"📄 {source['source']} - {page_info}"
                        )

    # User input

    question = st.chat_input(
        "Ask something about your documents..."
    )

    if question:

        st.session_state.messages.append({

            "role": "user",

            "content": question

        })

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                model = load_embedding_model()

                retrieved = retrieve_documents(

                    question,

                    model,

                    st.session_state.embeddings,

                    st.session_state.chunks,

                    top_k=4

                )

                answer = generate_answer(

                    question,

                    retrieved,

                    st.session_state.messages[:-1]

                )

                st.markdown(answer)

                with st.expander("View Sources"):

                    for source in retrieved:

                        page = source["page"]

                        page_info = (

                            f"Page {page}"

                            if page is not None

                            else "Page unavailable"

                        )

                        st.write(
                            f"📄 {source['source']} - {page_info}"
                        )

                        st.caption(
                            f"Similarity Score: {source['score']:.3f}"
                        )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": retrieved

                })