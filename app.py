
import os
import tempfile

import streamlit as stl
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import InMemoryVectorStore


# ---------------- Page Configuration ----------------

stl.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="centered",
)


# ---------------- API Configuration ----------------

load_dotenv()


def get_api_key(key_name):
    """Read API keys from Streamlit Secrets or the local .env file."""
    try:
        key = stl.secrets.get(key_name)
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass

    return os.getenv(key_name)


GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    stl.error(
        "Gemini API key not found. Add GOOGLE_API_KEY to your "
        "local .env file or Streamlit Cloud Secrets."
    )
    stl.stop()


# ---------------- AI Models ----------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=GOOGLE_API_KEY,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=GOOGLE_API_KEY,
)


# ---------------- Session State ----------------

if "messages" not in stl.session_state:
    stl.session_state.messages = []

if "vector_db" not in stl.session_state:
    stl.session_state.vector_db = None

if "document_uploaded" not in stl.session_state:
    stl.session_state.document_uploaded = False

if "document_name" not in stl.session_state:
    stl.session_state.document_name = None


# ---------------- PDF Processing ----------------

def document_process(uploaded_file):
    """Load, split, embed, and index an uploaded PDF."""

    file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            file_path = temp_file.name

        # Load PDF pages.
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "The uploaded PDF contains no readable pages."
            )

        # Split document text into chunks.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        docs = text_splitter.split_documents(documents)

        if not docs:
            raise ValueError(
                "No readable text was found in this PDF. "
                "A scanned PDF may require OCR."
            )

        # Create the in-memory vector database.
        vector_db = InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
        )

        # Update session state only after successful processing.
        stl.session_state.vector_db = vector_db
        stl.session_state.document_uploaded = True
        stl.session_state.document_name = uploaded_file.name
        stl.session_state.messages = []

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# ---------------- Application UI ----------------

stl.title("📚 DocuMind AI")

stl.caption(
    "Your intelligent PDF assistant — upload, ask, and explore."
)


# ---------------- PDF Upload ----------------

stl.subheader("Upload your document")

uploaded_file = stl.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    key="pdf_uploader",
)


if uploaded_file is not None:
    is_new_document = (
        not stl.session_state.document_uploaded
        or stl.session_state.document_name != uploaded_file.name
    )

    if is_new_document:
        try:
            with stl.spinner(
                "Reading PDF and creating embeddings..."
            ):
                document_process(uploaded_file)

            stl.success(
                f"Successfully processed: {uploaded_file.name}"
            )

        except Exception as e:
            stl.error(f"Could not process the PDF: {e}")


# ---------------- Current Document ----------------

if stl.session_state.document_uploaded:
    stl.caption(
        f"Current document: {stl.session_state.document_name}"
    )

    if stl.button("Upload another PDF"):
        stl.session_state.vector_db = None
        stl.session_state.document_uploaded = False
        stl.session_state.document_name = None
        stl.session_state.messages = []

        stl.rerun()


# ---------------- Chat Interface ----------------

if (
    stl.session_state.document_uploaded
    and stl.session_state.vector_db is not None
):
    stl.divider()
    stl.subheader("💬 Chat with your PDF")

    # Display all previous messages on every rerun.
    for message in stl.session_state.messages:
        with stl.chat_message(message["role"]):
            stl.markdown(message["content"])

    # Input stays at the bottom of the chat interface.
    query = stl.chat_input(
        "Ask anything about your document..."
    )

    if query:
        # Save and display the user's question.
        stl.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with stl.chat_message("user"):
            stl.markdown(query)

        try:
            with stl.chat_message("assistant"):
                with stl.spinner(
                    "Searching document and generating answer..."
                ):

                    # Retrieve the most relevant PDF chunks.
                    docs = (
                        stl.session_state.vector_db
                        .similarity_search(query, k=3)
                    )

                    # Combine retrieved text into context.
                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    # Create the prompt.
                    prompt = f"""
You are DocuMind AI, a helpful PDF question-answering assistant.

Answer the user's question using the provided PDF context.
If the answer is not available in the context, clearly explain
that the document does not contain enough information.
Do not invent facts.

PDF Context:
{context}

User Question:
{query}

Answer:
"""

                    # Invoke Gemini only once.
                    response = llm.invoke(prompt)

                    # Extract readable text from the model response.
                    content = response.content

                    if isinstance(content, list):
                        response_text = "\n".join(
                            item["text"]
                            for item in content
                            if isinstance(item, dict)
                            and item.get("type") == "text"
                            and item.get("text")
                        )
                    else:
                        response_text = str(content)

                    # Handle an empty response safely.
                    if not response_text.strip():
                        response_text = (
                            "Sorry, I couldn't generate a text answer. "
                            "Please try asking another question."
                        )

                    # Display the clean answer.
                    stl.markdown(response_text)

                    # Save the answer for future reruns.
                    stl.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response_text,
                        }
                    )

        except Exception as e:
            stl.error(
                f"Failed to generate the answer: {e}"
            )
