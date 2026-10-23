
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import InMemoryVectorStore
import streamlit as stl
from time import sleep


# LLM Generation

llm = ChatGoogleGenerativeAI(
model="gemini-3.1-flash-lite"
)



# Step1 ---- >> Documents Loading

if "vector_db" not in stl.session_state:
    stl.session_state.vector_db = None

if "messages" not in stl.session_state:
    stl.session_state.messages = []

def document_process(file_path):
    loader = PyPDFLoader(file_path)
    document = loader.load()

    # Step2 ---- >> Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )
    docs = text_splitter.split_documents(document)

    # Step3 ---- >> Embeddings Generation
    embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    )

    vectore_db = InMemoryVectorStore.from_documents(
        documents=docs, embedding=embeddings
        )

    stl.session_state.vector_db = vectore_db
    stl.session_state.document_uploaded = True



# answer = llm.invoke()
# print(answer.text)

stl.title("Document AI Assistent-- Ask anything")

if "document_uploaded" not in stl.session_state:
    stl.session_state.document_uploaded = False

# ## Document Uploaded
if not stl.session_state.document_uploaded:
    file = stl.file_uploader("Upload a PDF document", type=["pdf"], key="file_uploader")
    if file:
        with open("uploaded_document.pdf", "wb") as f:
            f.write(file.getvalue())
        # stl.markdown("Document uploaded successfully!")

        with stl.spinner("Processing the document..."):
            file_path = "uploaded_document.pdf"
            document_process("uploaded_document.pdf")

        with stl.spinner("Document processed successfully!"):
            sleep(2)
        stl.rerun()

if stl.session_state.document_uploaded and stl.session_state.vector_db:
    query = stl.chat_input("Ask a question about the document:")

    stl.session_state.messages = []
    if query:
        stl.session_state.messages.append({"role": "user", "content": query})

        stl.chat_message("user").markdown(query)

        with stl.spinner("Generating answer..."):
            docs = stl.session_state.vector_db.similarity_search(query)

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            prompt = f"""
            Answer the question using the context below.

            Context:
            {context}

            Question:
            {query}
            """

            answer = llm.invoke(prompt)
            stl.session_state.messages.append({"role": "assistant", "content": answer.text})
            stl.chat_message("assistant").markdown(answer.text)
