# =========================
# Imports
# =========================

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

# =========================
# LLM
# =========================

def get_llm():
    return WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params={
            GenParams.TEMPERATURE: 0.5,
            GenParams.MAX_NEW_TOKENS: 256,
        },
    )

# =========================
# Embeddings (DO NOT CALL embed_documents)
# =========================

def get_embeddings():
    return WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params={
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512
        },
    )

# =========================
# Vector DB (SAFE + LAB-COMPATIBLE)
# =========================

def build_vectordb(pdf_path):
    # Load PDF
    docs = PyPDFLoader(pdf_path).load()

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    # Clean text
    texts = [
        c.page_content.strip()
        for c in chunks
        if c.page_content and len(c.page_content.strip()) > 80
    ]

    # IMPORTANT: let Chroma call embeddings internally
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=get_embeddings(),
        persist_directory=None
    )

    return vectordb

# =========================
# QA Bot
# =========================

def qa_bot(pdf_path, query):
    vectordb = build_vectordb(pdf_path)
    retriever = vectordb.as_retriever()
    llm = get_llm()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    return qa.invoke(query)["result"]

# =========================
# Gradio UI
# =========================

app = gr.Interface(
    fn=qa_bot,
    inputs=[
        gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"]),
        gr.Textbox(label="Question", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering Bot",
    description="Upload a PDF and ask questions using RAG."
)

# =========================
# Launch
# =========================

app.launch(server_name="0.0.0.0", server_port=7860)
