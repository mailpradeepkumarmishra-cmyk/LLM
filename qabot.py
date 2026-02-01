#source my_env/bin/activate # activate my_env


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

# =========================
# Suppress warnings / telemetry noise
# =========================

def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

# =========================
# LLM
# =========================

def get_llm():
    model_id = "ibm/granite-3-2-8b-instruct"
    parameters = {
        GenParams.TEMPERATURE: 0.5,
        GenParams.MAX_NEW_TOKENS: 256,
    }

    llm = WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=parameters,
    )
    return llm


# =========================
# Document Loader
# =========================

def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


# =========================
# Text Splitter
# =========================

def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    return chunks


# =========================
# Embeddings
# =========================

def get_embeddings():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512
    }

    embeddings = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    return embeddings


# =========================
# Vector Database (CRITICAL FIX)
# =========================

def vector_database(chunks):
    embeddings = get_embeddings()

    # 🔴 ABSOLUTE FIX: filter by content length
    safe_chunks = [
        c for c in chunks
        if c.page_content
        and len(c.page_content.strip()) > 80
    ]

    vectordb = Chroma.from_documents(
        safe_chunks,
        embeddings,
        persist_directory=None  # avoid sqlite issues
    )
    return vectordb


# =========================
# Retriever
# =========================

def retriever(file_path):
    documents = document_loader(file_path)
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


# =========================
# QA Chain
# =========================

def retriever_qa(file_path, query):
    llm = get_llm()
    retriever_obj = retriever(file_path)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )

    result = qa.invoke(query)
    return result["result"]


# =========================
# Gradio Interface
# =========================

rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_types=[".pdf"],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Ask a question about the PDF..."
        )
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering Bot",
    description="Upload a PDF and ask questions. The bot answers using document context."
)

# =========================
# Launch App
# =========================

rag_application.launch(
    server_name="0.0.0.0",
    server_port=7860
)



## Retriever  ✅ REQUIRED BY LAB
def retriever(file):
    documents = document_loader(file)
    chunks = text_splitter(documents)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )

    response = qa.invoke(query)
    return response["result"]


# Gradio Interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2)
    ],
    outputs=gr.Textbox(label="Answer"),
    title="PDF Question Answering Bot",
    description="Upload a PDF document and ask any question. The chatbot will answer using the document."
)

# Launch app
rag_application.launch(server_name="0.0.0.0", server_port=7860)
