from sentence_transformers import SentenceTransformer, util
from index_generator import open_and_read_pdf
from vectorstore_creator import vectorstore_pipeline
from tqdm import tqdm
import pandas as pd
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import fitz
from PIL import Image
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads


from dotenv import load_dotenv
load_dotenv()
import os
hf_key = os.getenv("HUGGINGFACE_API_KEY")
id = "mistralai/Mistral-7B-Instruct-v0.2"
id2 = "huggingfaceh4/zephyr-7b-alpha"
def load_language_model(model_id=id2, hf_token=hf_key):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
    llm = HuggingFaceEndpoint(
        repo_id=id,
        temperature= 0.1)
    return llm

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import BM25Retriever, EnsembleRetriever

class LineList(BaseModel):
    lines: list[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


def load_hybrid_retriever():
    vector_db, book_chunks, file_path = vectorstore_pipeline()
    st.session_state.file_path = file_path
    bm25_retriever = BM25Retriever.from_documents(book_chunks)
    milvus_retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    ens_ret = EnsembleRetriever(retrievers=[bm25_retriever, milvus_retriever], weights=[0.5, 0.5])
    return ens_ret

def load_query_prompt():
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question.
    Provide these alternative questions separated by newlines.
    Original question: {question}"""
    query_prompt = ChatPromptTemplate.from_template(template=template)
    return query_prompt


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results[:5]


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def reorder_documents(docs):
    reordering = LongContextReorder()
    new_docs = [doc[0] for doc in docs]
    reordered_docs = reordering.transform_documents(new_docs)
    return reordered_docs

def load_rag_prompt():
    from langchain import hub
    prompt = hub.pull("rlm/rag-prompt")
    return prompt


def output_parsing(text):
    output_parser = LineListOutputParser()
    op_lines = output_parser.parse(text)
    return op_lines.lines

def load_utilities():
    llm = load_language_model()
    retriever = load_hybrid_retriever()
    return llm, retriever


@st.cache_resource
def load_rag_chain():
    llm, retriever = load_utilities()
    QUERY_PROMPT = load_query_prompt()
    RAG_PROMPT = load_rag_prompt()
    query_chain = QUERY_PROMPT | llm | StrOutputParser() | output_parsing
    retriver_chain = query_chain | retriever.map() | reciprocal_rank_fusion
    rag_chain = (
        {"context": retriver_chain | reorder_documents | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriver_chain


def home_page():
    st.set_page_config(page_title="Look Up")
    st.header("Query your Book")
    st.sidebar.title("Pages to look for: -")

def button_config():
    space = st.sidebar.container(height=400)
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
    return space

def render_image(book_name, page_no, space):
    file_path = st.session_state.file_path
    doc = fitz.open(file_path)
    space.markdown(f"## {book_name}")
    for nos in page_no:
        page = doc.load_page(nos - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        with space.chat_message("assistant"):
            space.markdown(f"### Page Number: {nos - 1}")
        space.image(image)



def site():
    home_page()
    space = button_config()
    rag_chain, retriver_chain = load_rag_chain()

    if user_input := st.chat_input("Enter your query.."):
        st.session_state.messages.append(("User", user_input))
        with st.spinner("Bot is searching...."):
            answer = rag_chain.invoke(user_input)
            docs = retriver_chain.invoke(user_input)
            page_no = [doc[0].metadata['page_number'] for doc in docs][:2]
            book_name = [doc[0].metadata['book_name'] for doc in docs][0]
            render_image(book_name, page_no, space)
        st.session_state.messages.append(("Bot", answer))

    messages = st.session_state.get("messages", [])
    for role, message in messages:
        if role == "Bot":
            with st.chat_message('assistant'):
                st.markdown(message)
        elif role == "User":
            with st.chat_message("user"):
                st.markdown(message)

if __name__ == "__main__":
    site()