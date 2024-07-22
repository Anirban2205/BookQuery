from sentence_transformers import SentenceTransformer, util
from index_generator import open_and_read_pdf
from tqdm import tqdm
import torch
import pandas as pd
from langchain.docstore.document import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

def add_embedding_to_texts(pages_and_texts, embedding_model):
    for item in tqdm(pages_and_texts):
        item["embedding"] = embedding_model.encode(item["text"])
    return pages_and_texts

def generate_summary_pages(summaries, embedding_model, pages_and_texts):
    """"Generate embedding for the summaries and compare them to the text embeddings of the book.
    Then based on it assign page number and book name to the summary"""
    summary_pages = []
    for summary in tqdm(summaries):
        dot_scores = []
        summary_embedding = embedding_model.encode(summary)
        for item in pages_and_texts:
            scores = util.cos_sim(summary_embedding, item["embedding"])
            dot_scores.append(scores)
        dot_scores = torch.tensor(dot_scores)
        top_results_dot_product = torch.topk(dot_scores, k=5)
        idx = top_results_dot_product.indices.numpy()[0]
        page_number = pages_and_texts[idx]['page_number']
        book_name = pages_and_texts[idx]['book_name']
        summary_pages.append({"book_name": book_name,"page_number": page_number, "chunks": [summary]})
    return summary_pages


def create_book_documents(pages_and_texts):
    book_docs = []
    for page in pages_and_texts:
        for ch in page['chunks']:
            doc = Document(page_content=ch, metadata={"book_name": page['book_name'],"page_number": page['page_number']})
            book_docs.append(doc)
    return book_docs


def create_vectorstore(book_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    uri = "book_store.db"

    vectorstore = Milvus.from_documents(
        documents=book_docs,
        embedding=embeddings,
        connection_args={
            "uri": uri
        },
        drop_old=True,  # Drop the old Milvus collection if it exists
    )
    return vectorstore, uri

def vectorstore_pipeline():
    file_path = input("Enter the path of the book: ")
    bookname = input("Enter the name of the book: ")
    pages_and_texts = open_and_read_pdf(file_path, bookname)
    embedding_model = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")
    
    pages_and_texts = add_embedding_to_texts(pages_and_texts, embedding_model)
    # loaad the summaries generated from RAPTOR
    summaries = pd.read_csv(f"./{bookname}_summaries.csv")
    summary_pages = generate_summary_pages(summaries, embedding_model, pages_and_texts)
    pages_and_texts.extend(summary_pages)
    book_docs = create_book_documents(pages_and_texts)
    vectorstore, uri = create_vectorstore(book_docs)
    return vectorstore, book_docs, file_path


if __name__ == "__main__":
    vectorstore_pipeline()
