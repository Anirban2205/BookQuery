import pandas as pd
import fitz
from tqdm import tqdm
import nltk
import re
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from RAPTOR import recursive_embed_cluster_summarize

def clean_text(text):
    text = text.replace('\n', ' ')  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)  # Remove non-alphanumeric characters
    return text


def sentence_aware_chunking(input_text, max_tokens):
    sentences = sent_tokenize(input_text)
    chunks = []
    current_chunk = []
    current_len = 0
    for sentence in sentences:
        num_tokens = len(nltk.word_tokenize(sentence))
        if (current_len + num_tokens) > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_len = num_tokens
        else:
            current_chunk.append(sentence)
            current_len += num_tokens
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def open_and_read_pdf(pdf_path: str, bookname: str) -> list[dict]:
    """
    Opens a PDF file, reads its text content page by page, and collects statistics.

    Parameters:
        pdf_path (str): The file path to the PDF document to be opened and read.

    Returns:
        list[dict]: A list of dictionaries, each containing the page number
        (adjusted), character count, word count, sentence count, token count, and the extracted text
        for each page.
    """
    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        text = clean_text(text)
        pages_and_texts.append({"book_name": bookname,
                                "page_number": page_number + 1,
                                "page_char_count": len(text),
                                "page_word_count": len(text.split(" ")),
                                "page_sentence_count_raw": len(text.split(". ")),
                                "text": text,
                                "chunks": sentence_aware_chunking(text, 120)})
    return pages_and_texts



if __name__ == "__main__":
    file_path = input("Enter the file path of the book: ")
    bookname = input("Enter the name of the book: ")
    pages_and_texts = open_and_read_pdf(pdf_path=file_path, bookname=bookname)

    leaf_chunks = []
    for page in pages_and_texts:
        leaf_chunks.extend(page['chunks'])
    results = recursive_embed_cluster_summarize(leaf_chunks, level=1, n_levels=4)

    summaries = []
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summary = results[level][1]["summaries"].tolist()
        # Extend the list of summaries with the summaries from the current level
        summaries.extend(summary)
    
    # Save all the generated summaries in a dataframe
    cluster_sum = {"summary": summaries}
    cluster_sum_df = pd.DataFrame(cluster_sum)
    cluster_sum_df.to_csv(f"./{bookname}_summaries.csv")

    text_chunks = leaf_chunks + summaries
    all_text_chunks = {"Text": text_chunks}
    df = pd.DataFrame(all_text_chunks) 
    df.to_csv(f"./{bookname}_text_chunks.csv")