# Long Context RAG

This is a RAG application for long context content like Novels which are more than 300 pages.  
It uses RAPTOR to create clusters of chunks and summarize them repeatedly until a single chunk of text is formed.

## Project Files

- **RAPTOR.py:** It contains the code for RAPTOR indexing; the clustering and summarization task is executed which may take some time. The base code is taken from the repo of the publisher of the paper: [link](https://github.com/parthsarthi03/raptor/tree/master) and the langchain implementation of RAPTOR: [link](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb)
- **index_generator.py:** It uses the RAPTOR.py file and creates clusters and then save the summaries generated in a pandas dataframe along with the other chunks of text extracted from the book.
- **vectorstore_creator.py:** It creates the vector-store based on the summaries and text chunks generated from the index_generator.py file.
- **application.py:** It runs a streamlit application which works like a chatbot to interact with the book.

## Steps to Execute:-

1. Download the necessary libraries using pip.

```
pip install requirements.txt
```

2. Create a **.env** file to store your API keys(here I used the Huggingface API key).
3. Execute the **index_generator.py** file which will ask for the pdf file's path and the book's name.

```
python index_generator.py
Enter the file path of the book: <Enter the path of your pdf>
Enter the name of the book: <Enter the name of your book>
```

4. Now execute the **application.py** file which will again ask for the pdf file's path and the book's name.

```
python application.py
Enter the file path of the book: <Enter the path of your pdf>
Enter the name of the book: <Enter the name of your book>
```

It may take some time to load initially but once it loads everything it will launch a streamlit application onto your browser and then you can ask your queries.

### Cons:

- It is currently programmed for only one book but can be scaled easily.
- During the running of index_generator.py file the token limit of the API key might exceed if the book is longer. In that case, more than one api key can be used.
- At the moment it creates the vector database locally but can be hosted on the cloud.
