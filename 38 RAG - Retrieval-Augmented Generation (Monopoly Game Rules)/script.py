import os
import shutil
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "chroma"

# 1. Create a vector database
print("Clearing Database")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# load PDF documents
document_loader = PyPDFDirectoryLoader("")
documents = document_loader.load()

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(documents)

# Replace chunk IDs with the following format: "data/monopoly.pdf:page1:chunk2"
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:page{page}"

        # increment chunk index if  page id is the same as the last one
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # calculate the chunk id
        chunk_id = f"{current_page_id}:chunk{current_chunk_index}"
        last_page_id = current_page_id

        # replace chunk id
        chunk.metadata["id"] = chunk_id

    return chunks

# update chunk IDs
new_chunks = calculate_chunk_ids(chunks)
print(f"Adding new documents: {len(new_chunks)}")
new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

# pick embedding type to use in our vector database
def get_embedding():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    # embeddings = OpenAIEmbeddings()
    return embeddings

# add chunks to vector database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding())
db.add_documents(new_chunks, ids=new_chunk_ids)

# check what we have in vector database
existing_ids = db.get(include=[])
first_doc_id = existing_ids['ids'][0]
first_doc = db.get(first_doc_id)
print(f"First document in DB: {first_doc}")


# 2. Query model using RAG backed by our vector database

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # get top 5 documents related to the query
    relevant_chunks = db.similarity_search_with_score(query_text, k=5)
    # print('relevant_chunks')
    # print(relevant_chunks)

    # merge documents into one context
    context_text = "\n\n---\n\n".join([chunk.page_content for chunk, _score in relevant_chunks])

    # create a prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # query model
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # get ids of the top 5 documents to show as references
    chunk_ids = [chunk.metadata.get("id") for chunk, _score in relevant_chunks]

    response_with_references = f"Response: {response_text}\n\nReferences: {chunk_ids}"
    print(response_with_references)

while True:
    query_text = input("Enter your query (or type 'exit' to quit): ")
    if query_text.lower() == "exit":
        break
    query_rag(query_text)
    print("\n---\n")
