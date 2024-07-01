import os
import shutil
import gemini
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_precision, context_recall
from ragas import evaluate

CHROMA_PATH = "chroma"

# 1. Create a vector database
# clear existing database
print("Clearing Database")
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# load the book from a file into a document
document_loader = DirectoryLoader("", glob="alice_in_wonderland.md")
doc = document_loader.load()

# split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(doc)

# print a random chunk
print("Random chunk:")
document = chunks[10]
print(document.page_content)
print(document.metadata)

# pick embeddings type to use in our vector database
def get_embedding():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Ollama needs to be installed for this option
    # embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")
    # embeddings = OpenAIEmbeddings()
    return embeddings

# create a vector database
db = Chroma.from_documents(chunks, get_embedding(), persist_directory=CHROMA_PATH)  # from new chunks
# db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding())  # from existing file location

# check what we have in vector database
existing_ids = db.get(include=[])
first_doc_id = existing_ids['ids'][0]
first_doc = db.get(first_doc_id)
print(f"First document in DB: {first_doc}")


# 2. Query model using RAG that is backed by our vector database
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # get top 5 documents related to the query
    relevant_chunks = db.similarity_search_with_score(query_text, k=5)
    print('Relevant chunks:')
    for chunk, score in relevant_chunks:
        print(f"Chunk score: {score}\nChunk content:\n{chunk.page_content}\n----------")

    # merge documents into one context
    context_text = "\n\n---\n\n".join([chunk.page_content for chunk, _score in relevant_chunks])

    # create a prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # query model
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    print('--- RESPONSE ---')
    print(response_text)
    return response_text, context_text


# 3. Evaluate RAG using DeepEval
# Create a DeepEval model that will be evaluating results
def get_model():
    ## Hugging Face models
    # model_name = "distilbert/distilgpt2"  # fast, but very simple
    # model_name = "mistralai/Mistral-7B-v0.1"  # smart, but slow and requires a HF access token
    # model = hugging_face_model.new_model(model_name)
    ## Gemini model
    model_name = "gemini-1.5-pro-001"
    model = gemini.new_model(model_name)
    return model

# Faithfulness
# - classifies each response claim as truthful / untruthful based on the retrieved context
def faithfulness_deepeval(query_text, context_text, response_text):
    test_case = LLMTestCase(
        input=query_text,
        actual_output=response_text,
        retrieval_context=[context_text]
    )
    metric = FaithfulnessMetric(threshold=0.5, model=get_model())
    metric.measure(test_case)
    print('Faithfulness - DeepEval:')
    print(metric.score)
    print(metric.reason)
    print(metric.is_successful())
    print('----------')

# Answer Relevancy
# - compares response to user prompt + retrieved context
def answer_relevancy_deepeval(query_text, context_text, response_text):
    test_case = LLMTestCase(
        input=query_text,
        actual_output=response_text,
        retrieval_context=[context_text]
    )
    metric = AnswerRelevancyMetric(threshold=0.5, model=get_model())
    metric.measure(test_case)
    print('Answer Relevancy - DeepEval:')
    print(metric.score)
    print(metric.reason)
    print(metric.is_successful())
    print('----------')

# Context Relevancy
# - classifies relevancy of statements made in the retrieval context to user query
def context_relevancy_deepeval(query_text, context_text, response_text):
    test_case = LLMTestCase(
        input=query_text,
        actual_output=response_text,
        retrieval_context=[context_text]
    )
    metric = ContextualRelevancyMetric(threshold=0.5, model=get_model())
    metric.measure(test_case)
    print('Context Relevancy - DeepEval:')
    print(metric.score)
    print(metric.reason)
    print(metric.is_successful())
    print('----------')

# Context Precision
# - same as Contextual Relevancy + checks that more relevant nodes (chunks of information) are ranked higher in retrieval context
def context_precision_deepeval(query_text, context_text, response_text, expected_response_text):
    test_case=LLMTestCase(
        input=query_text,
        actual_output=response_text,
        expected_output=expected_response_text, # "ideal" LLM output (extra param for contextual metrics)
        retrieval_context=[context_text]
    )
    metric = ContextualPrecisionMetric(threshold=0.5, model=get_model())
    metric.measure(test_case)
    print('Context Precision - DeepEval:')
    print(metric.score)
    print(metric.reason)
    print(metric.is_successful())
    print('----------')

# Context Recall
# - ability to retrieve all relevant information for a given input to generate a complete response
def context_recall_deepeval(query_text, context_text, response_text, expected_response_text):
    test_case=LLMTestCase(
        input=query_text,
        actual_output=response_text,
        expected_output=expected_response_text, # "ideal" LLM output (extra param for contextual metrics)
        retrieval_context=[context_text]
    )
    metric = ContextualRecallMetric(threshold=0.5, model=get_model())
    metric.measure(test_case)
    print('Context Recall - DeepEval:')
    print(metric.score)
    print(metric.reason)
    print(metric.is_successful())
    print('----------')


# 4. Evaluate RAG using RAGAs
def ragas_scores(query_text, context_text, response_text, expected_response_text):
    data_dict = {
        'question': [query_text],
        'answer': [response_text],
        'contexts' : [[context_text]],
        'ground_truth': [expected_response_text]
    }
    dataset = Dataset.from_dict(data_dict)
    scores = evaluate(dataset,
                     metrics=[faithfulness, answer_relevancy, context_relevancy, context_precision, context_recall],
                     llm=Ollama(model="mistral"),
                     embeddings=get_embedding()
                     )
    print('RAGAs scores:')
    df = scores.to_pandas().drop(columns=['contexts', 'ground_truth'])
    print(df.to_string(index=False, max_rows=None, max_cols=None))  # hide row indexes, include all data


# Sample RAG variables that can be used for a quick test
query_text = "What's the name of the book?"
context_text = """
Title: Alice's Adventures in Wonderland

Author: Lewis Carroll

Release date: June 27, 2008 [eBook #11]
Most recently updated: March 30, 2021

Language: English

Credits: Arthur DiBianca and David Widger

START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND 
[Illustration]
----------
The Project Gutenberg eBook of Alice's Adventures in Wonderland
----------
“Now tell me, Pat, what’s that in the window?”

“Sure, it’s an arm, yer honour!” (He pronounced it “arrum.”)

“An arm, you goose! Who ever saw one that size? Why, it fills the whole
window!”

“Sure, it does, yer honour: but it’s an arm for all that.”
----------
on which the phrase “Project Gutenberg” appears, or with which the
phrase “Project Gutenberg” is associated) is accessed, displayed,
performed, viewed, copied or distributed:
----------
conversations in it, “and what is the use of a book,” thought Alice
“without pictures or conversations?”
"""
response_text = """ The name of the book is "Alice's Adventures in Wonderland." """
expected_response_text = "Alice in Wonderland."

# Interactive "prompt - response - eval" environment
while True:
    query_text = input("Enter your query (or type 'exit' to quit): ")  # eg: What's the name of the book?
    if query_text.lower() == "exit": break

    response_text, context_text = query_rag(query_text)

    # run evals
    print('--- RUNNING EVALS ---')
    # DeepEval
    faithfulness_deepeval(query_text, context_text, response_text)
    answer_relevancy_deepeval(query_text, context_text, response_text)
    context_relevancy_deepeval(query_text, context_text, response_text)
    context_precision_deepeval(query_text, context_text, response_text, expected_response_text)
    context_recall_deepeval(query_text, context_text, response_text, expected_response_text)
    # RAGAs
    ragas_scores(query_text, context_text, response_text, expected_response_text)
    print("\n--- ALL DONE ---\n")
