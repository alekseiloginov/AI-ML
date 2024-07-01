RAG - Evaluation (Alice in Wonderland)

Dataset is represented by Alice in Wonderland book from Project Gutenberg: https://www.gutenberg.org/ebooks/11.

Goals:
- Ingest the book into a Chroma vector database storing document embeddings.
- Apply RAG by pulling relevant context from our vector database and injecting it into the user prompt.
- Evaluate the RAG using DeepEval and RAGAs frameworks.
- Check for the following RAG metrics: Faithfulness, Answer Relevancy, Context Relevancy, Context Precision, Context Recall.
