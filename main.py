
import uuid
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()



EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "vector_db"

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

model = ChatGroq(
    model='llama-3.3-70b-versatile',
    temperature=0.3
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional PDF Assistant.
Use Context and Chat History to answer.

Rules:      
    1. First search for the answer in the provided PDF context.
    2. If the answer is available in the PDF context, answer using the PDF only.
    3. If the answer is from the PDF, mention the source page number if available.
    4. If the answer is NOT available in the PDF context, then answer using your general AI knowledge.
    5. When answering from general AI knowledge, clearly start your response with:

    [AI Generated Answer - Not Found In PDF]

    6. Never pretend that AI-generated information came from the PDF.
    7. Use chat history when helpful
     

Context:
{context}

Chat History:
{history}"""),
    ("human", "{question}")
])


def process_pdf(file_path):

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)


    collection_id = str(uuid.uuid4())

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"{PERSIST_DIR}/{collection_id}",
        collection_name=collection_id
    )

    return vector_db, chunks, collection_id


def build_retriever(vector_db, chunks):

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 10

    vector_retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    hybrid = EnsembleRetriever(
        retrievers=[
            bm25,
            vector_retriever
        ],
        weights=[
            0.4,
            0.6
        ]
    )

    return hybrid


import os

def main():

    print("🚀 RAG PDF Chat (CLI Version)")

    pdf_path = input("Enter PDF path: ").strip()
    pdf_path = pdf_path.strip('"').strip("'")

    if not os.path.exists(pdf_path):
        print("❌ File not found!")
        return

    print("🔍 Processing PDF...")

    vector_db, chunks, collection_id = process_pdf(
        pdf_path
    )

    retriever = build_retriever(
        vector_db,
        chunks
    )

    print("✅ Ready! Type 'exit' to quit.\n")

    chat_history = []

    while True:

        user_query = input("👤 You: ").strip()

        if user_query.lower() == "exit":
            break

        try:

            docs = retriever.invoke(user_query)

            context_text = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            formatted_history = "\n".join(
                chat_history[-6:]
            )

            chain = QA_PROMPT | model

            response = chain.invoke(
                {
                    "context": context_text,
                    "history": formatted_history,
                    "question": user_query
                }
            )

            answer = response.content

            print(f"\n🤖 AI: {answer}\n")

            chat_history.append(
                f"Human: {user_query}"
            )

            chat_history.append(
                f"AI: {answer}"
            )

        except Exception as e:

            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()




