import os
from dotenv import load_dotenv
load_dotenv()

from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =========================
# LOCAL LLM (FLAN-T5)
# =========================
model_path = "google/flan-t5-base"  # local cache OR downloaded folder

llm_pipeline = pipeline(
    "text-generation",
    model=model_path,
    max_new_tokens=512,
    temperature=0.5
)

def call_llm(prompt: str) -> str:
    result = llm_pipeline(prompt)
    return result[0]["generated_text"]

# =========================
# Custom Prompt
# =========================
custom_prompt = """Use the information provided in the context to answer the question.

If you don't know, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "question"]
    )

# =========================
# Load Vector DB
# =========================
DB_FAISS_PATH = "vector_store/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# =========================
# Format docs
# =========================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# =========================
# Prompt
# =========================
prompt = set_custom_prompt()

# =========================
# RAG PIPELINE
# =========================
def rag_pipeline(question: str):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    answer = call_llm(final_prompt)

    return {
        "answer": answer,
        "source_documents": docs
    }

# =========================
# RUN
# =========================
user_question = input("Ask a question: ")

response = rag_pipeline(user_question)

print("\nAnswer:\n", response["answer"])
print("\nSources:\n", response["source_documents"])