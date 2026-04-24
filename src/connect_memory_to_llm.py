import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Step 1: Setup LLM (Local FLAN-T5)
LOCAL_MODEL_PATH = "google/flan-t5-base"

def load_llm(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def call_llm(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    return call_llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context
Context: {context}
Question: {question}
Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vector_store/faiss_index"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_kwargs={'k': 3})

# Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Pipeline
def rag_pipeline(question: str):
    llm = load_llm(LOCAL_MODEL_PATH)
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)

    docs = retriever.invoke(question)
    context = format_docs(docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    answer = llm(final_prompt)

    return {
        "result": answer,
        "source_documents": docs
    }

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = rag_pipeline(user_query)
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])