import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader , DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DATA_DIR=Path("data_1")
CHROMA_DIR="chroma_db_1"
COLLECTION_NAME=""
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
TOP_K=3
SUPPORTED_SUFFIXS={".pdf",".txt",".docx"}

api_key=os.getenv("GEMINI_API_KEY")

def normalize(query):
    if query is None:
        return  ""
    clean=str(query).strip()
    clean=" ".join(clean.split())
    return clean

def file_discovery(data_dir:Path):
    files=[]
    for file_path in data_dir.iterdir():
        if file_path.suffix.lower() in SUPPORTED_SUFFIXS:
            files.append(file_path)
    files.sort()
    return files

def load_documents(file_paths):
    documnets=[]

    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.pdf",loader_cls=PyPDFLoader).load()
    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"}).load()
    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.docx",loader_cls=Docx2txtLoader).load()

    for document in documnets:
        document.metadata["source_file"]=Path(
            document.metadata["source"]
        ).name
        document.metadata["source_file"]=Path(
            document.metadata["source"]
        ).suffix.replace(".","")
    return documnets

def chunking(documents):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunk=text_spliter.split_documents(documents)
    return chunk


embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def vectorization(chunks=None):
    if Path(CHROMA_DIR).exists():
        vector_db=Chroma(embedding_function=embedding,persist_directory=CHROMA_DIR)
    else:
        vector_db=Chroma.from_documents(documents=chunks,persist_directory=CHROMA_DIR,embedding=embedding)
    return vector_db

def create_retriver(vector_db):
    retriver=vector_db.as_retriever(search_kwargs={"k":TOP_K})
    return retriver

def format_context(documents):
    context="\n\n".join("[Source: "+ i.metadata.get("source_file","unknown")+"]\n"+i.page_content for i in documents)
    return context

@tool
def search_banking_documents(query):
    """Search the supplied documents for account rules, card safety, payment disputes,
    ATM procedures, beneficiary activation, branch services, KYC, nominee updates,
    complaint handling evidence. Use this for a normal banking question."""
     
    clean_query=normalize(query)
    if clean_query=="":
        return "Please provide query"
    vector_db=vectorization()
    retriver=create_retriver(vector_db)
    context=retriver.invoke(clean_query)
    return context

@tool
def list_banking_sources():

    """List the supplied knowledge filenames."""
    files=file_discovery(DATA_DIR)
    name=[f.name for f in files]
    return "\n".join(name)

@tool
def get_banking_rule_excerpt(keywords,source_name):

    
    """Return relevant evidence from one requested source document."""
      
    vector_db=vectorization()
    retriver=create_retriver(vector_db)
    docs=retriver.invoke(keywords)

    matched=[]
    for doc in docs:
        source=doc.metadata.get("source_file","")
        if source.lower()==source_name.lower():
            matched.append(doc)
    return format_context(matched)

@tool
def prepare_banking_case(reason,evidance):
    """Create a deterministic hand-off note for the banking operations or fraud-review team."""
    return f"BANKING CASE | Reason: "+reason +"| Evidance: "+evidance


def build_agent():
    if not api_key:
        raise Exception("API key not found")
    model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0,api_key=api_key)
    tools=[
        search_banking_documents,
        list_banking_sources,
        get_banking_rule_excerpt,
        prepare_banking_case
    ]
    prompt="""You are a banking support assistant for Northbridge Community Bank.
        Answer only from the supplied documents. 
        Always call search_banking_documents before answering a normal question.
        If the user names a file, use get_banking_rule_excerpt.
        If the user asks which files are loaded, use list_banking_sources.
        For unauthorised transactions, ATM cash differences, duplicate debits,
        cash-deposit differences, KYC mismatches or suspected phishing, get evidence
        and then call prepare_banking_case.
        You cannot reverse transactions, approve refunds, confirm fraud, change KYC,
        block cards, or promise money will be credited
        Always mention the exact source filename and keep documented numbers exact.
        If evidence is not enough, say the supplied documents do not answer the question."""
    agent=create_agent(model=model,tools=tools,system_prompt=prompt)
    return agent

def main():
    files=file_discovery(DATA_DIR)
    doc=load_documents(files)
    chunks=chunking(doc)
    vectorization(chunks)
    while True:
        question=input("Ask question: ")
        if question.lower() in ["exit","end"]:
            break
        agent=build_agent()
        result=agent.invoke({
            "messages":[{
                "role":"user",
                "content":question
            }]
        },config={"recursion_limit":20}
        )
        print(result["messages"][-1].content[0]["text"])

if __name__=="__main__":
    main()