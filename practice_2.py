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

DATA_DIR=Path("data_2")
CHROMA_DIR="chroma_db_2"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
TOP_K=3
SUPPORTED_SUFFIX=[".pdf",".txt",".docx"]

api_key=os.getenv("GEMINI_API_KEY")

def normalize_text(value):
    if not value:
        return ""
    text=value.strip()
    text=" ".join(text.split())
    return text

def discover_supported_files(data_dir):
    if not data_dir:
        raise Exception("Data folder not found")
    included_list=[]
    for file in data_dir.iterdir():
        if file.suffix.lower() in SUPPORTED_SUFFIX:
            included_list.append(file)
    included_list.sort()
    return included_list

def load_documents(file_paths):
    documents=[]
    documents+=DirectoryLoader(DATA_DIR,glob="**/*.pdf",loader_cls=PyPDFLoader).load()
    documents+=DirectoryLoader(DATA_DIR,glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"}).load()
    documents+=DirectoryLoader(DATA_DIR,glob="**/*.docx",loader_cls=Docx2txtLoader).load()

    for doc in documents:
        doc.metadata["source_file"]=Path(
            doc.metadata["source"]
        ).name

        doc.metadata["source_type"]=Path(
            doc.metadata["source"]
        ).suffix.replace(".","")
    return documents

def split_document(documents):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunk=text_spliter.split_documents(documents)
    return chunk

embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_vector_store(chunks=None):
    if Path(CHROMA_DIR).exists():
        vector_db=Chroma(embedding_function=embedding,persist_directory=CHROMA_DIR)
    else:
        vector_db=Chroma.from_documents(documents=chunks,embedding=embedding,persist_directory=CHROMA_DIR)
    return vector_db

def create_retirver(vector_store):
    retriver=vector_store.as_retriever(search_kwargs={"k":TOP_K})
    return retriver

def format_context(documents):
    context="\n\n".join("[Source: "+i.metadata.get("source_file","unkown") +"]\n"+i.page_content for i in documents)
    return context

@tool
def search_travel_documents(query):
    """Validate the query, retrieve top evidence for flight booking holds, name corrections, ticket changes,
cancellation timing, hotel rules, package deposits, baggage reporting, schedule disruption, special assistance, 
format it, and return evidence only."""
    clean_query=normalize_text(query)
    vector_db=build_vector_store()
    retriver=create_retirver(vector_db)
    retrived_doc=retriver.invoke(clean_query)
    context=format_context(retrived_doc)

    return context

@tool
def list_travel_sources():
    """Return the three supplied filenames only, one per line or another stable plain-text format."""
    files=discover_supported_files(DATA_DIR)
    f=[i.name for i in files]
    return "\n".join(f)

@tool
def get_travel_rule_excerpt(keyword,source):
    """
Validate both inputs, retrieve by keyword, filter to the named file case-insensitively, and return matching-source evidence only.
"""
    vector_db=build_vector_store()
    retriver=create_retirver(vector_db)
    doc=retriver.invoke(keyword)

    men=[]
    for i in doc:
        source_file=i.metadata.get("source_file","")
        if source_file.lower()==source.lower():
            men.append(i)

    return format_context(men)
@tool
def prepare_travel_case(reason,evidance_source):
    """Return exactly TRAVEL CASE | Reason: <reason> | Evidence: <source>, with no extra commentary."""
    return f"TRAVEL CASE | Reason: {reason} | Evidance: {evidance_source}"

def build_agent():
    model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0,api_key=api_key)
    tools=[
        search_travel_documents,
        list_travel_sources,
        get_travel_rule_excerpt,
        prepare_travel_case
    ]
    prompt="""
The search and excerpt tools return evidence. The list tool returns deterministic filenames. The hand-off tool returns a deterministic one-line note. Gemini may then write a concise final response, but it must preserve the source name, exact documented values, and human-review limitation.
•	Do not call all four tools for every question. The agent should choose the smallest correct route.
•	A normal answer explains retrieved evidence. A hand-off answer explains what the documents say and clearly states that a human team must decide or act.
•	The final answer may be conversational, but deterministic tool outputs must remain exactly testable.
. When the user asks which files or policies are loaded, call list_travel_sources.

1. When the user asks which files or policies are loaded, call list_travel_sources.
2. When the user explicitly names one source file, call get_travel_rule_excerpt with both the topic keyword and filename.
3. For a normal question about flight booking holds, name corrections, ticket changes, cancellation timing, hotel rules, package deposits, baggage reporting, schedule disruption, special assistance, call search_travel_documents before writing the answer.
4. For airline cancellations, missed connections, denied boarding, duplicate charges, significant name mismatches, baggage loss, special-assistance failures, supplier record conflicts, or any decision controlled by an airline, hotel, insurer, or visa authority, retrieve supporting evidence when possible and then call prepare_travel_case.
5. When no relevant evidence is returned, state that the supplied documents are insufficient. Do not answer from general knowledge.
6. After a successful tool call, answer directly, mention the exact filename, preserve all documented values, and avoid promises beyond the evidence."""

    agent=create_agent(model=model,tools=tools,system_prompt=prompt)
    return agent

def main():
    files=discover_supported_files(DATA_DIR)
    doc=load_documents(files)
    chunk=split_document(doc)
    build_vector_store(chunk)
    

    question=input("Ask question: ")
    agent=build_agent()
    response=agent.invoke({
        "messages":[{
            "role":"user",
            "content":question
        }]
    },config={"recursion_limit":20})

    print(response["messages"][-1].content[0]["text"])

if __name__=="__main__":
    main()
