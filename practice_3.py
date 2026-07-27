import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader,DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "it_service_desk_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}
TOP_K = 3

# These globals may be initialised inside main/build_agent so tools can use them.
retriever = None
supported_files: List[Path] = []

def normalize_text(value: str) -> str:
    clean_value=value.strip()
    clean_value=" ".join(clean_value.split())
    return clean_value

def discover_supported_files(data_dir: Path) -> List[Path]:
    files=[]
    for i in data_dir.iterdir():
        if i.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(i)
    files.sort()
    return files

def load_documents(file_paths: List[Path]):
    documnets=[]

    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.pdf",loader_cls=PyPDFLoader).load()
    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.txt",loader_cls=TextLoader,loader_kwargs={"encoding":"utf-8"}).load()
    documnets+=DirectoryLoader(DATA_DIR,glob="**/*.docx",loader_cls=Docx2txtLoader).load()

    for doc in documnets:
        doc.metadata["source_file"]=Path(
            doc.metadata["source"]
        ).name
        doc.metadata["source_type"]=Path(
            doc.metadata["source"]
        ).suffix.replace(".","")
        
    return documnets

def split_documents(documents):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunk=text_splitter.split_documents(documents)
    return chunk

embedding=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
def build_vector_store(chunks=None):
    if Path(CHROMA_DIR).exists():
        vector_db=Chroma(persist_directory=CHROMA_DIR,embedding_function=embedding)
    else:
        vector_db=Chroma.from_documents(documents=chunks,persist_directory=CHROMA_DIR,embedding=embedding)
    return vector_db

def create_retriever(vector_store):
    retriever=vector_store.as_retriever(search_kwargs={"k":TOP_K})
    return retriever

def format_context(documents) -> str:
    context="\n\n".join("[Source: "+i.metadata.get("source_file","unkown")+i.page_content for i in documents)
    return context


@tool
def search_it_support_documents(query: str) -> str:
    """Search the supplied documents for password policy, lockouts, MFA, device use, administrator access, incident priorities, response targets, outage reporting, phishing, malware, lost devices, data exposure evidence."""
    clean_query=normalize_text(query)
    if query=="":
        raise Exception("Provide the question")
    vector_db=build_vector_store()
    retriever=create_retriever(vector_db)
    doc=retriever.invoke(clean_query)
    context=format_context(doc)
    return context



@tool
def list_it_sources() -> str:
    """List the supplied knowledge filenames."""
    files=discover_supported_files(DATA_DIR)
    f=[i.name for i in files]
    return "\n".join(f)


@tool
def get_it_rule_excerpt(keyword: str, source_name: str) -> str:
    """Return relevant evidence from one requested source document."""
    vector_db=build_vector_store()
    retriever=create_retriever(vector_db)
    doc=retriever.invoke(keyword)


    mn=[]
    for i in doc:
        source=i.metadata("source_file","")
        if i.lower()==source_name.lower():
            mn.append(i)
    return format_context(mn)

@tool
def prepare_security_escalation(reason: str, evidence_source: str) -> str:
    """Create a deterministic hand-off note for the IT service desk, infrastructure, or security response team."""
    # TODO: return exactly:
    # SECURITY ESCALATION | Reason: <reason> | Evidence: <source>
    pass

def build_agent(active_retriever):
    # TODO: expose retriever to tools.
    # TODO: initialise Gemini 3.1 Flash Lite with temperature=0.
    # TODO: create an agent using exactly the four required tools.
    pass

def main() -> None:
    # TODO: validate key -> discover -> load -> split -> index -> retrieve -> agent.
    # TODO: read one question, invoke the agent, print only the final response.
    pass

if __name__ == "__main__":
    main()

