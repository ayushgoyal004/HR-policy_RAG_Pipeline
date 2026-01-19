import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LC_Document

FAISS_INDEX_DIR = "faiss_index"

# Data Models
@dataclass
class Document:
    content: str
    metadata: Dict[str, str]

@dataclass
class Chunk:
    text: str
    metadata: Dict[str, str]

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base"
CHUNK_SIZE = 500
TOP_K = 4

# Utility Functions
def classify_document(filename: str) -> str:
    print("Classifying the documents")
    if "policy" in filename.lower():
        return "policy"
    return "noise"


def extract_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract date from filename patterns like:
    - policy_v2_2024.txt
    - policy_2024_01_15.txt
    - policy_Jan_2024.txt
    - WFH_Policy_2024.txt
    """
    print(f"  → Trying filename pattern extraction for: {filename}")
    
    # Pattern 1: YYYY (just year)
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        if 2000 <= year <= 2099:
            print(f"    ✓ Found year in filename: {year}")
            return datetime(year, 1, 1)
    
    # Pattern 2: YYYY_MM_DD or YYYY-MM-DD
    match = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    if match:
        try:
            date = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            print(f"    ✓ Found full date in filename: {date.strftime('%Y-%m-%d')}")
            return date
        except ValueError:
            pass
    
    # Pattern 3: Month_YYYY (e.g., Jan_2024)
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    match = re.search(r'([A-Za-z]{3,9})[-_](\d{4})', filename.lower())
    if match:
        month_str = match.group(1)[:3]
        year = int(match.group(2))
        if month_str in month_map and 2000 <= year <= 2099:
            date = datetime(year, month_map[month_str], 1)
            print(f"    ✓ Found month-year in filename: {date.strftime('%b %Y')}")
            return date
    
    print("    ✗ No date pattern found in filename")
    return None


def extract_date_from_content(text: str) -> Optional[datetime]:
    """
    Extract date from various patterns in document content.
    """
    print("  → Trying content date extraction")
    
    # Pattern 1: "Effective Date: Jan 1, 2024" (original)
    match = re.search(
        r"Effective Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        text,
        re.IGNORECASE
    )
    if match:
        try:
            date = datetime.strptime(match.group(1), "%b %d, %Y")
            print(f"    ✓ Found 'Effective Date': {date.strftime('%b %d, %Y')}")
            return date
        except ValueError:
            pass
    
    # Pattern 2: "(Effective Date: Jan 1, 2024)" with parentheses
    match = re.search(
        r"\(Effective Date:\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\)",
        text,
        re.IGNORECASE
    )
    if match:
        try:
            date = datetime.strptime(match.group(1), "%b %d, %Y")
            print(f"    ✓ Found 'Effective Date' (parenthesized): {date.strftime('%b %d, %Y')}")
            return date
        except ValueError:
            pass
    
    # Pattern 3: "Updated: Jan 2024" or "Last Updated: January 2024"
    match = re.search(
        r"(?:Updated|Last Updated):\s*([A-Za-z]+\s+\d{4})",
        text,
        re.IGNORECASE
    )
    if match:
        try:
            date = datetime.strptime(match.group(1), "%B %Y")
            print(f"    ✓ Found 'Updated': {date.strftime('%b %Y')}")
            return date
        except ValueError:
            try:
                date = datetime.strptime(match.group(1), "%b %Y")
                print(f"    ✓ Found 'Updated': {date.strftime('%b %Y')}")
                return date
            except ValueError:
                pass
    
    # Pattern 4: "As of January 2024" or "Valid from: Jan 1, 2024"
    match = re.search(
        r"(?:As of|Valid from|Issued):\s*([A-Za-z]+\s+\d{1,2}?,?\s+\d{4})",
        text,
        re.IGNORECASE
    )
    if match:
        date_str = match.group(1).replace(',', '').strip()
        for fmt in ["%B %d %Y", "%b %d %Y", "%B %Y", "%b %Y"]:
            try:
                date = datetime.strptime(date_str, fmt)
                print(f"    ✓ Found date variant: {date.strftime('%b %d, %Y')}")
                return date
            except ValueError:
                continue
    
    # Pattern 5: Just a year in the first few lines (fallback)
    first_200_chars = text[:200]
    match = re.search(r'\b(20\d{2})\b', first_200_chars)
    if match:
        year = int(match.group(1))
        date = datetime(year, 1, 1)
        print(f"    ✓ Found year in content: {year}")
        return date
    
    print("    ✗ No date pattern found in content")
    return None


def get_file_modification_date(filepath: str) -> datetime:
    """
    Get file modification timestamp as fallback.
    """
    print(f"  → Using file modification date as fallback")
    timestamp = os.path.getmtime(filepath)
    date = datetime.fromtimestamp(timestamp)
    print(f"    ✓ File modified: {date.strftime('%Y-%m-%d %H:%M:%S')}")
    return date


def extract_effective_date(text: str, filename: str, filepath: str = None) -> datetime:
    """
    Multi-strategy date extraction with fallbacks.
    
    Priority order:
    1. Explicit date in content (highest priority)
    2. Date pattern in filename
    3. File modification timestamp (fallback)
    4. Default to epoch start (last resort)
    """
    print(f"\nExtracting effective date for: {filename}")
    
    # Strategy 1: Extract from content (highest priority)
    content_date = extract_date_from_content(text)
    if content_date:
        return content_date
    
    # Strategy 2: Extract from filename
    filename_date = extract_date_from_filename(filename)
    if filename_date:
        return filename_date
    
    # Strategy 3: Use file modification date (if filepath provided)
    if filepath and os.path.exists(filepath):
        return get_file_modification_date(filepath)
    
    # Strategy 4: Default to very old date (last resort)
    print("    ⚠️  WARNING: No date found, using default date (1970-01-01)")
    print("    → This document will be ranked as oldest")
    return datetime(1970, 1, 1)


def extract_effective_date_with_llm(text: str, filename: str) -> Optional[datetime]:
    """
    Use LLM to extract date when regex patterns fail.
    This is an advanced fallback but costs API calls.
    """
    print("  → Trying LLM-based date extraction (advanced fallback)")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Extract the effective date or creation date from this policy document.
Return ONLY the date in format: YYYY-MM-DD
If no date is found, return: NONE

Document excerpt:
{text[:500]}

Date (YYYY-MM-DD or NONE):"""
    
    try:
        response = llm.invoke(prompt).content.strip()
        if response == "NONE":
            print("    ✗ LLM found no date")
            return None
        
        # Try to parse the response
        date = datetime.strptime(response, "%Y-%m-%d")
        print(f"    ✓ LLM extracted date: {date.strftime('%Y-%m-%d')}")
        return date
    except Exception as e:
        print(f"    ✗ LLM extraction failed: {e}")
        return None


# 1. Document Ingestion (IMPROVED)
def load_documents() -> List[Document]:
    documents = []

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract effective date using multi-strategy approach
        effective_date = extract_effective_date(content, filename, path)

        documents.append(
            Document(
                content=content,
                metadata={
                    "filename": filename,
                    "doc_type": classify_document(filename),
                    "effective_date": effective_date.isoformat(),  # Store as ISO string
                    "effective_year": str(effective_date.year)
                }
            )
        )
        

    return documents


# 2. Chunking (with metadata inheritance)
def chunk_documents(documents: List[Document]) -> List[Chunk]:
    chunks = []

    for doc in documents:
        paragraphs = [p.strip() for p in doc.content.split("\n") if p.strip()]
        buffer = ""

        for para in paragraphs:
            if len(buffer) + len(para) <= CHUNK_SIZE:
                buffer += " " + para
            else:
                chunks.append(
                    Chunk(
                        text=buffer.strip(),
                        metadata=doc.metadata.copy()  # Inherit all metadata
                    )
                )
                buffer = para

        if buffer:
            chunks.append(
                Chunk(
                    text=buffer.strip(),
                    metadata=doc.metadata.copy()
                )
            )

    return chunks


# 3. FAISS Vector Store
def to_langchain_documents(chunks: List[Chunk]) -> List[LC_Document]:
    return [
        LC_Document(
            page_content=chunk.text,
            metadata=chunk.metadata
        )
        for chunk in chunks
    ]


def build_vector_store(chunks: List[Chunk]) -> FAISS:
    embeddings = OpenAIEmbeddings()
    lc_docs = to_langchain_documents(chunks)
    vector_store = FAISS.from_documents(lc_docs, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"✓ Vector store saved to {FAISS_INDEX_DIR}")
    return vector_store


def load_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_chunks(vector_store: FAISS, query: str, k: int) -> List[Chunk]:
    docs = vector_store.similarity_search(query, k=k)
    
    chunks = [
        Chunk(
            text=doc.page_content,
            metadata=doc.metadata
        )
        for doc in docs
    ]
    
    # for i, chunk in enumerate(chunks, 1):
    #     print(f"\nChunk {i}:")
    #     print(f"  Filename: {chunk.metadata.get('filename')}")
    #     print(f"  Type: {chunk.metadata.get('doc_type')}")
    #     print(f"  Date: {chunk.metadata.get('effective_date', 'Unknown')}")
    
    return chunks


# 4. Noise Filtering
def filter_noise(chunks: List[Chunk]) -> List[Chunk]:

    
    filtered = [
        chunk for chunk in chunks
        if chunk.metadata.get("doc_type") == "policy"
    ]
    
    print(f"✓ Kept {len(filtered)} policy chunks (filtered {len(chunks) - len(filtered)} noise)")
    return filtered


# 5. Conflict Resolution (IMPROVED)
def resolve_policy_conflicts(chunks: List[Chunk]) -> Chunk:
    
    # Parse effective dates from metadata
    policies_with_dates = []
    
    for chunk in chunks:
        effective_date_str = chunk.metadata.get("effective_date")
        if effective_date_str:
            try:
                effective_date = datetime.fromisoformat(effective_date_str)
                policies_with_dates.append((effective_date, chunk))
                print(f"  → {chunk.metadata['filename']}: {effective_date.strftime('%Y-%m-%d')}")
            except ValueError:
                print(f"  ⚠️  Warning: Invalid date format in metadata for {chunk.metadata.get('filename')}")
    
    if not policies_with_dates:
        print("  ⚠️  WARNING: No dated policies found, using first available chunk")
        return chunks[0]
    
    # Sort by date (newest first)
    policies_with_dates.sort(key=lambda x: x[0], reverse=True)
    
    most_recent = policies_with_dates[0]
    print(f"\n✓ Selected most recent policy:")
    
    return most_recent[1]


# 6. Answer Generation
def generate_answer(policy_chunk: Chunk, query: str) -> str:    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are an HR policy assistant.

Rules:
- Use ONLY the policy text below.
- Prefer the most recent policy.
- Ignore irrelevant documents.
- Answer clearly and concisely.
- Always cite the filename.
- DO NOT invent source names.
- The source filename will be provided programmatically.

Policy Text:
{policy_chunk.text}

Question:
{query}

Answer format:
<answer>
Sources: <filename>
"""

    response = llm.invoke(prompt).content.strip()
    response = re.sub(r"Sources:.*", "", response).strip()
    response += f"\nSources: {policy_chunk.metadata['filename']}"

    return response


# 7. Pipeline Orchestration
def run_pipeline(query: str):
    print("\n" + "="*80)
    print("RUNNING RAG PIPELINE")
    print("="*80)
    
    if os.path.exists(FAISS_INDEX_DIR):
        print("\n→ Loading existing FAISS index...")
        vector_store = load_vector_store()
    else:
        print("\n→ Building FAISS index from scratch...")
        documents = load_documents()
        chunks = chunk_documents(documents)
        vector_store = build_vector_store(chunks)

    retrieved = retrieve_chunks(vector_store, query, TOP_K)
    filtered = filter_noise(retrieved)

    if not filtered:
        print("\n⚠️  No relevant policy documents found.")
        return
    
    authoritative_policy = resolve_policy_conflicts(filtered)
    answer = generate_answer(authoritative_policy, query)
    
    print("\n" + "="*80)
    print("FINAL ANSWER")
    print("="*80)
    print(answer)
    print("="*80 + "\n")


# Entry Point
if __name__ == "__main__":
    user_query = input("\nEnter your query: ")
    run_pipeline(user_query)