"""Load the knowledge base and past cases into a store.  Run: python -m autoclose.rag.ingest"""
from .chunking import load_kb, load_past_cases
from .stores import get_store, DOCS, CASES

def ingest(store) -> dict:
    docs, cases = load_kb(), load_past_cases()
    store.upsert(DOCS, docs)
    store.upsert(CASES, cases)
    return {"store": store.name, "docs": len(docs), "cases": len(cases)}

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv; load_dotenv()
    except ImportError:
        pass
    print(ingest(get_store()))
