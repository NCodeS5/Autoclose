"""Vector stores behind one interface. VECTOR_STORE=local (TF-IDF, no keys) or pinecone.

Namespaces: 'docs' (policies + contracts) and 'cases' (resolved past cases)."""
import os, time
from dataclasses import dataclass
from .chunking import Chunk

DOCS, CASES = "docs", "cases"

@dataclass
class Hit:
    id: str
    text: str
    score: float
    meta: dict

class LocalStore:
    name = "local-tfidf"
    def __init__(self):
        self._data = {}      # ns -> {id: Chunk}
        self._idx = {}       # ns -> (vectorizer, matrix, ids)
    def upsert(self, ns, chunks):
        d = self._data.setdefault(ns, {})
        for c in chunks: d[c.id] = c
        self._idx.pop(ns, None)
    def count(self, ns): return len(self._data.get(ns, {}))
    def wait_visible(self, ns, chunk, timeout=15.0):
        return True            # local writes are searchable immediately
    def purge_except(self, ns, keep_ids):
        """Delete every record in the namespace except keep_ids (used to forget learned cases, keep the seed cases)."""
        keep = set(keep_ids)
        self.delete_ids(ns, [i for i in list(self._data.get(ns, {})) if i not in keep])
    def delete_ids(self, ns, ids):
        for i in ids: self._data.get(ns, {}).pop(i, None)
        self._idx.pop(ns, None)
    def _build(self, ns):
        from sklearn.feature_extraction.text import TfidfVectorizer
        items = list(self._data.get(ns, {}).values())
        if not items: return None
        v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        m = v.fit_transform([c.text for c in items])
        self._idx[ns] = (v, m, items)
        return self._idx[ns]
    def search(self, ns, query, top_k=3, filter=None):
        idx = self._idx.get(ns) or self._build(ns)
        if not idx: return []
        v, m, items = idx
        sims = (m @ v.transform([query]).T).toarray().ravel()
        out = []
        for i in sims.argsort()[::-1]:
            c = items[i]
            if filter and any(c.meta.get(k) != val for k, val in filter.items()): continue
            if sims[i] <= 0: continue
            out.append(Hit(c.id, c.text, float(sims[i]), c.meta))
            if len(out) == top_k: break
        return out

class PineconeStore:
    """Pinecone serverless index with integrated embeddings (Pinecone embeds text for us).
    Needs PINECONE_API_KEY. Index is created on first use."""
    name = "pinecone"
    def __init__(self, client=None, index_name=None, model=None):
        from pinecone import Pinecone
        self.pc = client or Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index_name = index_name or os.getenv("PINECONE_INDEX", "autoclose-kb")
        self.model = model or os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2")
        if not self.pc.has_index(self.index_name):
            self.pc.create_index_for_model(
                name=self.index_name, cloud=os.getenv("PINECONE_CLOUD", "aws"),
                region=os.getenv("PINECONE_REGION", "us-east-1"),
                embed={"model": self.model, "field_map": {"text": "text"}})
        self.index = self.pc.Index(self.index_name)
    def upsert(self, ns, chunks):
        recs = [{"_id": c.id, "text": c.text, **{k: v for k, v in c.meta.items() if v is not None}} for c in chunks]
        for i in range(0, len(recs), 90):     # API batch limit for text records is small
            self.index.upsert_records(namespace=ns, records=recs[i:i + 90])
    def count(self, ns):
        s = self.index.describe_index_stats()
        nss = s["namespaces"] if isinstance(s, dict) else getattr(s, "namespaces", {})
        n = nss.get(ns)
        if n is None: return 0
        return n["vector_count"] if isinstance(n, dict) else n.vector_count
    def delete_ids(self, ns, ids): self.index.delete(ids=list(ids), namespace=ns)
    def purge_except(self, ns, keep_ids):
        """Delete every record in the namespace except keep_ids. Learned cases persist in Pinecone across
        server restarts, so this runs at startup and on Reset to give each demo a clean memory."""
        keep = set(keep_ids)
        try:
            ids = []
            for page in self.index.list(namespace=ns):
                for item in page:
                    ids.append(item if isinstance(item, str) else getattr(item, "id", None) or item.get("id"))
        except Exception:
            # listing unsupported or failed: fall back to the ids this app can create (CASE-C01 ... CASE-C28, with -rN reruns)
            ids = [f"CASE-C{n:02d}" + (f"-r{r}" if r else "") for n in range(1, 29) for r in [0] + list(range(2, 12))]
        doomed = [i for i in ids if i and i not in keep]
        for i in range(0, len(doomed), 100):
            self.index.delete(ids=doomed[i:i + 100], namespace=ns)
        return len(doomed)
    def wait_visible(self, ns, chunk, timeout=15.0, poll=1.0):
        """Pinecone is eventually consistent: a new record can take a few seconds to show up in search.
        Poll until the record is searchable so the next invoice can use it."""
        import time
        end = time.time() + timeout
        flt = {"vendor": chunk.meta["vendor"]} if chunk.meta.get("vendor") else None
        while time.time() < end:
            if any(h.id == chunk.id for h in self.search(ns, chunk.text, top_k=5, filter=flt)):
                return True
            time.sleep(poll)
        return False
    def search(self, ns, query, top_k=3, filter=None):
        q = {"inputs": {"text": query}, "top_k": top_k}
        if filter: q["filter"] = {k: {"$eq": v} for k, v in filter.items()}
        res = self.index.search(namespace=ns, query=q, fields=["text", "doc_type", "vendor", "title", "outcome"])
        result = res["result"] if isinstance(res, dict) else res.result
        hits = result["hits"] if isinstance(result, dict) else result.hits
        def g(o, *keys):
            for k in keys:
                v = o.get(k) if isinstance(o, dict) else getattr(o, k, None)
                if v is not None:
                    return v
            return None
        out = []
        for h in hits:
            f = g(h, "fields") or {}
            out.append(Hit(g(h, "id", "_id"), f.get("text", ""), float(g(h, "score", "_score") or 0.0), dict(f)))
        return out

def get_store(kind: str = None):
    kind = (kind or os.getenv("VECTOR_STORE", "local")).lower()
    return PineconeStore() if kind == "pinecone" else LocalStore()
