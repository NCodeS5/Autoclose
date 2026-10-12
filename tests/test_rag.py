from autoclose.rag import LocalStore, PineconeStore, ingest, load_kb, DOCS, CASES, Chunk

def test_kb_chunks_have_ids_and_vendor_metadata():
    ch = {c.id: c for c in load_kb()}
    assert {"P-1", "P-8", "BW-2", "PP-1"} <= set(ch)
    assert ch["BW-2"].meta["vendor"] == "Brightwell Consulting" and ch["P-2"].meta["doc_type"] == "policy"

def test_local_search_filters_by_vendor():
    s = LocalStore(); ingest(s)
    hits = s.search(DOCS, "fee cap not-to-exceed", 3, {"doc_type": "contract", "vendor": "Brightwell Consulting"})
    assert hits[0].id == "BW-2" and all(h.meta["vendor"] == "Brightwell Consulting" for h in hits)

def test_upsert_makes_new_case_searchable():
    s = LocalStore(); ingest(s)
    s.upsert(CASES, [Chunk("CASE-X", "CloudNimbus usage spike approved", {"vendor": "CloudNimbus", "outcome": "approved"})])
    assert s.search(CASES, "usage spike", 1, {"vendor": "CloudNimbus"})[0].id == "CASE-X"

class FakeIndex:
    def __init__(self): self.records = {}
    def upsert_records(self, namespace, records): self.records.setdefault(namespace, []).extend(records)
    def search(self, namespace, query, fields=None):
        r = self.records[namespace][0]
        return {"result": {"hits": [{"_id": r["_id"], "_score": 0.9, "fields": {k: v for k, v in r.items() if k != "_id"}}]}, "q": query}
    def describe_index_stats(self): return {"namespaces": {ns: {"vector_count": len(v)} for ns, v in self.records.items()}}
    def delete(self, ids, namespace): pass

class FakePC:
    def __init__(self): self.created = None; self.idx = FakeIndex()
    def has_index(self, n): return False
    def create_index_for_model(self, **kw): self.created = kw
    def Index(self, n): return self.idx

def test_pinecone_adapter_uses_integrated_embeddings_with_fake_client():
    pc = FakePC(); s = PineconeStore(client=pc, index_name="t")
    assert pc.created["embed"]["field_map"] == {"text": "text"}
    print(ingest(s))
    assert s.count(DOCS) == 25
    h = s.search(DOCS, "tolerance", 1, {"doc_type": "policy"})
    assert h[0].id and h[0].score == 0.9 and h[0].text


class AttrHit:                      # shape of the real pinecone SDK Hit: .id .score .fields
    def __init__(self): self.id, self.score, self.fields = "P-1", 0.8, {"text": "Three-way match tolerance.", "doc_type": "policy"}

class AttrIndex(FakeIndex):
    def search(self, namespace, query, fields=None):
        class R: pass
        r = R(); r.result = R(); r.result.hits = [AttrHit()]; return r

def test_pinecone_adapter_reads_sdk_hit_objects():
    pc = FakePC(); pc.idx = AttrIndex(); s = PineconeStore(client=pc, index_name="t")
    h = s.search(DOCS, "tolerance", 1)
    assert h[0].id == "P-1" and h[0].score == 0.8 and h[0].text.startswith("Three-way")


class SlowIndex(FakeIndex):
    """Record becomes searchable only after a few searches, like an eventually consistent index."""
    def __init__(self): super().__init__(); self.calls = 0
    def search(self, namespace, query, fields=None):
        self.calls += 1
        class R: pass
        r = R(); r.result = R()
        r.result.hits = [AttrHit()] if self.calls >= 3 else []
        return r

def test_wait_visible_polls_until_record_is_searchable():
    pc = FakePC(); pc.idx = SlowIndex(); s = PineconeStore(client=pc, index_name="t")
    c = Chunk("P-1", "Three-way match tolerance.", {"vendor": ""})
    assert s.wait_visible(DOCS, c, timeout=5, poll=0.01) is True and pc.idx.calls >= 3
    pc2 = FakePC(); pc2.idx = SlowIndex(); pc2.idx.calls = -100
    s = PineconeStore(client=pc2, index_name="t")
    assert s.wait_visible(DOCS, c, timeout=0.05, poll=0.01) is False

def test_local_store_is_visible_immediately():
    assert LocalStore().wait_visible(CASES, Chunk("X", "t")) is True


def test_purge_keeps_seed_cases_and_drops_learned_ones():
    s = LocalStore(); ingest(s)
    s.upsert(CASES, [Chunk("CASE-C22", "CloudNimbus spike approved", {"vendor": "CloudNimbus", "outcome": "approved"})])
    assert s.count(CASES) == 6
    s.purge_except(CASES, [c.id for c in __import__("autoclose.rag", fromlist=["load_past_cases"]).load_past_cases()])
    assert s.count(CASES) == 5 and not s.search(CASES, "CloudNimbus spike", 3, {"vendor": "CloudNimbus"})

class ListIndex(FakeIndex):
    def __init__(self): super().__init__(); self.deleted = []
    def list(self, namespace): yield ["CASE-H1", "CASE-C22", "CASE-C23-r2"]
    def delete(self, ids, namespace): self.deleted += list(ids)

def test_pinecone_purge_deletes_only_learned_ids():
    pc = FakePC(); pc.idx = ListIndex(); s = PineconeStore(client=pc, index_name="t")
    n = s.purge_except(CASES, ["CASE-H1"])
    assert n == 2 and sorted(pc.idx.deleted) == ["CASE-C22", "CASE-C23-r2"]
