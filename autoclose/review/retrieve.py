"""Retrieval planning: what to look up is decided from the checks, then fetched from the vector store."""
from ..rag import DOCS, CASES

def plan_queries(inv: dict, ck: dict) -> list:
    """Return [(label, query, filter, top_k)] for the docs namespace."""
    q = [("match", "invoice match tolerance purchase order approved", {"doc_type": "policy"}, 1)]
    if ck["variance_pct"] is not None and abs(ck["variance_pct"]) > 2:
        q.append(("overage", "overage approval variance above purchase order tolerance", {"doc_type": "policy"}, 1))
    if ck["duplicate_of"]:
        q.append(("duplicate", "duplicate invoice same invoice number void", {"doc_type": "policy"}, 1))
    if ck["category_mismatch"]:
        q.append(("category", "expense categorization vendor default category recode", {"doc_type": "policy"}, 1))
    if ck["doc_type"] != "invoice":
        q.append(("writeoff", "write-offs credit memos adjusting entries human accountant approve", {"doc_type": "policy"}, 1))
    if not ck["known_vendor"]:
        q.append(("newvendor", "new vendors vendor master onboarding may not be paid", {"doc_type": "policy"}, 1))
    if ck["total"] >= 25000:
        q.append(("materiality", "materiality threshold Controller approval invoice total", {"doc_type": "policy"}, 1))
    if ck["late_fee"]:
        q.append(("latefee", "payment terms late fees contract late-fee clause", {"doc_type": "policy"}, 1))
    if ck["usage_based"]:
        q.append(("usage", "usage-based services vary from purchase order", {"doc_type": "policy"}, 1))
    if ck["known_vendor"]:
        q.append(("contract", f'{inv["description"]} fees cap rates price payment terms late fee increase',
                  {"doc_type": "contract", "vendor": inv["vendor"]}, 4))
    return q

def retrieve(store, inv: dict, ck: dict, case_k: int = 3):
    """Runs every planned search at the same time (they are independent network calls), keeps plan order."""
    from concurrent.futures import ThreadPoolExecutor
    plan = plan_queries(inv, ck)
    cq = f'{inv["vendor"]} {inv["category"]} {inv["description"]}'
    with ThreadPoolExecutor(max_workers=8) as pool:
        doc_futs = [pool.submit(store.search, DOCS, q, k, flt) for _, q, flt, k in plan]
        case_fut = pool.submit(store.search, CASES, cq, case_k, {"vendor": inv["vendor"]})
        seen, passages = set(), []
        for (label, _, _, _), fut in zip(plan, doc_futs):
            for h in fut.result():
                if h.id not in seen:
                    seen.add(h.id); passages.append({"id": h.id, "text": h.text, "score": round(h.score, 3),
                                                      "why": label, "meta": h.meta})
        cases = [{"id": h.id, "text": h.text, "score": round(h.score, 3), "outcome": h.meta.get("outcome"),
                  "vendor": h.meta.get("vendor")} for h in case_fut.result()]
    return passages, cases
