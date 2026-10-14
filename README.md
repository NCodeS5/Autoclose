# AutoClose: invoice review agent with RAG and human escalation

A mock-up I built to reproduce a finance workflow I have worked on, using invented data so I can demo it and explain every decision.
An agent reviews supplier invoices against a company policy manual and vendor contracts, cites the clauses it used,
fixes routine problems itself, and pauses for a person only on three edge cases:

| Edge | Meaning | Example cases |
|---|---|---|
| E1 policy | policy says a person must approve | credit memo, write-off, overage 2-10%, total over $25,000, contract cap exceeded, unsupported late fee |
| E2 unexplained | evidence cannot explain it, or confidence < 0.75 | overage over 10%, no purchase order, bad model output |
| E3 novel | vendor not in the vendor master | unknown supplier |

The earlier month-end-close project is still included (`/month-end`, `run_eval.py`, `README_v1_monthend.md`).

## Run it (no keys needed)
```
py -m venv .venv ; .venv\Scripts\Activate.ps1          # macOS/Linux: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
python run_rag_eval.py
uvicorn autoclose.api:app --port 8000                   # open http://localhost:8000
```
Defaults: local TF-IDF vector store and an offline reasoner that reads the retrieved passages. Everything works without accounts.

## Turn on Pinecone (your key)
```
copy .env.example .env        # then set VECTOR_STORE=pinecone and PINECONE_API_KEY=...
python -m autoclose.rag.ingest
uvicorn autoclose.api:app --port 8000
```
Creates a serverless index with integrated embeddings (`llama-text-embed-v2`) and two namespaces: `docs` (policies and contracts) and `cases` (resolved past cases).
The dashboard header shows which store is live. The Pinecone adapter is tested against a fake client only; the first live run is the real test.
Cosine scores differ from TF-IDF scores, so you may need to tune `CASE_MATCH_MIN` (default 0.35).

## Turn on a real model
`LLM_PROVIDER=anthropic` (needs `ANTHROPIC_API_KEY`) or `LLM_PROVIDER=bedrock` (needs AWS credentials, Bedrock model access and `BEDROCK_MODEL_ID`).
The model gets the invoice, the code-computed checks and the retrieved passages, and must reply with JSON. The guard checks it the same way as the offline reasoner.

## How it works
```
intake -> checks (code) -> retrieve (Pinecone/local) -> reason (LLM or offline) -> guard -> route
                                                                                     |-- auto -> finalize -> feedback
                                                                                     `-- escalate -> human_review (LangGraph interrupt) -> finalize -> feedback
```
- **checks**: duplicate, variance vs purchase order, category vs vendor default, known vendor, contract-year billing, late fee. Plain code.
- **retrieve**: queries are planned from the checks (a duplicate looks up the duplicate policy, a late fee looks up late-fee terms). Contract lookups are filtered to the vendor. Similar past cases come from the `cases` namespace.
- **reason**: returns `{decision, edge, confidence, citations, rationale}`.
- **guard** (code, not model): write-offs, unknown vendors and totals of $25,000+ always escalate; citations must be ids that were actually retrieved; numbers in the rationale must appear in the evidence; confidence under 0.75 escalates; unparseable output escalates.
- **feedback**: when a person approves an E2 case, it is stored in `cases`. The next similar invoice from that vendor can cite it. A rejection teaches nothing.
- **audit trail**: every step is written to SQL (`DATABASE_URL`; in-memory SQLite by default, PostgreSQL when set).
- Agent actions are proposals (void, recode). Source data is never changed.

## Results (31 hand-written cases, local store, offline reasoner)
Run `python run_rag_eval.py`. Last run: 19 of 31 handled by the agent alone, 12 escalated; escalation precision and recall 100%; expected clause retrieved for every case; 0 errors among auto-handled invoices; C23 is handled alone only after C22 was reviewed (it escalates without that history).
Read these honestly: the cases and the rules were written by the same person, so this is a regression test of the pipeline, not independent accuracy. The time comparison uses stated assumptions (5 min per invoice by hand, 3 min per review).

## Limitations
- Offline reasoner is rules over retrieved text; real model behavior is untested until you add a key.
- One worker: paused cases live in process memory (MemorySaver). A PostgreSQL checkpointer is the next step.
- Docker image, Kubernetes manifests and the GitHub Actions workflow are written but were not run in the build environment.
- Only 31 cases and 6 vendors. No OCR or PDF parsing; invoices arrive as structured fields.

## Files
`data/kb/` policy manual and contracts, `data/cases.json` test invoices with expected answers, `autoclose/rag/` chunking, stores, ingest, `autoclose/review/` checks, retrieve, reasoner, guard, flow, audit, service, API, `autoclose/static/review.html` dashboard, `run_rag_eval.py`.
