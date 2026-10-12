# AutoClose: agent-run month-end close with three human edge cases

A personal mock-up of a workflow I've worked on: agents handle a month-end financial close
end to end and ask a human only when they should. **All data is synthetic with injected,
labeled anomalies, so every number below was measured here.**

## Run it
    python -m venv .venv
    .venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
    pip install -r requirements.txt -r requirements-dev.txt
    python -m pytest -q                 # 20 tests
    python run_eval.py 300              # offline stub LLM, no credentials needed

Optional, real LLM commentary via AWS Bedrock (needs AWS credentials and model access):

    export AUTOCLOSE_LLM=bedrock BEDROCK_MODEL_ID=<your model id> AWS_REGION=us-east-1
    python run_eval.py 300

CI-style gate: `python run_eval.py 300 --min-recall 0.95 --max-auto-error 0.01` exits 1 on a breach.

### API (FastAPI)
    uvicorn autoclose.api:app --port 8000
    curl -X POST localhost:8000/closes -H "content-type: application/json" -d '{"scenario":"writeoff"}'
    curl localhost:8000/queue
    curl -X POST localhost:8000/cases/<thread_id>/decision -H "content-type: application/json" -d '{"resolution":"issue"}'
Scenarios: clean, duplicate, misclassified, writeoff, missing_bank, novel_vendor. Also `/health`, `/metrics`.
Easiest way to demo: open http://localhost:8000/docs and click through the endpoints in the browser.

Windows PowerShell (curl there is a different tool):
    $b = @{scenario="writeoff"} | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/closes -ContentType "application/json" -Body $b
    Invoke-RestMethod http://localhost:8000/queue
    $d = @{resolution="issue"} | ConvertTo-Json
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/cases/<thread_id>/decision -ContentType "application/json" -Body $d

### Spark baseline job (needs Java 17+)
    pip install pyspark
    python spark/build_baseline.py --demo      # writes vendor_baseline.csv
Load it with `Baseline.from_csv("vendor_baseline.csv")`. A test checks it matches the Python baseline.

### Container, CI, Kubernetes
    docker build -t autoclose .  &&  docker run -p 8000:8000 autoclose
`.github/workflows/ci.yml` runs tests, the metric gate, then builds and smoke-tests the image.
`k8s/` has a Deployment, Service and HPA. Keep one replica until paused-case state moves to PostgreSQL.

**Verified by running:** tests (including API and Spark), eval gate. **Written but not run:** Dockerfile build,
GitHub Actions workflow, Kubernetes manifests (YAML parses), Bedrock calls.

## How it works (LangGraph)
ingest → validation agent → reconciliation agent → analytics agent → report (LLM + citation guard)
→ **escalation gate** → human review via `interrupt` (checkpointed) → feedback → end

**Agents resolve on their own:** exact duplicates (propose a void), miscategorized entries
(propose a recategorization), cent-rounding/timing differences, known benign patterns.
Agents only *propose* actions; source data is never edited.

**Humans see only three edge cases:**
- **E1 Policy:** any write-off or adjusting entry.
- **E2 Unexplained:** a missing bank match or large outlier the agents can't explain with evidence.
- **E3 Novel:** an unknown vendor, or a variance pattern unlike anything resolved before.

Safety net: a failed citation check on the LLM commentary routes to the same queue.
~5% of auto-completed runs are audit-sampled to a human to measure error on automated work.
Feedback loop: reviewer-confirmed benign patterns are remembered, so they auto-resolve next time.

## Results (300 runs, seed 7; 64 runs truly need a human: E1 20, E2 37, E3 14)
| Metric | Value |
|---|---|
| Automation rate (no human touch) | 66.3% |
| Escalation precision / recall | 78.0% / 100% |
| Recall by edge case (E1 / E2 / E3) | 20/20, 37/37, 14/14 |
| Agent fix actions correct | 37/37 |
| Error rate on auto-completed runs | 0/218 (audit sample 0/19) |
| Human minutes per 100 runs | 202 vs 2,000 manual baseline |

Assumptions (stated, not measured): manual close = 20 min/run; review = 6 min/case.

## Honest limitations
- The three agents are deterministic reference implementations returning the same JSON contract
  an LLM tool-calling agent would; the LLM is used for commentary only (stubbed offline).
  Next: Bedrock tool-calling agents behind the same contract.
- Perfect recall and 37/37 actions reflect simple synthetic anomalies; real data would be messier.
- Precision loss comes from benign seasonal swings that look suspicious the first time they appear.
- The feedback loop's effect on escalation rate is modest and noisy.
- Reviewer is simulated from labels; no review UI yet.

## Next
FastAPI review queue + PostgreSQL audit trail, Docker + GitHub Actions gate, Bedrock agents.
