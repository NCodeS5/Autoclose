"""LLM-drafted close commentary (LangChain + AWS Bedrock) with a citation guard.

Contract: the model may only restate numbers that appear in the findings, and every line
must cite a finding id like [F1]. Output that breaks the contract is retried once with
the specific problems fed back; if it still fails, we fall back to a deterministic template
and the escalation gate forces human review.
"""
import os, re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

SYSTEM = (
    "You write month-end close commentary for a finance team. Use ONLY the findings provided. "
    "Rules: (1) one bullet per finding, starting with '- '; (2) end each bullet with its id "
    "in brackets, e.g. [F2]; (3) do not state any number, vendor, or amount that is not in "
    "the finding text; (4) do not speculate about causes. Finish with one line 'Summary: ...' "
    "that cites every id it mentions."
)
HUMAN = "Run {run_id} ({entity}).\nFindings:\n{findings}\n{feedback}"

def number_tokens(text):
    return {m.replace(",", "").rstrip(".") for m in re.findall(r"\d[\d,]*\.?\d*", text)}

def label(findings):
    return [(f"F{i+1}", f) for i, f in enumerate(findings)]

def validate_commentary(text, findings):
    """Return a list of problems (empty list = valid)."""
    problems, ids = [], {fid for fid, _ in label(findings)}
    allowed = set()
    for fid, f in label(findings):
        allowed |= number_tokens(f["detail"]) | number_tokens(fid)
    for line in [l for l in text.splitlines() if l.strip()]:
        cited = set(re.findall(r"\[(F\d+)\]", line))
        if not cited:
            problems.append(f"line has no citation: {line[:60]!r}")
        for c in cited - ids:
            problems.append(f"cites unknown id {c}")
    covered = set(re.findall(r"\[(F\d+)\]", text))
    for missing in ids - covered:
        problems.append(f"finding {missing} not mentioned")
    for n in number_tokens(re.sub(r"\[F\d+\]", "", text)) - allowed:
        problems.append(f"number not in findings: {n}")
    return problems

def template_commentary(findings):
    lines = [f"- {f['detail']} [{fid}]" for fid, f in label(findings)]
    lines.append("Summary: " + ", ".join(f"[{fid}]" for fid, _ in label(findings)) + " noted.")
    return "\n".join(lines)

def build_llm():
    """AUTOCLOSE_LLM=bedrock uses Bedrock via LangChain (needs AWS creds + model access).
    Anything else uses an offline stub so tests and the eval run without credentials."""
    if os.getenv("AUTOCLOSE_LLM", "stub") == "bedrock":
        from langchain_aws import ChatBedrockConverse
        return ChatBedrockConverse(
            model=os.environ["BEDROCK_MODEL_ID"],
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            temperature=0, max_tokens=600)
    return RunnableLambda(lambda _: "STUB")  # signals: use template

def make_commentator(llm=None, max_attempts=2):
    llm = llm or build_llm()
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    chain = prompt | llm | StrOutputParser()

    def commentate(run, findings):
        """Returns (text, ok, attempts, problems)."""
        if not findings:
            return "No exceptions found.", True, 0, []
        listing = "\n".join(f"[{fid}] ({f['kind']}) {f['detail']}" for fid, f in label(findings))
        feedback, problems = "", []
        for attempt in range(1, max_attempts + 1):
            text = chain.invoke(dict(run_id=run["run_id"], entity=run["entity"],
                                     findings=listing, feedback=feedback))
            if text.strip() == "STUB":
                text = template_commentary(findings)
            problems = validate_commentary(text, findings)
            if not problems:
                return text, True, attempt, []
            feedback = "Your previous draft broke these rules, fix them:\n" + "\n".join(problems)
        return template_commentary(findings), False, max_attempts, problems
    return commentate
