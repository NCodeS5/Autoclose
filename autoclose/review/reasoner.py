"""The reasoning step. Reads ONLY the retrieved passages and similar cases; returns a JSON decision:
{decision: approve|void_duplicate|recode|escalate, edge: E1|E2|E3|null, confidence, citations, rationale, new_category}

LLM_PROVIDER=offline (default, rule reader over the retrieved text, no keys) | anthropic | bedrock."""
import json, os, re

CASE_MATCH_MIN = float(os.getenv("CASE_MATCH_MIN", "0.35"))

def _num(s): return float(s.replace(",", ""))

def _find(passages, pattern):
    for p in passages:
        m = re.search(pattern, p["text"], re.I)
        if m: return p, m
    return None, None

def offline_reason(inv, ck, passages, cases) -> dict:
    ids = {p["id"] for p in passages}
    def out(decision, edge, conf, cites, why, new_category=None):
        return {"decision": decision, "edge": edge, "confidence": conf, "citations": cites,
                "rationale": why, "new_category": new_category}
    if ck["doc_type"] != "invoice":
        p, _ = _find(passages, r"write-off|credit memo")
        return out("escalate", "E1", 0.95 if p else 0.6, [p["id"]] if p else [],
                   f'This is a {ck["doc_type"].replace("_", " ")}. Policy requires a human accountant to approve it.')
    if not ck["known_vendor"]:
        p, _ = _find(passages, r"not in the vendor master")
        return out("escalate", "E3", 0.95 if p else 0.6, [p["id"]] if p else [],
                   f'{inv["vendor"]} is not in the vendor master, so onboarding is required before payment.')
    if ck["duplicate_of"]:
        p, _ = _find(passages, r"is a duplicate when")
        return out("void_duplicate", None, 0.95 if p else 0.6, [p["id"]] if p else [],
                   f'Matches earlier invoice {ck["duplicate_of"]} from the same vendor. Propose voiding this entry.')
    p, m = _find(passages, r"total of \$([\d,]+) or more")
    if p and ck["total"] >= _num(m.group(1)):
        return out("escalate", "E1", 0.95, [p["id"]], f'Invoice total ${ck["total"]:,.2f} is at or above the ${m.group(1)} materiality threshold. Controller approval is required.')
    p, m = _find(passages, r"not-to-exceed \$([\d,]+)")
    if p and ck["ytd_billed"] is not None and ck["ytd_billed"] > _num(m.group(1)):
        return out("escalate", "E1", 0.95, [p["id"]], f'Billed this year would reach ${ck["ytd_billed"]:,.2f}, above the contract cap of ${m.group(1)}.')
    if ck["late_fee"]:
        p, _ = _find(passages, r"no late fee")
        pol, _ = _find(passages, r"late fee or finance charge may be paid only")
        if p:
            return out("escalate", "E1", 0.92, [x["id"] for x in (pol, p) if x],
                       "The invoice includes a late fee but the vendor contract has no late-fee clause. The AP Manager must review it.")
    if ck["category_mismatch"]:
        p, _ = _find(passages, r"default category")
        return out("recode", None, 0.93 if p else 0.6, [p["id"]] if p else [],
                   f'Coded to {inv["category"]} but the vendor default is {ck["vendor_default_category"]}. Propose recoding.',
                   ck["vendor_default_category"])
    var = ck["variance_pct"]
    if var is None:
        return out("escalate", "E2", 0.5, [], "No purchase order amount to match against and the description is too vague to explain.")
    pt, mt = _find(passages, r"within (\d+(?:\.\d+)?)% of the purchase order")
    if not pt:
        return out("escalate", "E2", 0.5, [], "The match tolerance policy was not retrieved, so the invoice cannot be judged.")
    tol = _num(mt.group(1))
    contract = next((p for p in passages if p["meta"].get("doc_type") == "contract"), None)
    cites = [pt["id"]] + ([contract["id"]] if contract else [])
    if abs(var) <= tol:
        return out("approve", None, 0.93, cites, f"Invoice is {var:+.2f}% from the purchase order, inside the {mt.group(1)}% tolerance.")
    po_, mo = _find(passages, r"up to (\d+(?:\.\d+)?)% above the purchase order")
    lim = _num(mo.group(1)) if mo else 10.0
    if abs(var) <= lim:
        return out("escalate", "E1", 0.9, [po_["id"]] if po_ else [pt["id"]],
                   f"Invoice is {var:+.2f}% from the purchase order, above the {mt.group(1)}% tolerance. The AP Manager must approve this overage.")
    # large unexplained variance: allowed only if a person already approved the same vendor pattern
    rej = next((c for c in cases if c["outcome"] == "rejected" and c["score"] >= CASE_MATCH_MIN), None)
    if rej:
        return out("escalate", "E2", 0.7, [po_["id"]] if po_ else [pt["id"]],
                   f'Invoice is {var:+.2f}% from the purchase order, beyond the limit, and a person rejected a similar earlier case ({rej["id"]}), so a person must decide.')
    for c in cases:
        if c["outcome"] == "approved" and c["score"] >= CASE_MATCH_MIN:
            pu, _ = _find(passages, r"usage") if ck["usage_based"] else (None, None)
            pol = pu or po_ or pt
            return out("approve", None, 0.85, [x for x in [pol and pol["id"], c["id"]] if x],
                       f'Variance is {var:+.2f}%. A person already approved the same pattern for this vendor ({c["id"]}), which policy allows us to follow.')
    return out("escalate", "E2", 0.7, [po_["id"]] if po_ else [pt["id"]],
               f"Invoice is {var:+.2f}% from the purchase order, beyond the {mo.group(1) if mo else '10'}% limit, with no earlier approved pattern to rely on.")

SYSTEM = """You are an accounts payable review agent. Decide using ONLY the policy and contract passages and similar past cases provided.
Rules: never approve write-offs, credit memos, unknown vendors, duplicates, or totals the policy says need higher approval.
If the evidence does not let you decide, escalate with edge E2 and low confidence. Cite only passage or case ids you were given.
Edge codes: E1 = policy names an approver for a routine case (credit memo, write-off, overage in the AP Manager's band, total above the materiality threshold, contract cap exceeded, unsupported late fee). E2 = unexplained or unclear (overage above the Controller limit, no purchase order, conflicting evidence, low confidence). E3 = novel (unknown vendor).
Take every number from the invoice, the checks or the passages. Do not quote numbers that are not in them.
A similar past case that a person approved, for the same vendor and the same cause, may be followed even if the percentage differs.
Reply with JSON only: {{"decision":"approve|void_duplicate|recode|escalate","edge":"E1|E2|E3|null","confidence":0-1,"citations":["id"],"rationale":"one or two sentences","new_category":null}}"""

def get_llm():
    prov = os.getenv("LLM_PROVIDER", "offline").lower()
    if prov == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"), max_tokens=600)   # no temperature: newer models reject it
    if prov == "bedrock":
        from langchain_aws import ChatBedrockConverse
        kw = dict(model=os.environ["BEDROCK_MODEL_ID"], region_name=os.getenv("AWS_REGION", "us-east-1"), temperature=0)
        try:
            from botocore.config import Config      # more patient retries with backoff for throttling
            return ChatBedrockConverse(**kw, config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=60))
        except TypeError:
            return ChatBedrockConverse(**kw)
    return None

def comparison_note(ck, passages, cases=()) -> str:
    """Do the arithmetic in code so the model does not have to: tolerance, overage band, and contract cap."""
    parts = []
    var = ck.get("variance_pct")
    _, m = _find(passages, r"within (\d+(?:\.\d+)?)% of the purchase order")
    _, mo = _find(passages, r"up to (\d+(?:\.\d+)?)% above the purchase order")
    if var is not None and m:
        tol = float(m.group(1)); lim = float(mo.group(1)) if mo else 10.0
        v = abs(var)
        if v <= tol:
            band = f"WITHIN the {tol:g}% match tolerance (no overage review needed)"
        elif v <= lim:
            band = f"ABOVE the {tol:g}% tolerance but not above {lim:g}%: edge E1, the AP Manager approves"
        else:
            band = f"ABOVE the {lim:g}% limit: edge E2, unless an earlier approved pattern for this vendor applies"
        parts.append(f"Computed by code: the invoice differs from the purchase order by {v:.2f}%, which is {band}.")
    _, mc = _find(passages, r"not-to-exceed \$([\d,]+)")
    if mc and ck.get("ytd_billed") is not None:
        cap = _num(mc.group(1)); ytd = ck["ytd_billed"]
        parts.append(f"Computed by code: billed this contract year including this invoice is ${ytd:,.2f}, which is "
                     f"{'ABOVE' if ytd > cap else 'within'} the ${cap:,.0f} contract cap.")
    rej = next((c for c in cases if c["outcome"] == "rejected" and c["score"] >= CASE_MATCH_MIN), None)
    if var is not None and abs(var) > 10 and rej:
        parts.append(f'Computed by code: for this same vendor, a person REJECTED the similar earlier case {rej["id"]} '
                     f'(similarity {rej["score"]:.2f}). Do not approve on the basis of any earlier approved case. Escalate with edge E2 and say a similar case was rejected.')
    elif var is not None and abs(var) > 10:
        for c in cases:
            if c["outcome"] == "approved" and c["score"] >= CASE_MATCH_MIN:
                cite = "P-9" if ck.get("usage_based") else "P-2"
                parts.append(f'Computed by code: for this same vendor, a person already approved the earlier case {c["id"]} '
                             f'(similarity {c["score"]:.2f}). Policy {cite} lets you follow an earlier approved pattern for the same '
                             f'vendor and cause, even if the percentage differs somewhat, so you may approve and cite {c["id"]} and {cite}.')
                break
    return " ".join(parts) or "No variance comparison is available."

def _invoke_json(chain, vars_):
    msg = chain.invoke(vars_)
    text = msg.content if isinstance(msg.content, str) else "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise ValueError(f"no JSON in model output: {text[:120]!r}")
    return json.loads(m.group(0))

def llm_reason(llm, inv, ck, passages, cases) -> dict:
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human",
        "INVOICE:\n{inv}\n\nCHECKS (computed by code):\n{ck}\n{cmp}\n\nPASSAGES:\n{ps}\n\nSIMILAR PAST CASES:\n{cs}\n\n{reminder}")])
    vars_ = {"inv": json.dumps(inv), "ck": json.dumps(ck), "cmp": comparison_note(ck, passages, cases),
             "ps": "\n".join(f'[{p["id"]}] {p["text"]}' for p in passages),
             "cs": "\n".join(f'[{c["id"]}] ({c["outcome"]}) {c["text"]}' for c in cases) or "none", "reminder": ""}
    try:
        d = _invoke_json(prompt | llm, vars_)
    except (ValueError, json.JSONDecodeError):          # one retry: models occasionally add prose or skip the JSON
        vars_["reminder"] = "Your previous reply was not valid JSON. Reply with ONLY the JSON object, nothing else."
        d = _invoke_json(prompt | llm, vars_)
    d["edge"] = None if d.get("edge") in (None, "null", "") else d["edge"]
    return d

def reason(inv, ck, passages, cases, llm=None) -> dict:
    """Returns the decision dict plus 'provider'. LLM failures come back as decision=None so the guard escalates."""
    if llm is None:
        d = offline_reason(inv, ck, passages, cases); d["provider"] = "offline"; return d
    try:
        d = llm_reason(llm, inv, ck, passages, cases); d["provider"] = os.getenv("LLM_PROVIDER")
        ref = offline_reason(inv, ck, passages, cases)        # reference answer from the rules, used only for the cross-check
        d["reference"] = {"decision": ref["decision"], "edge": ref["edge"]}
        return d
    except Exception as e:
        err = f"{type(e).__name__}: {e}"[:300]
        if os.getenv("LLM_FALLBACK", "offline") == "offline":
            d = offline_reason(inv, ck, passages, cases)      # same tested rules; clearly flagged, still guarded
            d["provider"] = f'offline fallback ({os.getenv("LLM_PROVIDER")} failed)'
            d["notes"] = [f"model unavailable, decided by built-in rules instead: {err}"]
            return d
        return {"decision": None, "edge": None, "confidence": 0.0, "citations": [], "rationale": "",
                "provider": os.getenv("LLM_PROVIDER"), "error": err}
