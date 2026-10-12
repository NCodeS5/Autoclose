"""Guard: checks the reasoner's decision before anyone acts on it. Hard rules live here, in code,
so no model output can approve a write-off, an unknown vendor, a duplicate or a $25,000+ invoice."""
import re

MIN_CONFIDENCE = 0.75
MATERIALITY = 25000.0

def _numbers(text):
    """Numbers as values, so $6,695 equals $6,695.00 and a trailing full stop is ignored."""
    out = set()
    for n in re.findall(r"\d[\d,]*\.?\d*", text):
        n = n.replace(",", "").rstrip(".")
        if n:
            out.add(round(float(n), 2))
    return out

def _haystack(inv, ck, passages, cases):
    parts = [str(v) for v in inv.values()] + [str(v) for v in ck.values() if v is not None]
    parts += [p["text"] for p in passages] + [c["text"] for c in cases]
    h = " ".join(parts)
    if ck["variance_pct"] is not None:
        v = abs(ck["variance_pct"]); h += f" {v:.2f} {v:.1f} {round(v)}"
    for k in ("amount", "total", "ytd_billed"):
        x = inv.get(k) if k in inv else ck.get(k)
        if isinstance(x, (int, float)): h += f" {x:.2f} {x:,.2f}"
    return _numbers(h)

def guard(d, inv, ck, passages, cases) -> dict:
    notes, g = list(d.get("notes", [])), dict(d)
    g["proposed"] = d.get("decision")
    if d.get("decision") not in ("approve", "void_duplicate", "recode", "escalate"):
        g.update(decision="escalate", edge="E2", confidence=0.0)
        notes.append("safety net: model output missing or unparseable" + (f' ({d["error"]})' if d.get("error") else ""))
        g["safety_net"] = True
    # hard rules (cannot be overridden by the model)
    forced = None
    if ck["doc_type"] != "invoice": forced = "E1"
    elif not ck["known_vendor"]: forced = "E3"
    elif ck["total"] >= MATERIALITY: forced = "E1"
    if forced and g["decision"] != "escalate":
        g.update(decision="escalate", edge=forced); notes.append(f"hard rule forced escalation ({forced})")
    elif forced and g.get("edge") != forced:
        g["edge"] = forced; notes.append(f"hard rule set the edge case to {forced}")
    if ck["duplicate_of"] and g["decision"] == "approve":
        g.update(decision="void_duplicate", edge=None); notes.append("hard rule: duplicate cannot be approved")
    # cross-check against the rules: a model may be MORE cautious than the rules, never less
    ref = d.get("reference")
    if ref and g["decision"] != "escalate" and ref["decision"] == "escalate":
        g.update(decision="escalate", edge=ref["edge"] or "E2")
        notes.append(f'model wanted {g["proposed"]} but the rules require a person ({ref["edge"]}); sent to a person')
    elif ref and g["decision"] == "escalate" and ref["decision"] != "escalate" and not g.get("safety_net"):
        notes.append(f'note: model was more cautious than the rules (rules said {ref["decision"]})')
    # citations must come from what was retrieved
    allowed = {p["id"] for p in passages} | {c["id"] for c in cases}
    bad = [c for c in g.get("citations", []) if c not in allowed]
    if bad:
        g["citations"] = [c for c in g["citations"] if c in allowed]
        g.update(decision="escalate", edge="E2", safety_net=True); notes.append(f"safety net: cited ids not retrieved {bad}")
    if g["decision"] in ("approve", "void_duplicate", "recode") and not g.get("citations"):
        g.update(decision="escalate", edge="E2", safety_net=True); notes.append("safety net: no citation for an automatic action")
    # numbers in the rationale must exist in the evidence
    if g.get("rationale"):
        text = g["rationale"]
        text = re.sub(r"\b[A-Z]{1,4}-[A-Za-z0-9]+\b", " ", text)   # invoice/clause/case ids are labels, not claims
        unknown = _numbers(text) - _haystack(inv, ck, passages, cases)
        if unknown and g["decision"] != "escalate":
            g.update(decision="escalate", edge="E2", safety_net=True); notes.append(f"safety net: numbers not in evidence {sorted(unknown)}")
    if g["decision"] != "escalate" and g.get("confidence", 0) < MIN_CONFIDENCE:
        g.update(decision="escalate", edge="E2"); notes.append(f'confidence {g.get("confidence")} below {MIN_CONFIDENCE}')
    if g["decision"] == "escalate" and not g.get("edge"):
        g["edge"] = "E2"
    g["guard_notes"] = notes
    g["overridden"] = bool(notes)
    return g
