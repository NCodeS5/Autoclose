"""Turn the markdown knowledge base into retrievable chunks (one chunk per '## ID Title' section)."""
import json, pathlib, re
from dataclasses import dataclass, field

DATA = pathlib.Path(__file__).resolve().parents[2] / "data"

@dataclass
class Chunk:
    id: str
    text: str
    meta: dict = field(default_factory=dict)

def _vendor_for(title: str, vendors: list):
    for v in vendors:
        if v.split()[0].lower() in title.lower():
            return v
    return ""

def load_kb(kb_dir: pathlib.Path = None) -> list:
    kb_dir = pathlib.Path(kb_dir) if kb_dir else DATA / "kb"
    vendors = [v["vendor"] for v in json.loads((DATA / "vendor_master.json").read_text())]
    chunks = []
    for f in sorted(kb_dir.glob("*.md")):
        lines = f.read_text().splitlines()
        title = lines[0].lstrip("# ").strip()
        is_contract = f.name.startswith("contract_")
        vendor = _vendor_for(title, vendors) if is_contract else ""
        cur = None
        for ln in lines[1:]:
            m = re.match(r"##\s+([A-Z]+-\d+)\s+(.*)", ln)
            if m:
                cur = Chunk(m.group(1), "", {"doc_type": "contract" if is_contract else "policy",
                                              "vendor": vendor, "title": m.group(2), "source": f.name})
                chunks.append(cur)
            elif cur is not None and ln.strip():
                cur.text += (" " if cur.text else "") + ln.strip()
        for c in chunks:
            if not c.text.startswith(c.meta["title"]) and c.meta["source"] == f.name:
                c.text = f'{c.meta["title"]}. {c.text}' if not c.text.startswith(c.meta["title"]) else c.text
    return chunks

def load_past_cases() -> list:
    return [Chunk(c["id"], c["text"], {"vendor": c["vendor"], "outcome": c["outcome"], "title": c["id"]})
            for c in json.loads((DATA / "past_cases.json").read_text())]
