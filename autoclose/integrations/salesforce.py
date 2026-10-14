"""Optional Salesforce link: an escalated invoice becomes a Case; the decision updates and closes it.

Switch it on with SF_DOMAIN (your My Domain URL), SF_CLIENT_ID and SF_CLIENT_SECRET (client credentials flow).
Without them this module does nothing. A Salesforce failure never stops a review: errors are recorded on the row."""
import os, time, threading
from urllib.parse import quote
import requests

API = os.getenv("SF_API_VERSION", "v60.0")

class SalesforceCases:
    def __init__(self, domain=None, client_id=None, client_secret=None):
        self.domain = (domain or os.getenv("SF_DOMAIN", "")).strip().rstrip("/")
        if self.domain and not self.domain.startswith("http"): self.domain = "https://" + self.domain
        self.cid = client_id or os.getenv("SF_CLIENT_ID", "")
        self.secret = client_secret or os.getenv("SF_CLIENT_SECRET", "")
        self.enabled = bool(self.domain and self.cid and self.secret)
        self._tok, self._exp, self._lock = None, 0, threading.Lock()
        self._vendors = {}
    def _token(self):
        with self._lock:
            if self._tok and time.time() < self._exp: return self._tok
            r = requests.post(self.domain + "/services/oauth2/token", timeout=15,
                              data={"grant_type": "client_credentials", "client_id": self.cid, "client_secret": self.secret})
            r.raise_for_status()
            self._tok = r.json()["access_token"]; self._exp = time.time() + 1500
            return self._tok
    def _call(self, method, path, **kw):
        h = {"Authorization": "Bearer " + self._token(), "Content-Type": "application/json"}
        r = requests.request(method, f"{self.domain}/services/data/{API}{path}", headers=h, timeout=20, **kw)
        r.raise_for_status()
        return r.json() if r.content else {}
    def _link(self, rid, obj="Account"):
        return f"{self.domain.replace('.my.salesforce.com', '.lightning.force.com')}/lightning/r/{obj}/{rid}/view"
    def find_vendor(self, name):
        """Look the vendor up as an Account (SOQL). Read only, cached for 10 minutes. Never raises."""
        if not self.enabled: return None
        hit = self._vendors.get(name)
        if hit and time.time() - hit[0] < 600: return hit[1]
        try:
            q = "SELECT Id, Name, Industry, Owner.Name FROM Account WHERE Name = '" + str(name).replace("\\", "\\\\").replace("'", "\\'") + "' LIMIT 1"
            rows = self._call("GET", "/query?q=" + quote(q)).get("records") or []
            out = ({"found": True, "id": rows[0]["Id"], "name": rows[0]["Name"], "industry": rows[0].get("Industry"),
                    "owner": (rows[0].get("Owner") or {}).get("Name"), "url": self._link(rows[0]["Id"])} if rows else {"found": False})
        except Exception as e:
            return {"found": None, "error": f"{type(e).__name__}: {e}"[:200]}
        self._vendors[name] = (time.time(), out)
        return out
    def seed_vendors(self, vendors):
        """Create the demo vendors as Accounts if they are missing. Returns {'created': [...], 'existing': [...], 'errors': [...]}."""
        res = {"created": [], "existing": [], "errors": []}
        if not self.enabled: return {**res, "errors": ["Salesforce is not configured"]}
        for name, v in vendors.items():
            self._vendors.pop(name, None)
            cur = self.find_vendor(name)
            if cur and cur.get("found"): res["existing"].append(name); continue
            if cur and cur.get("found") is None: res["errors"].append(f"{name}: {cur.get('error')}"); continue
            try:
                self._call("POST", "/sobjects/Account", json={"Name": name, "Description": f"Demo vendor for Autoclose. Category: {v.get('category')}. Usage based: {v.get('usage_based')}."})
                self._vendors.pop(name, None); res["created"].append(name)
            except Exception as e:
                res["errors"].append(f"{name}: {type(e).__name__}: {e}"[:200])
        return res
    def open_case(self, row):
        """Create a Case for an escalated invoice. Returns a small dict, or {'error': ...}. Never raises."""
        if not self.enabled: return None
        try:
            inv = row["invoice"]; amt = float(inv.get("amount") or 0)
            desc = (f"Invoice {inv.get('invoice_no')} from {inv.get('vendor')} for ${amt:,.2f}\n{inv.get('description', '')}\n\n"
                    f"Agent proposed: {row.get('proposed') or row.get('decision')} (edge {row.get('edge')}, confidence {row.get('confidence')})\n"
                    f"Rationale: {row.get('rationale')}\nCited: {', '.join(row.get('citations') or [])}\n{self._vendor_line(row)}Review id: {row['case_id']}")
            out = self._call("POST", "/sobjects/Case", json={
                "Subject": f"Invoice review {row['case_id']}: {inv.get('vendor')} ${amt:,.0f} ({row.get('edge') or 'audit sample'})",
                "Description": desc[:30000], "Origin": "Web", "Priority": "High" if row.get("edge") == "E1" else "Medium", "Status": "New"})
            cid = out["id"]
            num = self._call("GET", f"/sobjects/Case/{cid}?fields=CaseNumber").get("CaseNumber")
            host = self.domain.replace(".my.salesforce.com", ".lightning.force.com")
            return {"id": cid, "number": num, "url": f"{host}/lightning/r/Case/{cid}/view", "status": "New"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200]}
    def _vendor_line(self, row):
        v = row.get("sf_vendor")
        return f"Salesforce vendor record: {v['name']} ({v['url']})\n" if v and v.get("found") else ""
    def close_case(self, sf, resolution, note="", who="reviewer"):
        """Close the Case and leave a comment saying who decided. Never raises."""
        if not self.enabled or not sf or not sf.get("id"): return sf
        try:
            word = "Approved" if resolution == "benign" else "Rejected"
            self._call("PATCH", f"/sobjects/Case/{sf['id']}", json={"Status": "Closed"})
            self._call("POST", "/sobjects/CaseComment", json={"ParentId": sf["id"], "CommentBody": f"{word} by {who}. {note}".strip()})
            return {**sf, "status": "Closed", "resolution": word}
        except Exception as e:
            return {**sf, "error": f"{type(e).__name__}: {e}"[:200]}
