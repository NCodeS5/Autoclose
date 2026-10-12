"""Optional Salesforce link: an escalated invoice becomes a Case; the reviewer's decision updates it.

Set SF_DOMAIN (your My Domain URL), SF_CLIENT_ID and SF_CLIENT_SECRET to switch it on (client credentials flow).
Without them this module does nothing, and a Salesforce failure never stops a review.
"""
import os, time, threading
import requests

API = os.getenv("SF_API_VERSION", "v60.0")

class SalesforceCases:
    def __init__(self, domain=None, client_id=None, client_secret=None):
        self.domain = (domain or os.getenv("SF_DOMAIN", "")).rstrip("/")
        if self.domain and not self.domain.startswith("http"): self.domain = "https://" + self.domain
        self.cid = client_id or os.getenv("SF_CLIENT_ID", "")
        self.secret = client_secret or os.getenv("SF_CLIENT_SECRET", "")
        self.enabled = bool(self.domain and self.cid and self.secret)
        self._tok, self._exp, self._lock = None, 0, threading.Lock()
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
    def open_case(self, row):
        """Create a Case for an escalated invoice. Returns a small dict, or {'error': ...}. Never raises."""
        if not self.enabled: return None
        try:
            inv = row["invoice"]
            desc = (f"Invoice {inv.get('invoice_no')} from {inv.get('vendor')} for ${inv.get('amount'):,.2f}\n"
                    f"{inv.get('description', '')}\n\n"
                    f"Agent proposed: {row.get('proposed') or row.get('decision')} (edge {row.get('edge')}, confidence {row.get('confidence')})\n"
                    f"Rationale: {row.get('rationale')}\nCited: {', '.join(row.get('citations') or [])}\n"
                    f"Review id: {row['case_id']}")
            out = self._call("POST", "/sobjects/Case", json={
                "Subject": f"Invoice review {row['case_id']}: {inv.get('vendor')} ${inv.get('amount'):,.0f} ({row.get('edge') or 'audit sample'})",
                "Description": desc[:30000], "Origin": "Web", "Priority": "High" if row.get("edge") == "E1" else "Medium", "Status": "New"})
            cid = out["id"]
            num = self._call("GET", f"/sobjects/Case/{cid}?fields=CaseNumber").get("CaseNumber")
            return {"id": cid, "number": num, "url": f"{self.domain.replace('.my.salesforce.com', '.lightning.force.com')}/lightning/r/Case/{cid}/view", "status": "New"}
        except Exception as e:
            return {"error": str(e)[:200]}
    def close_case(self, sf, resolution, note=""):
        if not self.enabled or not sf or not sf.get("id"): return sf
        try:
            word = "Approved" if resolution == "benign" else "Rejected"
            self._call("PATCH", f"/sobjects/Case/{sf['id']}", json={"Status": "Closed"})
            self._call("POST", "/sobjects/CaseComment", json={"ParentId": sf["id"], "CommentBody": f"{word} by reviewer. {note}".strip()})
            return {**sf, "status": "Closed", "resolution": word}
        except Exception as e:
            return {**sf, "error": str(e)[:200]}
