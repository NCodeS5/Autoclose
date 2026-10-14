"""Optional Salesforce link: an escalated invoice becomes a Case; the decision updates and closes it.

Switch it on with SF_DOMAIN (your My Domain URL), SF_CLIENT_ID and SF_CLIENT_SECRET (client credentials flow).
Without them this module does nothing. A Salesforce failure never stops a review: errors are recorded on the row."""
import os, time, threading
import requests

API = os.getenv("SF_API_VERSION", "v60.0")

class SalesforceCases:
    def __init__(self, domain=None, client_id=None, client_secret=None):
        self.domain = (domain or os.getenv("SF_DOMAIN", "")).strip().rstrip("/")
        if self.domain and not self.domain.startswith("http"): self.domain = "https://" + self.domain
        self.cid = client_id or os.getenv("SF_CLIENT_ID", "")
        self.secret = client_secret or os.getenv("SF_CLIENT_SECRET", "")
        self.enabled = bool(self.domain and self.cid and self.secret)
        # Optional custom fields on Case (you create them in Setup; see README). Off unless SF_CUSTOM_FIELDS=true.
        self.custom = os.getenv("SF_CUSTOM_FIELDS", "").strip().lower() in ("1", "true", "yes")
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
            inv = row["invoice"]; amt = float(inv.get("amount") or 0)
            desc = (f"Invoice {inv.get('invoice_no')} from {inv.get('vendor')} for ${amt:,.2f}\n{inv.get('description', '')}\n\n"
                    f"Agent proposed: {row.get('proposed') or row.get('decision')} (edge {row.get('edge')}, confidence {row.get('confidence')})\n"
                    f"Rationale: {row.get('rationale')}\nCited: {', '.join(row.get('citations') or [])}\nReview id: {row['case_id']}")
            body = {
                "Subject": f"Invoice review {row['case_id']}: {inv.get('vendor')} ${amt:,.2f} ({row.get('edge') or 'audit sample'})",
                "Description": desc[:30000], "Origin": "Web", "Priority": "High" if row.get("edge") == "E1" else "Medium", "Status": "New"}
            used_custom = False
            if self.custom:
                extra = {"Autoclose_Review_Id__c": str(row["case_id"]), "Agent_Edge__c": str(row.get("edge") or "audit sample"),
                         "Agent_Confidence__c": float(row.get("confidence") or 0),
                         "Agent_Proposal__c": str(row.get("proposed") or row.get("decision") or "")[:40]}
                try:
                    out = self._call("POST", "/sobjects/Case", json={**body, **extra}); used_custom = True
                except Exception:
                    out = self._call("POST", "/sobjects/Case", json=body)       # a missing field never loses the Case
            else:
                out = self._call("POST", "/sobjects/Case", json=body)
            cid = out["id"]
            num = self._call("GET", f"/sobjects/Case/{cid}?fields=CaseNumber").get("CaseNumber")
            host = self.domain.replace(".my.salesforce.com", ".lightning.force.com")
            return {"id": cid, "number": num, "url": f"{host}/lightning/r/Case/{cid}/view", "status": "New", "custom_fields": used_custom}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:200]}
    def close_case(self, sf, resolution, note="", who="reviewer"):
        """Close the Case and leave a comment saying who decided. Never raises."""
        if not self.enabled or not sf or not sf.get("id"): return sf
        try:
            word = "Approved" if resolution == "benign" else "Rejected"
            if sf.get("custom_fields"):
                try: self._call("PATCH", f"/sobjects/Case/{sf['id']}", json={"Status": "Closed", "Final_Decision__c": word})
                except Exception: self._call("PATCH", f"/sobjects/Case/{sf['id']}", json={"Status": "Closed"})
            else:
                self._call("PATCH", f"/sobjects/Case/{sf['id']}", json={"Status": "Closed"})
            self._call("POST", "/sobjects/CaseComment", json={"ParentId": sf["id"], "CommentBody": f"{word} by {who}. {note}".strip()})
            return {**sf, "status": "Closed", "resolution": word}
        except Exception as e:
            return {**sf, "error": f"{type(e).__name__}: {e}"[:200]}
