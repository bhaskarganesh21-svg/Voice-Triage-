"""Cloud storage for triage results (Supabase REST, no extra dependencies).
Set SUPABASE_URL and SUPABASE_KEY on your host. Falls back to local JSON files if unset."""
import os, json, urllib.request
from pathlib import Path

URL   = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY   = os.getenv("SUPABASE_KEY", "")
TABLE = os.getenv("SUPABASE_TABLE", "triage_results")
LOCAL = Path(os.getenv("OUTPUT_DIR", Path(__file__).parent / "output"))

def _cloud() -> bool:
    return bool(URL and KEY)

def _req(method, query="", body=None):
    req = urllib.request.Request(
        f"{URL}/rest/v1/{TABLE}{query}", method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"apikey": KEY, "Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json", "Prefer": "return=minimal"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if raw else None

def save_result(d: dict) -> None:
    if _cloud():
        try:
            _req("POST", body={"case_id": d.get("case_id"), "data": d}); return
        except Exception as e:
            print(f"[storage] cloud save failed, using local: {e}")
    LOCAL.mkdir(parents=True, exist_ok=True)
    name = f"{d.get('case_id','LIVE')}_{str(d.get('timestamp','')).replace(':','-')}.json"
    (LOCAL / name).write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def list_results(limit: int = 50) -> list:
    if _cloud():
        try:
            rows = _req("GET", f"?select=data&order=created_at.desc&limit={limit}")
            return [r["data"] for r in rows or []]
        except Exception as e:
            print(f"[storage] cloud read failed: {e}")
    files = sorted(LOCAL.glob("*.json"), reverse=True)[:limit] if LOCAL.exists() else []
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]
