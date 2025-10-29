
import io
import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import numpy as np

try:
    from icalendar import Calendar
except Exception:
    Calendar = None

# Optional: S3 (Cloudflare R2) upload
S3_ENABLED = False
try:
    import boto3  # type: ignore
    S3_ENABLED = True
except Exception:
    S3_ENABLED = False

TMP_DIR = "/tmp/khl_json"
os.makedirs(TMP_DIR, exist_ok=True)

app = FastAPI(title="KHL PDF Parser (Cloud)", version="1.1.0")
app.mount("/files", StaticFiles(directory=TMP_DIR), name="files")

# -------------------- Utilities --------------------

DASHES = r"[–—\-]"
TEAM_SEP = re.compile(rf"\s{DASHES}\s")

RE_DATE = re.compile(r"(\d{2}[./-]\d{2}[./-]\d{4})")
RE_TIME = re.compile(r"(\d{2}:\d{2})")
RE_MATCH_NO = re.compile(r"(?:Матч\s*№|№\s*матча)\s*([0-9]+)", re.IGNORECASE)
RE_ARENA = re.compile(r"(?:ДС|ЛД|Арена|Дворец спорта)\s*[«\"]?([^»\"\n]+)[»\"]?")
RE_REFEREES = re.compile(r"Главн(?:ые)?\s+судьи?\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
RE_LINESMEN = re.compile(r"Линейн(?:ые|ые судьи)\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)
RE_TEAMS_LINE = re.compile(rf"^\s*([A-Za-zА-Яа-яЁё .\"«»\-]+)\s{DASHES}\s([A-Za-zА-Яа-яЁё .\"«»\-]+)\s*$")

# Goalies parsing heuristics
RE_PLAYER_LINE = re.compile(r"^\s*#?\s*(\d{1,2})\s+([A-Za-zА-Яа-яЁё\-]+)\s+([A-Za-zА-Яа-яЁё\-]+)(?:\s*\((С|Р)\))?\s*(?:\b(С|Р)\b)?", re.IGNORECASE)

def norm_text(txt: str) -> str:
    txt = txt.replace("\xa0", " ").replace("\u2009", " ").replace("\u202f", " ")
    return re.sub(r"[ \t]+", " ", txt)

def extract_text_mupdf(pdf_bytes: bytes) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    pages_words = []
    for page in doc:
        pages_text.append(page.get_text("text"))
        words = page.get_text("words")
        pages_words.append(words)
    doc.close()
    return {"pages_text": pages_text, "pages_words": pages_words}

def ocr_pdf(pdf_bytes: bytes, dpi: int = 320, max_pages: int = 2) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    all_txt = []
    for im in images[:max_pages]:
        txt = pytesseract.image_to_string(im, lang="rus+eng")
        all_txt.append(txt)
    return "\n".join(all_txt)

def guess_teams_from_header(lines: List[str]) -> Optional[List[str]]:
    for ln in lines[:50]:
        m = RE_TEAMS_LINE.search(ln.strip())
        if m:
            t1 = m.group(1).strip(' "«»')
            t2 = m.group(2).strip(' "«»')
            return [t1, t2]
    return None

def parse_goalies(lines: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    sections = []
    idxs = [i for i, ln in enumerate(lines) if re.search(r"\bВратар[ьи]\b", ln, re.IGNORECASE)]
    if not idxs:
        return {"home": [], "away": []}

    for idx in idxs:
        block = []
        for j in range(idx+1, min(idx+22, len(lines))):
            txt = lines[j].strip()
            if not txt or re.search(r"^(\w+\s*:)", txt):  # new section
                break
            block.append(txt)

        sections.append(block)

    def parse_block(block: List[str]) -> List[Dict[str, Any]]:
        res = []
        for ln in block:
            m = RE_PLAYER_LINE.search(ln)
            if m:
                num = int(m.group(1))
                last = m.group(2).title()
                first = m.group(3).title()
                stat = (m.group(4) or m.group(5) or "").upper()
                status = {"С":"starter","Р":"reserve"}.get(stat, "")
                res.append({"number": num, "name": f"{last} {first}", "status": status})
        return res

    home_blk = sections[0] if len(sections) >= 1 else []
    away_blk = sections[1] if len(sections) >= 2 else []
    return {"home": parse_block(home_blk), "away": parse_block(away_blk)}

def parse_lines(lines: List[str]) -> Dict[str, Any]:
    """
    Heuristic extraction of lines/units.
    Looks for sections starting with 'Состав' or 'Звено' or 'Пятерка'.
    Returns structure: {home: {lines: [...]}, away: {...}}
    """
    result = {"home": {"lines": []}, "away": {"lines": []}}
    idxs = [i for i, ln in enumerate(lines) if re.search(r"\b(Состав|Звено|Пятерка)\b", ln, re.IGNORECASE)]
    blocks = []
    for idx in idxs:
        block = []
        for j in range(idx, min(idx+28, len(lines))):
            t = lines[j].strip()
            if not t:
                break
            block.append(t)
        if block:
            blocks.append(block)

    def parse_block_to_lines(block: List[str]) -> List[str]:
        acc = []
        for ln in block:
            if re.search(r"\b(Звено|Пятерка)\s*\d", ln, re.IGNORECASE):
                acc.append(ln)
            elif re.search(r"^\s*(LW|RW|C|D|LD|RD|ЛВ|ПВ|Ц|Н|З)\b", ln, re.IGNORECASE):
                acc.append(ln)
            elif re.search(r"#\d{1,2}\s+[А-Яа-яA-Za-z\-]+", ln):
                acc.append(ln)
        return acc[:20]

    parsed_blocks = [parse_block_to_lines(b) for b in blocks]
    if parsed_blocks:
        result["home"]["lines"] = parsed_blocks[0]
    if len(parsed_blocks) > 1:
        result["away"]["lines"] = parsed_blocks[1]
    return result

def parse_pdf_text(pages_text: List[str], pages_words: List[list]) -> Dict[str, Any]:
    full_text = norm_text("\n".join(pages_text))
    lines = [norm_text(l) for l in full_text.splitlines() if l.strip()]

    teams = guess_teams_from_header(lines) or [None, None]

    date = None
    time_ = None
    for ln in lines[:100]:
        if date is None:
            m = RE_DATE.search(ln)
            if m:
                date = m.group(1).replace("/", ".").replace("-", ".")
        if time_ is None:
            m2 = RE_TIME.search(ln)
            if m2:
                time_ = m2.group(1)

    match_no = None
    m = RE_MATCH_NO.search(full_text)
    if m:
        match_no = int(m.group(1))

    arena = None
    ma = RE_ARENA.search(full_text)
    if ma:
        arena = ma.group(0)

    referees: List[str] = []
    linesmen: List[str] = []
    mr = RE_REFEREES.search(full_text)
    if mr:
        names_line = mr.group(1)
        refs = re.split(r"[;,]", names_line)
        referees = [r.strip(" .") for r in refs if r.strip()]
    ml = RE_LINESMEN.search(full_text)
    if ml:
        names_line = ml.group(1)
        lns = re.split(r"[;,]", names_line)
        linesmen = [r.strip(" .") for r in lns if r.strip()]

    goalies = parse_goalies(lines)
    lineup = parse_lines(lines)

    return {
        "date": date,
        "start_time_msk": time_,
        "arena": arena,
        "teams": teams,
        "match_number": match_no,
        "goalies": goalies,
        "main_referees": referees,
        "linesmen": linesmen,
        "lineups": lineup
    }

def build_pdf_url(uid: int, season: int = 1369) -> str:
    return f"https://www.khl.ru/pdf/{season}/{uid}/game-{uid}-start-ru.pdf"

def save_json_local(uid: int, data: dict) -> str:
    out_path = os.path.join(TMP_DIR, f"{uid}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path

def upload_r2(uid: int, data: dict) -> Optional[str]:
    if not S3_ENABLED:
        return None
    acc = os.getenv("R2_ACCOUNT_ID")
    bucket = os.getenv("R2_BUCKET")
    key_id = os.getenv("R2_ACCESS_KEY_ID")
    secret = os.getenv("R2_SECRET_ACCESS_KEY")
    if not all([acc, bucket, key_id, secret]):
        return None
    endpoint_url = f"https://{acc}.r2.cloudflarestorage.com"
    s3 = boto3.client("s3",
                      endpoint_url=endpoint_url,
                      aws_access_key_id=key_id,
                      aws_secret_access_key=secret)
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=f"khl/json/{uid}.json", Body=body, ContentType="application/json; charset=utf-8")
    public_base = os.getenv("R2_PUBLIC_BASE", f"https://pub.example.com")
    return f"{public_base}/khl/json/{uid}.json"

def parse_ics_to_uids(ics_bytes: bytes, window_min: int = 65) -> List[Tuple[int, datetime]]:
    if Calendar is None:
        return []
    cal = Calendar.from_ical(ics_bytes)
    now = datetime.now(timezone.utc)
    upcoming: List[Tuple[int, datetime]] = []
    for comp in cal.walk():
        if comp.name != "VEVENT":
            continue
        dtstart = comp.get("DTSTART").dt
        if not isinstance(dtstart, datetime):
            continue
        if now <= dtstart <= now + timedelta(minutes=window_min):
            uid_field = comp.get("UID")
            uid_num = None
            for field in [str(comp.get("SUMMARY","")), str(comp.get("DESCRIPTION","")), str(comp.get("URL","")), str(uid_field or "")]:
                m = re.search(r"(\d{6,})", field)
                if m:
                    uid_num = int(m.group(1)); break
            if uid_num:
                upcoming.append((uid_num, dtstart))
    return upcoming

# -------------------- API --------------------

@app.get("/health")
def health():
    return {"ok": True, "service": "khl-pdf-parser", "version": "1.1.0"}

@app.get("/extract")
async def extract(pdf_url: str = Query(..., description="Direct URL to KHL match PDF (start-ru.pdf)"),
                  uid: Optional[int] = Query(None, description="Optional UID for naming outputs"),
                  ocr_fallback: bool = Query(True, description="Use OCR if text layer is weak"),
                  publish: bool = Query(True, description="Save JSON locally and/or to R2 if configured")):
    t0 = time.time()
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        resp = await client.get(pdf_url, headers={"Referer": "https://www.khl.ru/"})
        resp.raise_for_status()
        pdf_bytes = resp.content

    mu = extract_text_mupdf(pdf_bytes)
    parsed = parse_pdf_text(mu["pages_text"], mu["pages_words"])

    weak_text = sum(len(p or "") for p in mu["pages_text"]) < 500
    if ocr_fallback and (weak_text or not parsed.get("teams")[0]):
        ocr_txt = ocr_pdf(pdf_bytes, dpi=360)
        parsed_ocr = parse_pdf_text([ocr_txt], [[]])
        for k in ["date","start_time_msk","arena","teams","goalies","main_referees","linesmen","lineups","match_number"]:
            if not parsed.get(k) or (k=="teams" and not parsed["teams"][0]):
                parsed[k] = parsed_ocr.get(k, parsed.get(k))

    out = {
        "ok": True,
        "dur_s": round(time.time() - t0, 3),
        "source_pdf_len": len(pdf_bytes),
        "data": parsed,
        "pdf_url": pdf_url,
        "uid": uid
    }

    links = {}
    if publish and uid:
        save_json_local(uid, out)
        links["local_url"] = f"/files/{uid}.json"
        r2_url = upload_r2(uid, out)
        if r2_url:
            links["r2_url"] = r2_url
    out["links"] = links
    return JSONResponse(out)

@app.get("/extract_batch")
async def extract_batch(uids: str = Query(..., description="Comma-separated UIDs"),
                        season: int = Query(1369),
                        publish: bool = Query(True)):
    uid_list = [int(x) for x in uids.split(",") if x.strip().isdigit()]
    results = []
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        for uid in uid_list:
            pdf_url = build_pdf_url(uid, season=season)
            try:
                r = await client.get(pdf_url, headers={"Referer": "https://www.khl.ru/"})
                r.raise_for_status()
                mu = extract_text_mupdf(r.content)
                parsed = parse_pdf_text(mu["pages_text"], mu["pages_words"])
                weak_text = sum(len(p or "") for p in mu["pages_text"]) < 500
                if weak_text or not parsed.get("teams")[0]:
                    ocr_txt = ocr_pdf(r.content, dpi=360)
                    parsed_ocr = parse_pdf_text([ocr_txt], [[]])
                    for k in ["date","start_time_msk","arena","teams","goalies","main_referees","linesmen","lineups","match_number"]:
                        if not parsed.get(k) or (k=="teams" and not parsed["teams"][0]):
                            parsed[k] = parsed_ocr.get(k, parsed.get(k))
                out = {"uid": uid, "ok": True, "data": parsed, "pdf_url": pdf_url}
                if publish:
                    save_json_local(uid, out)
                    out["links"] = {"local_url": f"/files/{uid}.json"}
                results.append(out)
            except Exception as e:
                results.append({"uid": uid, "ok": False, "error": str(e), "pdf_url": pdf_url})
    return JSONResponse({"ok": True, "count": len(results), "results": results})

@app.get("/cron")
async def cron(ics_url: str = Query(..., description="Public ICS URL"),
               season: int = Query(1369),
               window_min: int = Query(65, description="scan upcoming N minutes"),
               publish: bool = Query(True)):
    if Calendar is None:
        return JSONResponse({"ok": False, "error": "icalendar not installed"}, status_code=500)
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        ics = await client.get(ics_url)
        ics.raise_for_status()
        upcoming = parse_ics_to_uids(ics.content, window_min=window_min)
    uids = [uid for uid, dt in upcoming]
    if not uids:
        return {"ok": True, "message": "no upcoming matches within window", "uids": []}
    joined = ",".join(map(str, uids))
    res = await extract_batch(uids=joined, season=season, publish=publish)
    return res
