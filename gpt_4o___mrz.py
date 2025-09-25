# ==== MRZ Batch v2.5 ===========================================
# Цілі:
# 1) Стабільність: Tesseract → якщо валідно по checksum → приймаємо, GPT не викликаємо.
# 2) Лише якщо слабко: GPT як fallback (1 ROI, 0°; за потреби 180°; за потреби легкий preproc).
# 3) Ранній вихід: як тільки valid_score >= 80 і doc/final пройшли — стоп.
# 4) Обмеження витрат: максимум 3-4 виклики моделі на файл.
# 5) Посилений system prompt, щоб уникати «вигаданих» GBR/UTO і т.п.
# 6) Ротація й препроцесинг — суворо умовні.

!pip -q install --upgrade openai pillow pytesseract opencv-python-headless
!apt -qq update && apt -qq install -y tesseract-ocr >/dev/null

from google.colab import files
from openai import OpenAI
from PIL import Image, ImageOps, UnidentifiedImageError
import io, base64, getpass, json, re, datetime, time, csv, os, sys
import pytesseract
import numpy as np
import cv2

# ================== CONFIG ==================
MODEL = "gpt-4o"
TEMPERATURE = 0

JPEG_QUALITY = 90
CROP_BOTTOM = 0.30                 # нижні 30% (ROI для MRZ)
DEBUG_PRINT = True
PRINT_RAW_FIRST_N = 3

OUT_CSV = "/content/mrz_batch_summary.csv"
OUT_JSONL = "/content/mrz_batch_logs.jsonl"
OUT_METRICS = "/content/mrz_gt_metrics.json"

# ---- Стратегія спроб ----
EARLY_STOP_SCORE = 80              # як тільки >=80 і doc/final пройшли — стоп
MAX_GPT_TRIES = 3                  # максимум викликів GPT на файл (base0, rot180, preproc0)
USE_ROTATION_180_IF_WEAK = True    # дозволяти 180° тільки якщо слабко
USE_PREPROC_IF_WEAK = True         # дозволяти легкий preproc тільки якщо слабко

# Пороги «слабко»:
WEAK_IF_NO_PICK = True
WEAK_SCORE_BELOW = 70
REQUIRE_DOC_AND_FINAL = True       # якщо одночасно doc_pass і final_pass = False → вважаємо слабко

# ---- Формати/ОСР ----
IMG_EXTS = {".jpg",".jpeg",".png",".webp",".tif",".tiff",".bmp"}
ALLOWED = re.compile(r"^[A-Z0-9<]{20,}$")
TESS_CFG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'

# ================== УТИЛІТИ ==================
def dprint(*a, **kw):
    if DEBUG_PRINT: print(*a, **kw)

def normalize_chevrons(text: str) -> str:
    if not text: return text
    table = str.maketrans({
        "«":"<","‹":"<","⟨":"<","〈":"<",
        "»":"<","›":"<","⟩":"<","〉":"<",
        "│":"<","|":"<"
    })
    return text.translate(table)

def sanitize_keep_nl(s: str) -> str:
    if s is None: return ""
    s = s.upper()
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", s)
    s = normalize_chevrons(s)
    s = re.sub(r"[^\nA-Z0-9<]", "", s)
    return s

def sanitize(s: str) -> str:
    s = (s or "").upper().replace(" ", "")
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", s)
    s = re.sub(r"[^A-Z0-9<]", "", s)
    return s

def pad44(s: str) -> str:
    s = sanitize(s)
    if len(s) < 44: s += "<" * (44 - len(s))
    return s[:44]

def char_value(c):
    if c == '<': return 0
    if c.isdigit(): return int(c)
    return ord(c) - 55

def checksum(s):
    w = [7,3,1]
    return str(sum(char_value(c)*w[i%3] for i,c in enumerate(s)) % 10)

def yymmdd_to_ddmmyyyy(s):
    s = re.sub(r"[^0-9]", "", s or "")
    if len(s) != 6: return None
    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    century = 1900 if yy >= 50 else 2000
    try:
        return datetime.datetime(century+yy, mm, dd).strftime("%d-%m-%Y")
    except: return None

def _split_name_safely(namefld: str):
    if "<<" in namefld:
        left, right = namefld.split("<<", 1)
    elif "<" in namefld:
        m = re.search(r"<+", namefld)
        left, right = namefld[:m.start()], namefld[m.end():]
    else:
        return namefld.replace("<","") or None, None
    surname = left.replace("<","").strip() or None
    given = re.sub(r"<+", " ", right).strip()
    given = re.sub(r"\s+", " ", given) or None
    return surname, given

def _score(pass_doc, pass_dob, pass_exp, pass_pid, pass_fin):
    return (30 if pass_doc else 0) + (25 if pass_dob else 0) + (25 if pass_exp else 0) + \
           (10 if pass_pid else 0) + (10 if pass_fin else 0)

def _checks_for_l2(l2):
    doc_c = l2[9]; dob_c = l2[19]; exp_c = l2[27]; pid_c = l2[42]; fin_c = l2[43]
    pass_doc = (checksum(l2[0:9])  == doc_c)
    pass_dob = (checksum(l2[13:19])== dob_c)
    pass_exp = (checksum(l2[21:27])== exp_c)
    pass_pid = (checksum(l2[28:42])== pid_c)
    composite = l2[0:10] + l2[13:20] + l2[21:28] + l2[28:43]
    expected_final = checksum(composite)
    pass_fin = (expected_final == fin_c)
    return {
      "pass_doc": pass_doc, "pass_dob": pass_dob, "pass_exp": pass_exp,
      "pass_pid": pass_pid, "pass_fin": pass_fin, "expected_final": expected_final,
      "doc_c": doc_c, "dob_c": dob_c, "exp_c": exp_c, "pid_c": pid_c, "fin_c": fin_c
    }

def _repair_l2_candidates(l2):
    cands = [("as_is", l2)]
    if len(l2) == 44:
        # 1) swap pos8↔pos9
        if l2[9] == '<' and l2[8].isdigit():
            l2_swap = l2[:8] + '<' + l2[8] + l2[10:]
            cands.append(("swap8_9", l2_swap))
        # 2) move pos10→pos9
        if not l2[9].isdigit() and l2[10].isdigit():
            l2_fix = l2[:9] + l2[10] + l2[11:] + "<"
            cands.append(("move10_to_9", l2_fix[:44]))
    return cands

def _best_l2_by_score(l2):
    best = None
    for tag, cand in _repair_l2_candidates(l2):
        if len(cand) != 44: continue
        ch = _checks_for_l2(cand)
        sc = _score(ch["pass_doc"], ch["pass_dob"], ch["pass_exp"], ch["pass_pid"], ch["pass_fin"])
        if best is None or sc > best["score"]:
            best = {"tag": tag, "l2": cand, "checks": ch, "score": sc}
    return best

def build_json_from_lines(l1, l2, repair_meta):
    namefld = l1[5:44]
    surname, given = _split_name_safely(namefld)
    number_raw=l2[0:9]; nat=l2[10:13]; dob=l2[13:19]; sex=l2[20]; exp=l2[21:27]; pid=l2[28:42]
    ch = repair_meta["checks"]
    return {
      "0": {"status": "success"},
      "1": {"message": "Process of recognize was successfully finished."},
      "2": {"hash": None},
      "3": {
        "country": l1[2:5],
        "date_of_birth": yymmdd_to_ddmmyyyy(dob),
        "date_of_issue": None,
        "expiration_date": yymmdd_to_ddmmyyyy(exp),
        "mrz": f"{l1}\n{l2}",
        "mrz_type": "TD3",
        "names": given or "",
        "nationality": nat,
        "number": number_raw.replace("<",""),
        "photo": None,
        "sex": sex if sex in ("M","F","<") else sex,
        "surname": surname or None,
        "personal_number": pid.replace("<","") or None,
        "document_number_check": ch["doc_c"],
        "birth_check": ch["dob_c"],
        "expiry_check": ch["exp_c"],
        "personal_number_check": ch["pid_c"],
        "final_check": ch["fin_c"],
        "document_number_pass": ch["pass_doc"],
        "birth_pass": ch["pass_dob"],
        "expiry_pass": ch["pass_exp"],
        "personal_number_pass": ch["pass_pid"],
        "final_pass": ch["pass_fin"],
        "valid_score": _score(ch["pass_doc"], ch["pass_dob"], ch["pass_exp"], ch["pass_pid"], ch["pass_fin"])
      }
    }

# ================== МОДЕЛЬ/ОСР ==================
api_key = getpass.getpass("Enter your OpenAI API key: ")
client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = (
    "You are an OCR specialized ONLY in ICAO 9303 MRZ (TD3). "
    "Return EXACTLY two lines, each EXACTLY 44 characters, using ONLY A-Z, 0-9, and '<'. "
    "UPPERCASE. No spaces. No extra text. No code fences. Do not invent example data. "
    "If a character is unreadable, use '<'. Do not substitute plausible samples like 'UTO' or 'GBR' "
    "unless they are clearly visible in the image. Keep line breaks between the two lines."
)

def call_gpt4o_mrz(data_url, system_txt=SYSTEM_PROMPT):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role":"system","content":system_txt},
            {"role":"user","content":[
                {"type":"text","text":"Extract the two MRZ lines (TD3)."},
                {"type":"image_url","image_url":{"url": data_url}}
            ]}
        ]
    )
    t1 = time.time()
    out = {
        "raw": (resp.choices[0].message.content or "").strip(),
        "latency_s": round(t1 - t0, 3),
        "tokens": {}
    }
    try:
        out["tokens"] = {
            "prompt": resp.usage.prompt_tokens,
            "completion": resp.usage.completion_tokens,
            "total": resp.usage.total_tokens
        }
    except: pass
    return out

def pick_two_lines_from_text(raw_text: str):
    txt = sanitize_keep_nl(raw_text)
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    cands = []
    for ln in lines:
        cands.extend(re.findall(r"[A-Z0-9<]{20,}", ln))
    if not cands:
        return None, []
    cand_sorted = sorted(cands, key=len, reverse=True)[:8]
    l1 = next((c for c in cand_sorted if c.startswith("P<")), None)
    l2 = next((c for c in cand_sorted if c is not l1 and not c.startswith("P<")), None)
    if not l1 or not l2:
        if len(cand_sorted) >= 2:
            l1, l2 = cand_sorted[0], cand_sorted[1]
        else:
            return None, cands
    return [l1, l2], cands

# ================== IMAGE PIPE ==================
def pil_to_jpeg_data_url(pil_img, quality=JPEG_QUALITY):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
    jpg = buf.getvalue()
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode(), jpg

def make_roi(img: Image.Image):
    W, H = img.size
    y0 = int(H * (1 - CROP_BOTTOM))
    roi = img.crop((0, y0, W, H))
    return roi, (W, H, y0)

def preprocess_for_mrz(pil_img: Image.Image) -> Image.Image:
    # grayscale -> gaussian blur -> adaptive threshold -> autocontrast
    arr = np.array(pil_img.convert("L"))
    arr = cv2.GaussianBlur(arr, (3,3), 0)
    thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 21, 7)
    out = Image.fromarray(thr)
    out = ImageOps.autocontrast(out)
    return out

def rotate_180(img: Image.Image) -> Image.Image:
    return img.rotate(180, expand=True, resample=Image.BICUBIC)

# ================== НОРМАЛІЗАЦІЯ/ОЦІНКА ==================
def _normalize_and_score(picked, fallback_note=None):
    repair_meta = None
    norm = None
    issue_tags = []
    result_json = None

    if picked:
        l1 = pad44(picked[0]); l2 = pad44(picked[1])

        # м'яке вирівнювання nationality до pos10
        def is_alpha3(s): return len(s) == 3 and s.isalpha()
        nat_at = None
        for i in range(8, 16):
            if is_alpha3(l2[i:i+3]): nat_at = i; break
        if nat_at is not None:
            nat_shift = 10 - nat_at
            if nat_shift > 0: l2 = l2[:8] + ("<"*nat_shift) + l2[8:]
            elif nat_shift < 0:
                k = min(-nat_shift, max(0, len(l2)-8))
                l2 = l2[:8] + l2[8+k:]
            l2 = (l2 + "<"*44)[:44]

        # pos10 → pos9 (частий OCR-сплив)
        if not l2[9].isdigit() and len(l2) >= 11 and l2[10].isdigit():
            l2 = (l2[:9] + l2[10] + l2[11:] + "<")[:44]

        norm = [pad44(l1), pad44(l2)]

        # попередження про одношевронні імена
        namefld = norm[0][5:44]
        if "<<" not in namefld and "<" in namefld:
            issue_tags.append("name_single_chevron")

        # найкращий L2 за checksum-score
        best = _best_l2_by_score(norm[1])
        if best:
            if best["tag"] != "as_is": issue_tags.append("l2_repaired_"+best["tag"])
            repair_meta = {"strategy": best["tag"], "score": best["score"], "checks": best["checks"],
                           "before": norm[1], "after": best["l2"], "fallback": fallback_note or ""}
            norm[1] = best["l2"]
            result_json = build_json_from_lines(norm[0], norm[1], repair_meta)

    if result_json is None:
        result_json = {
          "0":{"status":"success"},
          "1":{"message":"Process of recognize was successfully finished."},
          "2":{"hash":None},
          "3": {k: None for k in [
              "country","date_of_birth","date_of_issue","expiration_date","mrz","mrz_type",
              "names","nationality","number","photo","sex","surname",
              "personal_number","document_number_check","birth_check","expiry_check",
              "personal_number_check","final_check","document_number_pass","birth_pass",
              "expiry_pass","personal_number_pass","final_pass","valid_score"
          ]}
        }

    out3 = result_json.get("3", {})
    valid_score = out3.get("valid_score")
    pass_doc = out3.get("document_number_pass")
    pass_fin = out3.get("final_pass")
    return result_json, norm, issue_tags, valid_score, pass_doc, pass_fin

def _weak(row_core):
    if row_core is None: return True
    if WEAK_IF_NO_PICK and not row_core.get("picked"): return True
    if REQUIRE_DOC_AND_FINAL and (row_core.get("doc_pass") is False and row_core.get("fin_pass") is False):
        return True
    vs = row_core.get("valid_score")
    if vs is None: return True
    return vs < WEAK_SCORE_BELOW

def _good_enough(row_core):
    if row_core is None: return False
    vs = row_core.get("valid_score") or 0
    return (vs >= EARLY_STOP_SCORE) and (row_core.get("doc_pass") is True or row_core.get("fin_pass") is True)

def run_gpt_on_roi(pil_img, idx, total, print_raw=True):
    roi, (W,H,y0) = make_roi(pil_img)
    roi = ImageOps.autocontrast(roi.convert("L"))
    data_url, jpg = pil_to_jpeg_data_url(roi)
    model_res = call_gpt4o_mrz(data_url)
    raw = model_res["raw"]
    if idx <= PRINT_RAW_FIRST_N and print_raw:
        print(f"\n🔎 Raw[{idx}/{total}] {pil_img.info.get('__debug','')}:\n{raw[:200]}{'...' if len(raw)>200 else ''}")

    picked, _ = pick_two_lines_from_text(raw)
    fallback = None
    if not picked:
        t_text = pytesseract.image_to_string(roi, config=TESS_CFG)
        picked_t, _ = pick_two_lines_from_text(t_text)
        if picked_t:
            picked = [pad44(picked_t[0]), pad44(picked_t[1])]
            fallback = "tesseract"

    result_json, norm, issue_tags, valid_score, pass_doc, pass_fin = _normalize_and_score(picked, fallback)

    row_core = {
        "api_s": model_res["latency_s"],
        "jpeg_kb": round(len(jpg)/1024,1),
        "tokens_total": model_res.get("tokens",{}).get("total"),
        "picked": 1 if picked else 0,
        "fallback": fallback or "",
        "valid_score": valid_score,
        "doc_pass": pass_doc,
        "fin_pass": pass_fin,
        "issues": "|".join(issue_tags) if issue_tags else ""
    }
    meta = {
        "image": {"orig_px":[W,H], "crop_from_y": y0, "crop_px":[roi.size[0], roi.size[1]], "jpeg_kb": round(len(jpg)/1024,1)},
        "model": {"name": MODEL, "temperature": TEMPERATURE, **model_res.get("tokens",{})},
        "latency_s": {"api": model_res["latency_s"]},
        "raw": raw, "picked": picked, "norm": norm,
        "issues": issue_tags, "fallback": fallback or ""
    }
    return row_core, meta, result_json

# ================== ПАЙПЛАЙН НА ФАЙЛ ==================
def set_dbg(pil, tag): pil.info["__debug"] = tag; return pil

def try_tesseract_first(base_rgb):
    roi, _ = make_roi(base_rgb)
    roi = ImageOps.autocontrast(roi.convert("L"))
    t_text = pytesseract.image_to_string(roi, config=TESS_CFG)
    picked, _ = pick_two_lines_from_text(t_text)
    if not picked: return None, None
    result_json, norm, issue_tags, valid_score, pass_doc, pass_fin = _normalize_and_score(picked, "tesseract-first")
    row_core = {
        "api_s": 0.0,
        "jpeg_kb": None,
        "tokens_total": 0,
        "picked": 1 if picked else 0,
        "fallback": "tesseract",
        "valid_score": valid_score,
        "doc_pass": pass_doc,
        "fin_pass": pass_fin,
        "issues": "|".join(issue_tags) if issue_tags else ""
    }
    meta = {"tesseract_first": True, "raw": t_text, "picked": picked, "norm": norm}
    return row_core, (meta, result_json)

def process_one_v25(fname, data_bytes, idx, total):
    T0 = time.time()
    try:
        base_img = Image.open(io.BytesIO(data_bytes)).convert("RGB")
    except UnidentifiedImageError:
        empty_meta = {
            "file": fname, "api_s": 0, "total_s": 0, "jpeg_kb": 0, "tokens_total": 0,
            "picked": 0, "fallback": "", "repair_strategy": None, "valid_score": None,
            "doc_pass": None, "fin_pass": None, "issues": "not_an_image"
        }
        return empty_meta, {"file": fname, "error": "UnidentifiedImageError"}, {
            "0":{"status":"success"},"1":{"message":"Skipped non-image"},"2":{"hash":None},
            "3": {k: None for k in [
                "country","date_of_birth","date_of_issue","expiration_date","mrz","mrz_type",
                "names","nationality","number","photo","sex","surname",
                "personal_number","document_number_check","birth_check","expiry_check",
                "personal_number_check","final_check","document_number_pass","birth_pass",
                "expiry_pass","personal_number_pass","final_pass","valid_score"
            ]}
        }

    # 0) Tesseract-first
    t_row, t_pack = try_tesseract_first(base_img)
    if t_row and t_row["picked"] and t_row["valid_score"]:
        if _good_enough(t_row):
            T1 = time.time()
            out_row = {
                "file": fname, "api_s": 0.0, "total_s": round(T1-T0,3), "jpeg_kb": None, "tokens_total": 0,
                "picked": 1, "fallback": "tesseract", "repair_strategy": "as_is",
                "valid_score": t_row["valid_score"], "doc_pass": t_row["doc_pass"], "fin_pass": t_row["fin_pass"],
                "issues": t_row["issues"], "rotation_used": 0, "preproc_used": 0, "extra_tries": 0
            }
            meta_final = {"file": fname, "picked_variant": "tesseract", **(t_pack[0] or {})}
            return out_row, meta_final, t_pack[1]

    # 1) GPT base 0°
    tries = 0
    base0 = set_dbg(base_img.copy(), f"{fname} | base0")
    row_core, meta_core, result_json = run_gpt_on_roi(base0, idx, total, print_raw=True)
    best = {"row": row_core, "meta": meta_core, "json": result_json, "label": "base0", "rot": 0, "pre": 0}
    tries += 1
    if _good_enough(best["row"]) or tries >= MAX_GPT_TRIES:
        T1 = time.time()
        return {
            "file": fname, "api_s": best["row"]["api_s"], "total_s": round(T1-T0,3),
            "jpeg_kb": best["row"]["jpeg_kb"], "tokens_total": best["row"]["tokens_total"],
            "picked": best["row"]["picked"], "fallback": best["row"]["fallback"],
            "repair_strategy": "as_is", "valid_score": best["row"]["valid_score"],
            "doc_pass": best["row"]["doc_pass"], "fin_pass": best["row"]["fin_pass"],
            "issues": best["row"]["issues"], "rotation_used": best["rot"], "preproc_used": best["pre"],
            "extra_tries": tries-1
        }, {"file": fname, "picked_variant": best["label"], **best["meta"]}, best["json"]

    # 2) Якщо слабко → 180°
    def better(a,b):
        av = a["row"]["valid_score"] or 0
        bv = b["row"]["valid_score"] or 0
        return a if av >= bv else b

    if USE_ROTATION_180_IF_WEAK and _weak(best["row"]) and tries < MAX_GPT_TRIES:
        rot180 = set_dbg(rotate_180(base_img), f"{fname} | rot180")
        r_row, r_meta, r_json = run_gpt_on_roi(rot180, idx, total, print_raw=False)
        cand = {"row": r_row, "meta": r_meta, "json": r_json, "label": "rot180", "rot": 180, "pre": 0}
        best = better(best, cand)
        tries += 1
        if _good_enough(best["row"]) or tries >= MAX_GPT_TRIES:
            T1 = time.time()
            return {
                "file": fname, "api_s": best["row"]["api_s"], "total_s": round(T1-T0,3),
                "jpeg_kb": best["row"]["jpeg_kb"], "tokens_total": best["row"]["tokens_total"],
                "picked": best["row"]["picked"], "fallback": best["row"]["fallback"],
                "repair_strategy": "as_is", "valid_score": best["row"]["valid_score"],
                "doc_pass": best["row"]["doc_pass"], "fin_pass": best["row"]["fin_pass"],
                "issues": best["row"]["issues"], "rotation_used": best["rot"], "preproc_used": best["pre"],
                "extra_tries": tries-1
            }, {"file": fname, "picked_variant": best["label"], **best["meta"]}, best["json"]

    # 3) Якщо все ще слабко → легкий препроцесинг 0°
    if USE_PREPROC_IF_WEAK and _weak(best["row"]) and tries < MAX_GPT_TRIES:
        pre0 = set_dbg(preprocess_for_mrz(base_img), f"{fname} | pre0")
        p_row, p_meta, p_json = run_gpt_on_roi(pre0, idx, total, print_raw=False)
        pcand = {"row": p_row, "meta": p_meta, "json": p_json, "label": "pre0", "rot": 0, "pre": 1}
        best = better(best, pcand)
        tries += 1

    T1 = time.time()
    out_row = {
        "file": fname, "api_s": best["row"]["api_s"], "total_s": round(T1-T0,3),
        "jpeg_kb": best["row"]["jpeg_kb"], "tokens_total": best["row"]["tokens_total"],
        "picked": best["row"]["picked"], "fallback": best["row"]["fallback"],
        "repair_strategy": "as_is", "valid_score": best["row"]["valid_score"],
        "doc_pass": best["row"]["doc_pass"], "fin_pass": best["row"]["fin_pass"],
        "issues": best["row"]["issues"], "rotation_used": best["rot"], "preproc_used": best["pre"],
        "extra_tries": tries-1
    }
    meta_final = {"file": fname, "picked_variant": best["label"], **best["meta"]}
    return out_row, meta_final, best["json"]

# ================== GT/IO ==================
def canon_key(name: str) -> str:
    base = os.path.splitext(name)[0]
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    return base

def is_image_name(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in IMG_EXTS

def norm_text(v):
    if v is None: return None
    v = str(v).upper().strip()
    v = re.sub(r"\s+", " ", v)
    return v

def norm_date(v):
    if v is None or str(v).strip()=="":
        return None
    s = str(v).strip()
    m = re.match(r"^(\d{2})-(\d{2})-(\d{4})$", s)
    if m: return s
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m: return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    ds = re.sub(r"\D","",s)
    if len(ds)==8:
        return f"{ds[6:8]}-{ds[4:6]}-{ds[0:4]}"
    return s

def norm_mrz(v):
    if v is None: return None
    s = v.replace("\r","").strip()
    s = "\n".join(x.strip() for x in s.split("\n"))
    return s

def equal(a,b):
    return (a is None and b is None) or (a == b)

def update_prf(stats, field, pred, gt):
    S = stats.setdefault(field, {"TP":0,"FP":0,"FN":0,"TOTAL":0})
    S["TOTAL"] += 1
    if gt in [None,""] and (pred in [None,""]): return
    if gt not in [None,""] and equal(pred, gt): S["TP"] += 1
    elif pred not in [None,""] and gt not in [None,""] and not equal(pred, gt): S["FP"] += 1
    elif pred in [None,""] and gt not in [None,""]: S["FN"] += 1
    else:
        if pred not in [None,""] and gt in [None,""]: S["FP"] += 1

def prf_report(S):
    P = S["TP"] / (S["TP"] + S["FP"]) if (S["TP"]+S["FP"])>0 else None
    R = S["TP"] / (S["TP"] + S["FN"]) if (S["TP"]+S["FN"])>0 else None
    A = S["TP"] / max(1,(S["TP"]+S["FP"]+S["FN"]))
    return {"precision": P, "recall": R, "accuracy": A, "support": (S["TP"]+S["FP"]+S["FN"])}

# ================== RUN ==================
uploaded = files.upload()
file_map = dict(uploaded)

# GT завантаження
gt_key = None
for k in list(file_map.keys()):
    if re.match(r"(?i)^ground_truth(\s*\(\d+\))?\.csv$", k.strip()):
        gt_key = k
        break

gt_rows_raw = {}
if gt_key:
    gt_bytes = file_map.pop(gt_key)
    s = gt_bytes.decode("utf-8", errors="ignore")
    reader = csv.DictReader(s.splitlines())
    for row in reader:
        fn = (row.get("file") or "").strip()
        if not fn: continue
        gt_rows_raw[fn] = {k: (row.get(k) if row.get(k) not in ["", None] else None) for k in row.keys()}
    dprint(f"🟢 Ground truth loaded: {len(gt_rows_raw)} rows from '{gt_key}'")
else:
    dprint("ℹ️ No ground_truth*.csv detected; GT-mode disabled.")

# лише зображення
non_images = [k for k in file_map.keys() if not is_image_name(k)]
for k in non_images: dprint(f"⏭️ Skipping non-image file: {k}")
image_files = [k for k in file_map.keys() if is_image_name(k)]
print(f"📦 Images: {len(image_files)}")

# індекси GT
def build_gt_indexes(gt_rows_raw):
    gt_by_file = {}
    gt_by_file_name = {}
    for k, row in gt_rows_raw.items():
        gt_by_file[canon_key(k)] = row
        fn = row.get("file_name")
        if fn: gt_by_file_name[fn] = row
    return gt_by_file, gt_by_file_name

gt_by_file, gt_by_file_name = build_gt_indexes(gt_rows_raw)

# CSV заголовок
csv_fields = [
    "file","api_s","total_s","jpeg_kb","tokens_total",
    "picked","fallback","repair_strategy","valid_score","doc_pass","fin_pass",
    "issues","rotation_used","preproc_used","extra_tries"
]
GT_FIELDS = ["country","date_of_birth","expiration_date","mrz","mrz_type","names","nationality","number","sex","surname"]
if gt_rows_raw:
    for f in GT_FIELDS:
        csv_fields.extend([f"gt_{f}", f"pred_{f}", f"match_{f}"])

open(OUT_JSONL, "w").close()
with open(OUT_CSV, "w", newline="") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=csv_fields); writer.writeheader()

rows, metas, json_results = [], [], {}
prf_stats = {f: {"TP":0,"FP":0,"FN":0,"TOTAL":0} for f in GT_FIELDS}

for i, fname in enumerate(image_files, 1):
    row, meta, mrz_json = process_one_v25(fname, file_map[fname], i, len(image_files))
    rows.append(row); metas.append(meta); json_results[fname] = mrz_json

    if gt_rows_raw:
        key = canon_key(fname)
        gt_row = gt_by_file.get(key) or gt_by_file_name.get(key)
        if gt_row:
            out3 = mrz_json.get("3", {})
            normers = {
                "country": norm_text, "date_of_birth": norm_date, "expiration_date": norm_date,
                "mrz": norm_mrz, "mrz_type": norm_text, "names": norm_text, "nationality": norm_text,
                "number": norm_text, "sex": norm_text, "surname": norm_text
            }
            for f in GT_FIELDS:
                gt_v  = normers[f](gt_row.get(f)) if f in gt_row else None
                pred_v= normers[f](out3.get(f))
                row[f"gt_{f}"]   = gt_v
                row[f"pred_{f}"] = pred_v
                row[f"match_{f}"]= 1 if equal(pred_v, gt_v) else 0
                update_prf(prf_stats, f, pred_v, gt_v)
        else:
            row["issues"] = (row.get("issues","") + "|no_gt_match").strip("|")

    with open(OUT_CSV, "a", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=csv_fields); writer.writerow(row)

    with open(OUT_JSONL, "a") as fj:
        fj.write(json.dumps({"file": fname, "meta": meta, "result": mrz_json}, ensure_ascii=False) + "\n")

# ==== Summary ====
print("\n✅ DONE. Batch summary:")
print(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"\n📄 CSV: {OUT_CSV}\n🧾 JSONL: {OUT_JSONL}")

if gt_rows_raw:
    print("\n📊 Ground Truth metrics per field:")
    metrics = {}
    for f, S in prf_stats.items():
        rep = prf_report(S); metrics[f] = rep
        P = '—' if rep['precision'] is None else f"{rep['precision']:.3f}"
        R = '—' if rep['recall'] is None else f"{rep['recall']:.3f}"
        print(f"{f:16s} → acc={rep['accuracy']:.3f}  prec={P}  recall={R}  support={rep['support']}")
    with open(OUT_METRICS, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n📈 Metrics saved to: {OUT_METRICS}")
