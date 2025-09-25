# ==== MRZ Batch v2.5 ===========================================
# Цілі:
# 1) Стабільність: Tesseract → якщо валідно по checksum → приймаємо, GPT не викликаємо.
# 2) Лише якщо слабко: GPT як fallback (1 ROI, 0°; за потреби легкий preproc).
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

# ---- Вартість ----
COST_PER_1K_PROMPT = 0.005         # $5 за 1M prompt-токенів (GPT-4o)
COST_PER_1K_COMPLETION = 0.015     # $15 за 1M completion-токенів (GPT-4o)

OUT_CSV = "/content/mrz_batch_summary.csv"
OUT_JSONL = "/content/mrz_batch_logs.jsonl"
OUT_METRICS = "/content/mrz_gt_metrics.json"

# ---- Стратегія спроб ----
EARLY_STOP_SCORE = 80              # як тільки >=80 і doc/final пройшли — стоп
MAX_GPT_TRIES = 3                  # максимум викликів GPT на файл (base0 + опційний preproc)
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

def calc_cost_usd(prompt_tokens, completion_tokens) -> float:
    p = (prompt_tokens or 0)
    c = (completion_tokens or 0)
    return (p / 1000.0) * COST_PER_1K_PROMPT + (c / 1000.0) * COST_PER_1K_COMPLETION

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
@@ -248,53 +256,50 @@ def pick_two_lines_from_text(raw_text: str):
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
@@ -348,192 +353,296 @@ def _weak(row_core):
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

    tokens_meta = model_res.get("tokens", {})
    row_core = {
        "api_s": model_res["latency_s"],
        "jpeg_kb": round(len(jpg)/1024,1),
        "tokens_prompt": tokens_meta.get("prompt"),
        "tokens_completion": tokens_meta.get("completion"),
        "tokens_total": tokens_meta.get("total"),
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
        "tokens_prompt": 0,
        "tokens_completion": 0,
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
    attempts = []
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    tokens_sum = 0
    total_cost = 0.0
    try:
        base_img = Image.open(io.BytesIO(data_bytes)).convert("RGB")
    except UnidentifiedImageError:
        empty_meta = {
            "file": fname, "api_s": 0, "total_s": 0, "jpeg_kb": 0,
            "tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0, "cost_usd": 0.0,
            "picked": 0, "fallback": "", "repair_strategy": None, "valid_score": None,
            "doc_pass": None, "fin_pass": None, "issues": "not_an_image"
        }
        meta_err = {"file": fname, "error": "UnidentifiedImageError", "attempts": []}
        return empty_meta, meta_err, {
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
    if t_row:
        attempts.append({
            "step": "tesseract",
            "picked": bool(t_row["picked"]),
            "valid_score": t_row.get("valid_score"),
            "doc_pass": t_row.get("doc_pass"),
            "fin_pass": t_row.get("fin_pass"),
            "issues": t_row.get("issues"),
            "latency_s": t_row.get("api_s", 0.0),
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "cost_usd": 0.0,
            "fallback_used": "tesseract"
        })
        if t_row["picked"] and t_row["valid_score"] and _good_enough(t_row):
            T1 = time.time()
            out_row = {
                "file": fname, "api_s": 0.0, "total_s": round(T1-T0,3), "jpeg_kb": None,
                "tokens_prompt": 0, "tokens_completion": 0, "tokens_total": 0, "cost_usd": 0.0,
                "picked": 1, "fallback": "tesseract", "repair_strategy": "as_is",
                "valid_score": t_row["valid_score"], "doc_pass": t_row["doc_pass"], "fin_pass": t_row["fin_pass"],
                "issues": t_row["issues"], "preproc_used": 0, "extra_tries": 0
            }
            meta_final = {"file": fname, "picked_variant": "tesseract", **(t_pack[0] or {}),
                          "attempts": attempts, "cost_usd": 0.0, "tokens_sum": {
                              "prompt": 0, "completion": 0, "total": 0
                          }}
            return out_row, meta_final, t_pack[1]

    # 1) GPT base 0°
    tries = 0
    base0 = set_dbg(base_img.copy(), f"{fname} | base0")
    row_core, meta_core, result_json = run_gpt_on_roi(base0, idx, total, print_raw=True)
    prompt_tokens = row_core.get("tokens_prompt") or 0
    completion_tokens = row_core.get("tokens_completion") or 0
    total_tokens = row_core.get("tokens_total") or (prompt_tokens + completion_tokens)
    cost = calc_cost_usd(prompt_tokens, completion_tokens)
    prompt_tokens_sum += prompt_tokens
    completion_tokens_sum += completion_tokens
    tokens_sum += total_tokens
    total_cost += cost
    attempts.append({
        "step": "gpt_base0",
        "picked": bool(row_core["picked"]),
        "valid_score": row_core.get("valid_score"),
        "doc_pass": row_core.get("doc_pass"),
        "fin_pass": row_core.get("fin_pass"),
        "issues": row_core.get("issues"),
        "latency_s": row_core.get("api_s"),
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
        "cost_usd": round(cost, 6),
        "fallback_used": row_core.get("fallback") or ""
    })
    best = {"row": row_core, "meta": meta_core, "json": result_json, "label": "base0", "pre": 0}
    tries += 1
    if _good_enough(best["row"]) or tries >= MAX_GPT_TRIES:
        T1 = time.time()
        out_row = {
            "file": fname, "api_s": best["row"]["api_s"], "total_s": round(T1-T0,3),
            "jpeg_kb": best["row"]["jpeg_kb"], "tokens_prompt": prompt_tokens_sum,
            "tokens_completion": completion_tokens_sum, "tokens_total": tokens_sum,
            "cost_usd": round(total_cost, 6),
            "picked": best["row"]["picked"], "fallback": best["row"]["fallback"],
            "repair_strategy": "as_is", "valid_score": best["row"]["valid_score"],
            "doc_pass": best["row"]["doc_pass"], "fin_pass": best["row"]["fin_pass"],
            "issues": best["row"]["issues"], "preproc_used": best["pre"],
            "extra_tries": tries-1
        }
        meta_final = {"file": fname, "picked_variant": best["label"], **best["meta"],
                      "attempts": attempts, "cost_usd": round(total_cost,6),
                      "tokens_sum": {"prompt": prompt_tokens_sum, "completion": completion_tokens_sum, "total": tokens_sum}}
        return out_row, meta_final, best["json"]

    # 2) Якщо все ще слабко → легкий препроцесинг 0°
    def better(a,b):
        av = a["row"]["valid_score"] or 0
        bv = b["row"]["valid_score"] or 0
        return a if av >= bv else b

    if USE_PREPROC_IF_WEAK and _weak(best["row"]) and tries < MAX_GPT_TRIES:
        pre0 = set_dbg(preprocess_for_mrz(base_img), f"{fname} | pre0")
        p_row, p_meta, p_json = run_gpt_on_roi(pre0, idx, total, print_raw=False)
        prompt_tokens = p_row.get("tokens_prompt") or 0
        completion_tokens = p_row.get("tokens_completion") or 0
        total_tokens = p_row.get("tokens_total") or (prompt_tokens + completion_tokens)
        cost = calc_cost_usd(prompt_tokens, completion_tokens)
        prompt_tokens_sum += prompt_tokens
        completion_tokens_sum += completion_tokens
        tokens_sum += total_tokens
        total_cost += cost
        attempts.append({
            "step": "gpt_pre0",
            "picked": bool(p_row["picked"]),
            "valid_score": p_row.get("valid_score"),
            "doc_pass": p_row.get("doc_pass"),
            "fin_pass": p_row.get("fin_pass"),
            "issues": p_row.get("issues"),
            "latency_s": p_row.get("api_s"),
            "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
            "cost_usd": round(cost, 6),
            "fallback_used": p_row.get("fallback") or ""
        })
        pcand = {"row": p_row, "meta": p_meta, "json": p_json, "label": "pre0", "pre": 1}
        best = better(best, pcand)
        tries += 1

    T1 = time.time()
    out_row = {
        "file": fname, "api_s": best["row"]["api_s"], "total_s": round(T1-T0,3),
        "jpeg_kb": best["row"]["jpeg_kb"], "tokens_prompt": prompt_tokens_sum,
        "tokens_completion": completion_tokens_sum, "tokens_total": tokens_sum,
        "cost_usd": round(total_cost, 6),
        "picked": best["row"]["picked"], "fallback": best["row"]["fallback"],
        "repair_strategy": "as_is", "valid_score": best["row"]["valid_score"],
        "doc_pass": best["row"]["doc_pass"], "fin_pass": best["row"]["fin_pass"],
        "issues": best["row"]["issues"], "preproc_used": best["pre"],
        "extra_tries": tries-1
    }
    meta_final = {"file": fname, "picked_variant": best["label"], **best["meta"],
                  "attempts": attempts, "cost_usd": round(total_cost,6),
                  "tokens_sum": {"prompt": prompt_tokens_sum, "completion": completion_tokens_sum, "total": tokens_sum}}
    return out_row, meta_final, best["json"]

def _fmt_pass(v):
    if v is True: return "✅"
    if v is False: return "❌"
    return "—"

def _fmt_bool(v):
    if v is True: return "✅"
    if v is False: return "❌"
    return "—"

def print_detailed_log(row, meta):
    total_s = row.get("total_s")
    cost = row.get("cost_usd") or 0.0
    score = row.get("valid_score")
    print(f"\n🧾 {row.get('file')} → score={score if score is not None else '—'} | doc={_fmt_pass(row.get('doc_pass'))} | fin={_fmt_pass(row.get('fin_pass'))} | time={total_s if total_s is not None else '—'}s | cost=${cost:.4f}")
    print(f"    tokens: prompt={row.get('tokens_prompt') or 0}, completion={row.get('tokens_completion') or 0}, total={row.get('tokens_total') or 0}")
    if row.get("fallback"):
        print(f"    fallback: {row.get('fallback')}")
    if row.get("issues"):
        print(f"    issues: {row.get('issues')}")

    attempts = meta.get("attempts") if isinstance(meta, dict) else None
    if not attempts:
        return

    label_map = {
        "gpt_base0": "GPT base",
        "gpt_pre0": "GPT preproc",
        "tesseract": "Tesseract"
    }
    for att in attempts:
        step = att.get("step")
        label = label_map.get(step, step or "?")
        tokens = att.get("tokens") or {}
        att_cost = att.get("cost_usd") or 0.0
        print(
            f"    • {label}: picked={_fmt_bool(att.get('picked'))} | score={att.get('valid_score') if att.get('valid_score') is not None else '—'} | "
            f"doc={_fmt_pass(att.get('doc_pass'))} | fin={_fmt_pass(att.get('fin_pass'))} | latency={att.get('latency_s') if att.get('latency_s') is not None else '—'}s | "
            f"tokens={tokens.get('total', 0)} | cost=${att_cost:.6f}"
        )
        if att.get("fallback_used"):
            print(f"        fallback_used: {att.get('fallback_used')}")
        if att.get("issues"):
            print(f"        issues: {att.get('issues')}")

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
@@ -588,68 +697,69 @@ if gt_key:
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
    "file","api_s","total_s","jpeg_kb","tokens_prompt","tokens_completion","tokens_total","cost_usd",
    "picked","fallback","repair_strategy","valid_score","doc_pass","fin_pass",
    "issues","preproc_used","extra_tries"
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
    print_detailed_log(row, meta)
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
