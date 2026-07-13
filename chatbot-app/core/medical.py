from core.settings import *
from core.textutil import *
from core.llm import *


def _screen_upload(image_bytes, image_type):
    # Run the OOD/quality gate on an upload. Returns a GateResult or None (if the
    if not DICOM_SUPPORT:
        return None
    modality = 'dental_xray' if image_type == 'dental' else image_type
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            return check_input(im, modality)
    except Exception:  # noqa: BLE001
        return None


def analyze_image_with_api(image_bytes, image_type):
    # Send image to appropriate medical API for analysis.
    try:
        if image_type == 'bone':
            api_url = f"{BONE_DETECT_API}/predict_for_llm"
        elif image_type == 'dental_photo':
            # Intraoral photos -> ConvNeXt classifier with GradCAM-derived bbox
            api_url = f"{ORAL_CLASSIFY_API}/predict_for_llm"
        elif image_type in ['dental', 'dental_xray']:
            # Panoramic dental X-rays -> YOLO detection
            api_url = f"{ORAL_DETECT_API}/predict_for_llm"
        elif image_type == 'chest':
            api_url = f"{CHEST_XRAY_API}/predict_for_llm?include_gradcam=true"
        elif image_type == 'prescription':
            # Handwritten prescription -> Qwen2-VL VLM + RapidFuzz against Egyptian drugs DB
            api_url = f"{PRESCRIPTION_API}/predict_for_llm"
        else:
            return {"error": f"Unknown image type: {image_type}"}

        
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        # Prescriptions can require lazy-loading a VLM (~minutes on first call).
        upload_timeout = 1800 if image_type == 'prescription' else 180
        response = requests.post(api_url, files=files, timeout=upload_timeout)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to {image_type} API. Make sure it's running."}
    except Exception as e:
        return {"error": str(e)}


def parse_detections_for_overlay(analysis_result):
    # Extract bbox detections from API results for visualization.
    detections = []

    raw_detections = analysis_result.get("detections")
    if isinstance(raw_detections, list):
        for det in raw_detections:
            bbox = det.get("bbox") if isinstance(det, dict) else None
            if isinstance(bbox, list) and len(bbox) >= 4:
                detections.append({
                    "bbox": bbox[:4],
                    "label": det.get("class_name") or det.get("finding") or "Finding",
                    "confidence": det.get("confidence"),
                })

    if detections:
        return detections

    raw_findings = analysis_result.get("ai_findings")
    if isinstance(raw_findings, list):
        for finding in raw_findings:
            if not isinstance(finding, dict):
                continue
            location = str(finding.get("location", ""))
            if "bbox" not in location:
                continue
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", location)]
            if len(nums) >= 4:
                detections.append({
                    "bbox": nums[:4],
                    "label": finding.get("finding") or "Finding",
                    "confidence": finding.get("confidence"),
                })

    return detections


def prepare_detections_for_overlay(detections, image_type):
    # Clean and cap detections to keep output overlays readable.
    if not isinstance(detections, list):
        return []

    def confidence_to_score(value):
        try:
            if value is None:
                return -1.0
            if isinstance(value, str):
                numeric = float(value.strip().replace('%', ''))
                return numeric / 100.0 if numeric > 1.0 else numeric
            numeric = float(value)
            return numeric / 100.0 if numeric > 1.0 else numeric
        except (TypeError, ValueError):
            return -1.0

    cleaned = []
    for det in detections:
        if not isinstance(det, dict):
            continue

        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        cleaned.append({
            "bbox": [x1, y1, x2, y2],
            "label": det.get("label", "Finding"),
            "confidence": det.get("confidence"),
            "score": confidence_to_score(det.get("confidence")),
        })

    cleaned.sort(key=lambda item: item["score"], reverse=True)
    limit = OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE.get(image_type, OVERLAY_DEFAULT_MAX_DETECTIONS)
    if limit > 0:
        cleaned = cleaned[:limit]

    return cleaned


def draw_detections_on_image(image_bytes, detections, image_type='dental'):
    # Draw simplified detection markers on the image and return base64 JPEG.
    if not IMAGE_SUPPORT or not detections:
        return None

    try:
        detections = prepare_detections_for_overlay(detections, image_type)
        if not detections:
            return None

        # Open image safely. PIL converting 16-bit grayscale X-rays directly to
        # RGB can produce a fully-white image due to precision loss, so we
        # normalize high-bitdepth modes first.
        raw = Image.open(io.BytesIO(image_bytes))
        if raw.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            arr = np.array(raw).astype(np.float32)
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
            image = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        else:
            image = raw.convert("RGB")
        draw = ImageDraw.Draw(image)

        img_w, img_h = image.size
        min_side = max(1, min(img_w, img_h))
        line_w = max(2, int(min_side * 0.003))

        try:
            from PIL import ImageFont
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                max(14, int(min_side * 0.022)),
            )
        except Exception:
            font = None

        for det in detections:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(float(x1), img_w - 1))
            y1 = max(0.0, min(float(y1), img_h - 1))
            x2 = max(0.0, min(float(x2), img_w - 1))
            y2 = max(0.0, min(float(y2), img_h - 1))

            color = (255, 80, 80)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

            label = str(det.get("label", "Finding"))
            conf = det.get("confidence")
            try:
                cv = float(conf)
                if cv <= 1.0:
                    cv *= 100.0
                label = f"{label} {cv:.0f}%"
            except (TypeError, ValueError):
                pass

            if font is not None:
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    tw, th = tb[2] - tb[0], tb[3] - tb[1]
                except Exception:
                    tw, th = (len(label) * 8, 14)
                pad = 3
                ty = max(0, y1 - th - 2 * pad)
                draw.rectangle([x1, ty, x1 + tw + 2 * pad, ty + th + 2 * pad], fill=color)
                draw.text((x1 + pad, ty + pad), label, fill=(255, 255, 255), font=font)

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=92)
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.read()).decode()
    except Exception as e:
        print(f"⚠️ Error drawing detections: {e}")
        return None


def fetch_detection_overlay_detections(image_bytes, image_type):
    # Fetch bbox detections from detection APIs for overlay if missing.
    if image_type == 'bone':
        api_url = f"{BONE_DETECT_API}/predict"
    elif image_type == 'dental_photo':
        api_url = f"{ORAL_CLASSIFY_API}/predict_for_llm"
    elif image_type in ['dental', 'dental_xray']:
        api_url = f"{ORAL_DETECT_API}/predict"
    else:
        return []

    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(api_url, files=files, timeout=60)
        if response.status_code != 200:
            return []
        data = response.json()
        detections = data.get("detections")
        if isinstance(detections, list):
            return [
                {
                    "bbox": det.get("bbox"),
                    "label": det.get("class_name") or det.get("finding") or "Finding",
                    "confidence": det.get("confidence"),
                }
                for det in detections
                if isinstance(det, dict)
            ]
    except Exception:
        return []

    return []


def extract_pdf_text(pdf_bytes):
    # Extract text from PDF file.
    if not PDF_SUPPORT:
        return {"error": "PDF support not available. Install pdfplumber."}
    
    try:
        text_content = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- صفحة {i+1} ---\n{page_text}")
        
        if text_content:
            return {"success": True, "text": "\n\n".join(text_content), "pages": len(text_content)}
        else:
            return {"error": "لم يتم العثور على نص في الملف"}
    except Exception as e:
        return {"error": f"خطأ في قراءة PDF: {str(e)}"}


def normalize_oral_label_ar(label):
    # Normalize technical oral labels to patient-friendly Arabic terms.
    text = str(label or '').strip().lower().replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)

    if 'ulcer' in text:
        return 'قرحة فموية'
    if 'gingiv' in text:
        return 'التهاب اللثة'
    if 'calculus' in text or 'tartar' in text:
        return 'جير على الأسنان'
    if 'hypodontia' in text or 'missing tooth' in text or 'missing' in text:
        return 'سن مفقود'
    if 'wisdom' in text:
        return 'ضرس العقل'
    if 'apical periodontitis' in text or 'periodontitis' in text:
        return 'التهاب لب جذر السن'
    if 'root canal' in text:
        return 'حشوة عصب (علاج جذر)'
    if 'porcelain crown' in text or ('crown' in text and 'porcelain' in text):
        return 'تاج خزفي'
    if 'crown' in text:
        return 'تاج سن'
    if 'ceramic bridge' in text or 'bridge' in text:
        return 'جسر سني'
    if 'implant' in text:
        return 'زرعة سن'
    if 'dental filling' in text or 'filling' in text:
        return 'حشوة أسنان'
    if 'discolor' in text or 'stain' in text:
        return 'تغير لون الأسنان'
    if 'caries' in text or 'decay' in text or 'cavity' in text:
        return 'تسوس أسنان'
    if re.search(r'class\s*\d+', text):
        return 'تسوس أسنان'

    return 'مؤشر فموي يحتاج تقييم'


def normalize_chest_label_ar(label):
    # Translate chest class labels to Arabic-friendly names.
    mapping = {
        'Atelectasis': 'انخماص الرئة',
        'Cardiomegaly': 'تضخم القلب',
        'Consolidation': 'تصلب رئوي',
        'Edema': 'وذمة رئوية',
        'Effusion': 'انصباب جنبي',
        'Emphysema': 'انتفاخ الرئة',
        'Fibrosis': 'تليف رئوي',
        'Infiltration': 'ارتشاح رئوي',
        'Mass': 'كتلة رئوية',
        'Nodule': 'عقيدة رئوية',
        'Pleural_Thickening': 'سماكة غشاء الجنب',
        'Pneumonia': 'التهاب رئوي',
        'Pneumothorax': 'استرواح صدري',
    }
    return mapping.get(str(label or '').strip(), str(label or 'مؤشر صدري'))


def _bbox_iou(a, b):
    try:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def dedup_detections_by_iou(detections, iou_threshold=0.4):
    # Greedy NMS-style dedup: keep highest-conf box per class cluster.
    sorted_dets = sorted(detections, key=lambda d: d.get('conf', 0), reverse=True)
    kept = []
    for d in sorted_dets:
        duplicate = False
        for k in kept:
            if k['class'] != d['class']:
                continue
            if _bbox_iou(k.get('bbox') or [0, 0, 0, 0], d.get('bbox') or [0, 0, 0, 0]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


# Curated Arabic treatment recommendations, keyed by normalised finding label.
# Restored from the pre-refactor app.py; build_per_finding_recommendations() below
# is its only consumer.
PER_FINDING_RECOMMENDATIONS_AR = {
    # ---- Oral X-ray (9-class) ----
    "wisdom tooth": "متابعة دورية للأسنان العقل؛ في حال كانت مدفونة أو تسبب التهاب أو ازدحام يُنصح بالخلع الجراحي بعد تقييم طبيب الأسنان.",
    "missing tooth": "النظر في خيار زرعة أو جسر لتعويض الفراغ ومنع تحرك الأسنان المجاورة وفقدان عظم الفك.",
    "decay": "حشو السن المتسوس بعد إزالة النخر؛ في الحالات المتقدمة قد يلزم علاج عصب أو تركيب تاج.",
    "caries": "حشو السن المتسوس بعد إزالة النخر؛ في الحالات المتقدمة قد يلزم علاج عصب أو تركيب تاج.",
    "porcelain crown": "تاج موجود — متابعة دورية للتأكد من سلامة التاج واللثة المحيطة وعدم وجود تسوس تحت التاج.",
    "crown": "تاج موجود — متابعة دورية للتأكد من سلامة التاج واللثة المحيطة وعدم وجود تسوس تحت التاج.",
    "ceramic bridge": "جسر سيراميك موجود — متابعة دورية ونظافة جيدة للأسنان الداعمة للجسر.",
    "bridge": "جسر سني موجود — متابعة دورية ونظافة جيدة للأسنان الداعمة.",
    "implant": "زرعة سنية موجودة — متابعة دورية للتأكد من ثبات الزرعة وصحة العظم واللثة المحيطة.",
    "dental filling": "حشوة موجودة — متابعة دورية للتأكد من سلامة الحشوة وعدم وجود تسوس متكرر حولها.",
    "filling": "حشوة موجودة — متابعة دورية للتأكد من سلامة الحشوة وعدم وجود تسوس متكرر حولها.",
    "root canal filling": "علاج عصب موجود — متابعة دورية بالأشعة للتأكد من نجاح العلاج وعدم وجود التهاب جذري.",
    "root canal": "علاج عصب موجود — متابعة دورية بالأشعة للتأكد من نجاح العلاج وعدم وجود التهاب جذري.",
    "apical periodontitis": "التهاب حول قمي يحتاج علاج عصب الجذر؛ في الحالات المتقدمة قد يلزم استئصال القمة أو خلع السن.",
    # ---- Oral photo (6-class intraoral) ----
    "calculus": "تنظيف وتلميع الأسنان لدى طبيب الأسنان (سكلينج) مع تعليمات نظافة فموية يومية واستخدام خيط الأسنان.",
    "gingivitis": "تنظيف لثوي عميق ومضمضة بمحلول مطهر (مثل الكلورهيكسيدين) لمدة محددة؛ تحسين النظافة الفموية لمنع التطور لالتهاب لثة مزمن.",
    "ulcer": "علاج موضعي مهدئ (جل أو غسول)؛ تجنب الأطعمة الحارة؛ مراجعة الطبيب إذا استمرت القرحة أكثر من أسبوعين.",
    "mouth ulcer": "علاج موضعي مهدئ (جل أو غسول)؛ تجنب الأطعمة الحارة؛ مراجعة الطبيب إذا استمرت القرحة أكثر من أسبوعين.",
    "tooth discoloration": "تحديد سبب التغير (داخلي أو خارجي)؛ تنظيف احترافي أو تبييض أسنان حسب الحالة.",
    "discoloration": "تحديد سبب التغير (داخلي أو خارجي)؛ تنظيف احترافي أو تبييض أسنان حسب الحالة.",
    "hypodontia": "استشارة أخصائي تقويم وتعويضات سنية لتقييم خيارات تعويض الأسنان المفقودة خلقياً.",
    # ---- Bone X-ray ----
    "fracture": "تثبيت الكسر بالجبيرة أو الجبس أو التدخل الجراحي حسب نوع الكسر؛ متابعة بالأشعة للتأكد من الالتئام السليم.",
    "boneanomaly": "تقييم طبي تفصيلي للشذوذ العظمي مع أشعة إضافية إذا لزم الأمر.",
    "bonelesion": "تقييم آفة العظم بأشعة مقطعية أو رنين مغناطيسي وقد يلزم خزعة لتحديد الطبيعة.",
    "foreignbody": "تقييم لإزالة الجسم الغريب جراحياً إذا كان يسبب أعراضاً أو خطر عدوى.",
    "metal": "تثبيت معدني موجود — متابعة الالتئام وتقييم وجود التهاب أو ارتخاء.",
    "periostealreaction": "متابعة الالتئام وتقييم وجود عدوى أو ورم؛ قد يلزم تصوير إضافي.",
    "pronatorsign": "علامة على كسر دقيق محتمل في الساعد — تقييم سريري وأشعة إضافية.",
    "softtissue": "تقييم تورم الأنسجة الرخوة لاحتمال وجود التهاب أو نزيف داخلي.",
    "text": "",
    # ---- Chest X-ray ----
    "pneumonia": "علاج بالمضادات الحيوية حسب نوع العدوى؛ راحة وترطيب جيد؛ متابعة بأشعة بعد 4–6 أسابيع للتأكد من الشفاء.",
    "tuberculosis": "علاج طويل الأمد بأدوية السل المركبة (6 أشهر على الأقل)؛ متابعة دورية وعزل حسب الإرشادات.",
    "covid-19": "عزل وراحة ومتابعة الأكسجين؛ علاج عرضي وأدوية مضادة للفيروسات في الحالات المتوسطة والشديدة.",
    "covid": "عزل وراحة ومتابعة الأكسجين؛ علاج عرضي وأدوية مضادة للفيروسات في الحالات المتوسطة والشديدة.",
    "lung opacity": "تقييم إضافي لتحديد سبب العتامة (التهاب، وذمة، أو ورم)؛ قد يلزم أشعة مقطعية.",
    "viral pneumonia": "علاج عرضي وراحة وترطيب؛ مضادات فيروسية حسب نوع الفيروس؛ متابعة طبية.",
    "bacterial pneumonia": "علاج بمضادات حيوية مناسبة؛ متابعة بأشعة بعد العلاج للتأكد من الشفاء.",
    "edema": "علاج السبب الأساسي (قصور قلب، فشل كلوي)؛ مدرات بول ومتابعة وظائف القلب.",
    "effusion": "بزل السائل من الجنب لتحليله إذا كان كبيراً؛ علاج السبب الأساسي.",
    "pleural effusion": "بزل السائل من الجنب لتحليله إذا كان كبيراً؛ علاج السبب الأساسي.",
    "consolidation": "غالباً علامة على التهاب رئوي — علاج بالمضادات الحيوية ومتابعة دورية بالأشعة.",
    "infiltration": "تقييم لتحديد السبب (التهاب، عدوى، حساسية)؛ متابعة بأشعة لاحقة.",
    "cardiomegaly": "تقييم وظائف القلب بإيكو وتخطيط قلب؛ علاج السبب (ارتفاع ضغط، اعتلال عضلة قلب).",
    "atelectasis": "تنفس عميق وتمارين تنفسية؛ علاج السبب الأساسي (انسداد قصبي، ضغط من خارج الرئة).",
    "pneumothorax": "إذا كان كبيراً يتطلب تدخل عاجل لتركيب أنبوب صدري لتفريغ الهواء؛ متابعة دقيقة.",
    "mass": "أشعة مقطعية بالصبغة وقد يلزم خزعة لتحديد طبيعة الكتلة (حميدة/خبيثة).",
    "nodule": "متابعة بأشعة مقطعية كل 3–6 أشهر لتقييم النمو؛ خزعة إذا كانت العقدة مشبوهة.",
    "fibrosis": "تقييم بأشعة مقطعية ووظائف رئة؛ علاج داعم لإبطاء التطور.",
    "emphysema": "إيقاف التدخين فوراً، علاج بموسعات قصبية واستنشاقات، تأهيل رئوي.",
    "pleural thickening": "متابعة دورية بالأشعة؛ تقييم تاريخ التعرض للأسبستوس أو الالتهابات السابقة.",
    "hernia": "تقييم جراحي لاحتمال الإصلاح حسب الحجم والأعراض.",
    "no finding": "لا توجد نتائج مرضية واضحة — استمر في المتابعة الدورية والوقاية الصحية.",
    "normal": "النتائج طبيعية — استمر في المتابعة الدورية والوقاية الصحية.",
}


def _normalize_label(label):
    if not label:
        return ""
    return str(label).strip().lower()


def build_per_finding_recommendations(analysis_result, image_type=None):
    # Return an Arabic block of per-finding clinical recommendations.
    if not isinstance(analysis_result, dict):
        return ""

    seen = set()
    items = []  # list of (display_label, recommendation)

    def _add(label):
        key = _normalize_label(label)
        if not key or key in seen:
            return
        rec = PER_FINDING_RECOMMENDATIONS_AR.get(key)
        if not rec:
            return
        seen.add(key)
        items.append((label, rec))

    for det in analysis_result.get("detections", []) or []:
        if isinstance(det, dict):
            _add(det.get("class_name") or det.get("label") or det.get("class"))

    ai = analysis_result.get("ai_findings") or {}
    if isinstance(ai, dict):
        _add(ai.get("primary_diagnosis"))
        for d in ai.get("differential_diagnoses") or []:
            if isinstance(d, dict):
                _add(d.get("condition") or d.get("diagnosis") or d.get("label"))
            else:
                _add(d)

    if not items:
        return ""

    lines = ["", "📋 توصيات علاجية مقترحة لكل حالة (مرجع للطبيب):"]
    for label, rec in items:
        lines.append(f"- {label}: {rec}")
    return "\n".join(lines)


def build_prescription_report(analysis_result):
    # Build a structured Arabic report for the prescription parser output.
    if not isinstance(analysis_result, dict):
        return "تعذّر قراءة الروشتة."

    ai = analysis_result.get('ai_findings') or {}
    meds = ai.get('medications') or []
    if not meds and isinstance(analysis_result.get('report_data'), dict):
        meds = analysis_result['report_data'].get('medications') or []

    lines = ["📋 تحليل الروشتة (Handwritten Prescription):"]
    if not meds:
        lines.append("لم يتم استخراج أي أدوية واضحة من الصورة.")
        lines.append("")
        lines.append("⏰ التوصية: حاول رفع صورة أوضح للروشتة أو تواصل مع الصيدلي.")
        return "\n".join(lines)

    def _clean_text(v):
        return re.sub(r"\s+", " ", str(v or "").strip())

    def _alpha_len(v):
        return len(re.findall(r"[A-Za-z\u0621-\u064A]", _clean_text(v)))

    def _is_weak_name(v):
        t = _clean_text(v).lower()
        if not t or t in {"-", "—", "n/a", "na", "none", "null"}:
            return True
        return _alpha_len(t) <= 2

    def _simple_schedule_ar(freq_raw, dosage_raw):
        freq = _clean_text(freq_raw).lower()
        dosage = _clean_text(dosage_raw).lower()

        if not freq:
            return "غير واضح"
        if re.search(r"عند\s*اللزو?م|عند\s*الحاجة|\bprn\b", freq):
            return "عند اللزوم"
        m = re.search(r"(?:كل\s*)(\d{1,2})\s*(?:ساع|ساعة|ساعات)|(?:\bq\s*)(\d{1,2})\s*h", freq)
        if m:
            h = m.group(1) or m.group(2)
            return f"كل {h} ساعات"

        count = ""
        if re.search(r"\b(bid|مرتين|twice)\b", freq):
            count = "مرتين يوميا"
        elif re.search(r"\b(tid|ثلاث|3\s*مرات|three times)\b", freq):
            count = "3 مرات يوميا"
        elif re.search(r"\b(qid|اربع|4\s*مرات|four times)\b", freq):
            count = "4 مرات يوميا"
        elif re.search(r"\b(od|qd|once|daily|يومي|مرة)\b", freq) or freq in {"1", "1x", "1 tab", "1 cap", "1 dose", "1 amp", "1 drop"}:
            count = "مرة يوميا"
        elif freq in {"2", "2x", "2 tab", "2 cap", "2 dose"}:
            count = "مرتين يوميا"
        elif freq in {"3", "3x", "3 tab", "3 cap", "3 dose"}:
            count = "3 مرات يوميا"

        parts = []
        if count:
            parts.append(count)

        if re.search(r"قبل\s*ال[اأ]?كل|before\s*meal|before\s*food", freq):
            parts.append("قبل الأكل")
        elif re.search(r"بعد\s*ال[اأ]?كل|after\s*meal|after\s*food", freq):
            parts.append("بعد الأكل")

        if re.search(r"قبل\s*النوم|before\s*sleep|bedtime|\bhs\b", freq):
            parts.append("قبل النوم")

        times = []
        if re.search(r"صباح|صبح|morning|\bam\b|الفطار", freq):
            times.append("صباحا")
        if re.search(r"ظهر|noon|الغدا", freq):
            times.append("ظهرا")
        if re.search(r"مساء|ليل|عشاء|bedtime|\bpm\b|\bhs\b|evening|night", freq):
            times.append("مساء")
        if times:
            parts.append(" و ".join(times))

        schedule = " - ".join(parts).strip(" -")
        if schedule == "مرة يوميا" and any(k in dosage for k in ("amp", "ampoule", "vial", "inj", "حقن")):
            return "جرعة واحدة"
        return schedule or "غير واضح"

    def _display_name_and_flag(med):
        official = _clean_text(med.get('drug'))
        extracted = _clean_text(med.get('drug_extracted'))
        score = float(med.get('confidence_score') or 0)
        official_match = bool(med.get('official_match'))

        official_weak = _is_weak_name(official)
        extracted_weak = _is_weak_name(extracted)

        # Avoid showing short/noisy tokens as final names.
        if official_weak and not extracted_weak:
            return extracted, "⚠️ يحتاج مراجعة صيدلي"
        if official_match and score >= 92 and not official_weak:
            return official, "✅ مطابق قوي"
        if official_match and score >= 82 and not official_weak:
            return official, "🟡 مطابق تقريبي"
        if not extracted_weak:
            return extracted, "⚠️ يحتاج مراجعة صيدلي"
        if not official_weak:
            return official, "⚠️ يحتاج مراجعة صيدلي"
        return "اسم غير واضح", "⚠️ يحتاج مراجعة صيدلي"

    verified_strong = 0

    lines.append(f"تم رصد {len(meds)} دواء من الروشتة.")
    lines.append("")
    lines.append("💊 الأدوية المكتشفة (نسخة مبسطة):")
    for i, med in enumerate(meds, 1):
        name, name_flag = _display_name_and_flag(med)

        score = float(med.get('confidence_score') or 0)
        if name_flag.startswith("✅"):
            verified_strong += 1

        lines.append(f"{i}) {name} ({name_flag})")
        if 0 < score < 92:
            lines.append(f"   - درجة المطابقة: {score:.0f}%")

    lines.append("")
    lines.append(f"✅ أسماء مطابقة بقوة: {verified_strong} من {len(meds)}")
    lines.append("⏰ المتابعة: لأي تفاصيل جرعة/توقيت، الرجاء الرجوع مباشرة إلى نص الروشتة أو الصيدلي.")
    return "\n".join(lines)


def generate_medical_report(analysis_result, file_type, image_type=None, target_language='ar'):
    # Generate medical report using LLM based on analysis results.
    response_language = normalize_language_code(target_language)

    if not LLM_AVAILABLE:
        if response_language == 'en':
            return f"Analysis results:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
        return f"نتائج التحليل:\n{json.dumps(analysis_result, ensure_ascii=False, indent=2)}"
    
    if file_type == 'image':
        if 'error' in analysis_result:
            if response_language == 'en':
                return f"Analysis error: {analysis_result['error']}"
            return f"خطأ في التحليل: {analysis_result['error']}"

        # ---- Prescription parsing has its own structured report (no LLM needed) ----
        if image_type == 'prescription':
            prescription_report = build_prescription_report(analysis_result)
            if response_language == 'en':
                translation_prompt = f"""Translate the following medical report from Arabic to clear English.
Rules:
- Preserve the original structure, numbering, and emojis.
- Keep medication names as they are when possible.
- Keep safety warnings explicit.

Report:
{prescription_report}
"""
                translated = resolve_llm_response(
                    get_llm_response(
                        translation_prompt,
                        "You are a medical translator. Return only the translated report in English.",
                    )
                )
                return clean_markdown(translated)
            return prescription_report

        findings_list = []

        if image_type in ['dental', 'dental_photo', 'dental_xray']:
            # Group detections by class with NMS dedup so each tooth/lesion is
            # counted once. Then list count + confidence range per class.
            raw = []
            for d in analysis_result.get('detections', []):
                if not isinstance(d, dict):
                    continue
                class_name = d.get('class_name')
                if not class_name:
                    continue
                conf_pct = confidence_to_percent(d.get('confidence'))
                if conf_pct is None or conf_pct < 30:
                    continue
                bbox = d.get('bbox') or [0, 0, 0, 0]
                raw.append({'class': class_name, 'conf': conf_pct, 'bbox': bbox})
            kept = dedup_detections_by_iou(raw, iou_threshold=0.4)
            grouped = {}
            for d in kept:
                label_ar = normalize_oral_label_ar(d['class'])
                grouped.setdefault(label_ar, []).append(d['conf'])
            for label_ar, confs in sorted(grouped.items(), key=lambda x: max(x[1]), reverse=True):
                confs_sorted = sorted(confs, reverse=True)
                if len(confs_sorted) == 1:
                    findings_list.append(f"- يرجح وجود {label_ar} (احتمال {confs_sorted[0]:.0f}%)")
                else:
                    confs_str = ", ".join(f"{c:.0f}%" for c in confs_sorted)
                    findings_list.append(
                        f"- يرجح وجود {label_ar} في {len(confs_sorted)} مواضع (احتمالات: {confs_str})"
                    )

            # Classifier-style fallback: intraoral photo classifier returns
            # no detections, only ai_findings + all_probabilities. List the
            # primary diagnosis and any differentials >=20%.
            if not findings_list:
                ai_findings = analysis_result.get('ai_findings')
                if isinstance(ai_findings, dict):
                    primary = ai_findings.get('primary_diagnosis')
                    primary_conf = confidence_to_percent(ai_findings.get('confidence'))
                    if primary:
                        label_ar = normalize_oral_label_ar(primary)
                        if primary_conf is not None:
                            findings_list.append(
                                f"- يرجح وجود {label_ar} (احتمال {primary_conf:.0f}%)"
                            )
                        else:
                            findings_list.append(f"- يرجح وجود {label_ar}")

                all_probs = analysis_result.get('all_probabilities') or {}
                seen_labels = set()
                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    pct = confidence_to_percent(prob)
                    if pct is None or pct < 20:
                        continue
                    label_ar = normalize_oral_label_ar(cls)
                    if label_ar in seen_labels:
                        continue
                    # Skip the primary one we already added.
                    primary_ar = normalize_oral_label_ar(
                        (analysis_result.get('ai_findings') or {}).get('primary_diagnosis') or ''
                    )
                    if label_ar == primary_ar:
                        seen_labels.add(label_ar)
                        continue
                    seen_labels.add(label_ar)
                    findings_list.append(f"- احتمال إضافي: {label_ar} ({pct:.0f}%)")

        elif image_type == 'chest':
            ai_findings = analysis_result.get('ai_findings')
            if isinstance(ai_findings, dict):
                primary = ai_findings.get('primary_diagnosis')
                primary_conf = confidence_to_percent(ai_findings.get('confidence'))
                if primary:
                    primary_label = normalize_chest_label_ar(primary)
                    if primary_conf is not None:
                        findings_list.append(f"- التشخيص الأكثر ترجيحًا: {primary_label} (احتمال {primary_conf:.0f}%)")
                    else:
                        findings_list.append(f"- التشخيص الأكثر ترجيحًا: {primary_label}")

            for cond in analysis_result.get('detected_conditions', []):
                if not isinstance(cond, dict):
                    continue
                raw_label = cond.get('condition') or cond.get('class')
                conf_pct = confidence_to_percent(cond.get('probability'))
                if not raw_label or conf_pct is None or conf_pct < 30:
                    continue
                findings_list.append(
                    f"- احتمال إضافي: {normalize_chest_label_ar(raw_label)} ({conf_pct:.0f}%)"
                )

        else:
            # Bone / generic: dedup overlapping detections (YOLO can return
            # near-duplicate boxes for the same fracture) then group per class.
            raw = []
            for d in analysis_result.get('detections', []):
                if not isinstance(d, dict) or not d.get('class_name'):
                    continue
                conf_pct = confidence_to_percent(d.get('confidence'))
                if conf_pct is None or conf_pct < 30:
                    continue
                bbox = d.get('bbox') or [0, 0, 0, 0]
                raw.append({'class': d['class_name'], 'conf': conf_pct, 'bbox': bbox})
            kept = dedup_detections_by_iou(raw, iou_threshold=0.4)
            grouped = {}
            for d in kept:
                grouped.setdefault(d['class'], []).append(d['conf'])
            for cls, confs in sorted(grouped.items(), key=lambda x: max(x[1]), reverse=True):
                confs_sorted = sorted(confs, reverse=True)
                if len(confs_sorted) == 1:
                    findings_list.append(f"- {cls} (احتمال {confs_sorted[0]:.0f}%)")
                else:
                    confs_str = ", ".join(f"{c:.0f}%" for c in confs_sorted)
                    findings_list.append(
                        f"- {cls}: {len(confs_sorted)} مواضع (احتمالات: {confs_str})"
                    )

            ai_findings = analysis_result.get('ai_findings', [])
            if isinstance(ai_findings, list) and not findings_list:
                # Only add ai_findings entries when we don't already have
                # per-detection lines; otherwise we'd duplicate the same
                # finding under two different formats.
                for f in ai_findings:
                    if isinstance(f, dict) and f.get('finding'):
                        findings_list.append(f"- {f['finding']} (ثقة: {f.get('confidence', '؟')})")

        if findings_list:
            findings_intro = "Detected findings:" if response_language == 'en' else "النتائج المكتشفة:"
            findings_text = f"{findings_intro}\n" + "\n".join(findings_list)
        else:
            all_probs = analysis_result.get('all_probabilities', {})
            if all_probs:
                top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                top3_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in top3)
                if response_language == 'en':
                    findings_text = (
                        "No strong abnormal findings above the configured confidence threshold. "
                        f"Top current probabilities: {top3_str}"
                    )
                else:
                    findings_text = f"لا توجد مؤشرات قوية فوق عتبة الثقة المعتمدة. أعلى الاحتمالات الحالية: {top3_str}"
            else:
                if response_language == 'en':
                    findings_text = "No strong abnormal indicators were detected in the image based on the configured threshold."
                else:
                    findings_text = "لا توجد مؤشرات قوية غير طبيعية في الصورة وفق العتبة المعتمدة."

        # Build per-finding recommendations so the LLM has explicit suggested
        # treatment for each detected condition (instead of generic advice).
        findings_text += build_per_finding_recommendations(analysis_result, image_type)

        if response_language == 'en':
            image_type_label = {
                'bone': 'Bone X-ray (wrist)',
                'chest': 'Chest X-ray',
                'dental': 'Dental image',
                'dental_photo': 'Dental photo',
                'dental_xray': 'Dental X-ray',
                'prescription': 'Handwritten prescription',
            }.get(image_type, 'Medical image')

            prompt = f"""You are a medical specialist. Write a concise English report based on the analysis below.

Image type: {image_type_label}
{findings_text}

Write the report in this exact structure (use emojis):

📋 Detected Findings:
[List each detected condition clearly, or state that no strong abnormal finding is detected]

⚠️ Severity: [low/medium/high]

💊 Recommendations:
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

⏰ Follow-up: [When to seek follow-up]

Important rules:
- Use probabilistic wording only (for example: "suggests", "may indicate"). Do not provide definitive diagnosis.
- Never expose technical class IDs like Class_4 or Class 7 in the patient report.
- In "Detected Findings", preserve each provided finding once without duplication.
- If no strong finding exists, mention that clinical follow-up is recommended if symptoms persist.
- If the input includes "📋 توصيات علاجية مقترحة لكل حالة", use those recommendations as the primary basis for the recommendations section.

Keep it concise and clear. Do not use tables."""
        else:
            image_type_label = {
                'bone': 'أشعة عظام (رسغ)',
                'chest': 'أشعة صدر',
                'dental': 'صورة أسنان',
                'dental_photo': 'صورة أسنان',
                'dental_xray': 'أشعة أسنان',
                'prescription': 'روشتة طبية مكتوبة بخط اليد',
            }.get(image_type, 'صورة طبية')

            prompt = f"""أنت طبيب متخصص. اكتب تقرير طبي مختصر بالعربية بناءً على نتائج التحليل.

نوع الصورة: {image_type_label}
{findings_text}

اكتب التقرير بالشكل التالي (استخدم الرموز):

📋 النتائج المكتشفة:
[اذكر كل حالة مكتشفة بالعربي أو اذكر أن كل شيء طبيعي]

⚠️ الشدة: [منخفضة/متوسطة/عالية]

💊 التوصيات:
1. [توصية 1]
2. [توصية 2]
3. [توصية 3]

⏰ المتابعة: [متى يجب المتابعة]

قواعد مهمة جدًا:
- استخدم صياغة احتمالية فقط مثل: "يرجح"، "قد تشير النتائج"، ولا تكتب تشخيصًا مؤكدًا.
- لا تذكر أي أسماء تقنية مثل Class_4 أو Class 7، واعتبرها ضمن "تسوس أسنان".
- في قسم "النتائج المكتشفة" انسخ كل بند من القائمة المعطاة كما هو بدون تكرار وبدون إضافة بنود جديدة. لا تعيد صياغة نفس البند بأكثر من سطر.
- لو لا توجد نتائج قوية، اذكر أن المتابعة السريرية مطلوبة مع إعادة التقييم عند استمرار الأعراض.
- إذا وُجد قسم "📋 توصيات علاجية مقترحة لكل حالة" ضمن البيانات أعلاه، استخدم هذه التوصيات الموصى بها كأساس لقسم "💊 التوصيات" واكتب توصية محددة لكل حالة مكتشفة (بدلاً من التوصيات العامة)، ويمكنك إضافة توصيات سلوكية عامة في النهاية.

اجعل التقرير مختصر وواضح. لا تستخدم جداول."""

    
    elif file_type == 'pdf':
        if 'error' in analysis_result:
            if response_language == 'en':
                return f"File reading error: {analysis_result['error']}"
            return f"خطأ في قراءة الملف: {analysis_result['error']}"

        if response_language == 'en':
            prompt = f"""You are a medical specialist. Analyze and summarize the following medical report in English:

Report content:
{analysis_result.get('text', '')[:3000]}

Provide a concise summary including:
1. Main findings
2. Any abnormal values
3. Recommendations (if available)"""
        else:
            prompt = f"""أنت طبيب متخصص. قم بتحليل التقرير الطبي التالي وتلخيصه باللغة العربية:

محتوى التقرير:
{analysis_result.get('text', '')[:3000]}

قدم ملخصاً مختصراً يتضمن:
1. النتائج الرئيسية
2. أي قيم غير طبيعية
3. التوصيات إن وجدت"""
    
    else:
        return "نوع ملف غير معروف"
    
    return get_llm_response(prompt)


def generate_followup_for_uploaded_file(user_message, base_report, image_type=None, target_language='ar'):
    # Answer an optional user prompt that is sent with an uploaded file.
    if not user_message:
        return ""

    user_message = str(user_message).strip()
    if not user_message:
        return ""

    response_language = resolve_response_language(user_message, target_language)

    if not LLM_AVAILABLE:
        if response_language == 'en':
            return "Your question was received with the file, but the AI service is currently unavailable for a detailed reply."
        return "تم استلام سؤالك مع الملف. الخدمة الذكية غير متاحة حالياً للرد التفصيلي."

    if response_language == 'en':
        image_type_label = {
            'bone': 'Bone X-ray',
            'chest': 'Chest X-ray',
            'dental': 'Dental image/X-ray',
            'dental_photo': 'Dental photo',
            'dental_xray': 'Dental X-ray',
            'prescription': 'Prescription',
            None: 'Medical file',
        }.get(image_type, 'Medical file')

        prompt = f"""You have the following {image_type_label} analysis context:
{base_report}

Attached user question:
{user_message}

Required:
- Answer the question directly and briefly.
- Base your answer only on the provided analysis context.
- If information is uncertain, say so clearly.
- Avoid definitive diagnosis or changing prescribed treatment.
- Reply in the user's language.
"""

        system = (
            "You are a medical explanation assistant. "
            "Provide a concise and safe answer based only on the available analysis, "
            "and do not replace the treating physician's decision."
        )
    else:
        image_type_label = {
            'bone': 'أشعة عظام',
            'chest': 'أشعة صدر',
            'dental': 'صورة/أشعة أسنان',
            'dental_photo': 'صورة أسنان',
            'dental_xray': 'أشعة أسنان',
            'prescription': 'روشتة طبية',
            None: 'ملف طبي',
        }.get(image_type, 'ملف طبي')

        prompt = f"""لديك سياق تحليل {image_type_label} التالي:
{base_report}

سؤال المستخدم المرفق مع الملف:
{user_message}

المطلوب:
- أجب على السؤال مباشرة وباختصار عملي.
- اربط الإجابة بنتيجة التحليل المذكورة أعلاه فقط.
- إذا كانت المعلومة غير مؤكدة من التحليل، اذكر ذلك بوضوح.
- تجنب التشخيص القطعي أو تغيير وصفة الطبيب.
- الرد بنفس لغة المستخدم.
"""

        system = (
            "أنت مساعد طبي توضيحي. "
            "قدم إجابة مختصرة وآمنة بناءً على نتيجة التحليل المتاحة فقط، "
            "ولا تستبدل قرار الطبيب المعالج."
        )

    response = get_llm_response(prompt, system)
    return clean_markdown(resolve_llm_response(response))

