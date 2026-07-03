from utils.labels import normalize_oral_label_ar, normalize_chest_label_ar
from services.image_overlay import dedup_detections_by_iou, confidence_to_percent

FOLLOW_UP_TIPS = {
    "bone": {
        "en": "💡 Care tips: Keep the affected area immobilized, apply ice, and follow up with an orthopedist.",
        "ar": "💡 نصائح: حافظ على تثبيت المنطقة المصابة، وضع ثلج، وتابع مع طبيب عظام."
    },
    "chest": {
        "en": "💡 Care tips: Monitor your breathing, avoid smoking or dusty environments, and follow up with a pulmonologist.",
        "ar": "💡 نصائح: راقب تنفسك، تجنب التدخين أو الأماكن المليئة بالغبار، وتابع مع طبيب أمراض صدرية."
    },
    "dental_photo": {
        "en": "💡 Care tips: Maintain good oral hygiene (brush twice daily, floss). Use prescribed mouthwash if provided.",
        "ar": "💡 نصائح: حافظ على نظافة الفم (غسيل الأسنان مرتين واستخدام الخيط). استخدم غسول الفم إذا تم وصفه."
    },
    "dental_xray": {
        "en": "💡 Care tips: Maintain good oral hygiene (brush twice daily, floss). Use prescribed mouthwash if provided.",
        "ar": "💡 نصائح: حافظ على نظافة الفم (غسيل الأسنان مرتين واستخدام الخيط). استخدم غسول الفم إذا تم وصفه."
    },
    "prescription": {
        "en": "💡 Care tips: Take medication exactly as prescribed. Do not skip doses.",
        "ar": "💡 نصائح: تناول الأدوية تماماً كما وصفها الطبيب. لا تفوت الجرعات."
    }
}

PER_FINDING_RECOMMENDATIONS_AR = {
    # Dental
    'تسوس أسنان': 'يُنصح بزيارة طبيب الأسنان لإزالة التسوس وعمل حشوة مناسبة قبل أن يصل للعصب.',
    'حشوة عصب (علاج جذر)': 'تحتاج المتابعة مع الطبيب للتأكد من نجاح العلاج ووضع تاج لحماية السن إن لزم الأمر.',
    'جير على الأسنان': 'يجب إجراء تنظيف جير احترافي بالعيادة لتجنب التهابات اللثة وتآكل العظم.',
    'التهاب اللثة': 'الالتزام بتفريش الأسنان مرتين واستخدام غسول مطهر، مع حجز موعد فحص للثة.',
    'سن مفقود': 'يُنصح باستشارة طبيب زراعة أو تركيبات لتعويض السن المفقود لمنع تحرك الأسنان المجاورة.',
    'قرحة فموية': 'استخدام مرهم موضعي للتقرحات وتجنب الأطعمة الحارة. إذا استمرت لأكثر من أسبوعين يجب فحصها.',
    
    # Bone
    'fracture': 'مؤشر لكسر أو شق عظمي؛ يجب التوجه الفوري للطوارئ لعمل جبيرة أو تثبيت.',
    'dislocation': 'خلع في المفصل؛ يتطلب رداً فورياً في المستشفى لتقليل تلف الأنسجة والأعصاب.',
    
    # Chest
    'التهاب رئوي': 'يُرجى العرض على طبيب أمراض صدرية فوراً لتقييم الحاجة لمضادات حيوية.',
    'استرواح صدري': 'حالة طارئة تتطلب تدخلاً طبياً عاجلاً لتقييم ضغط الرئة.',
    'تضخم القلب': 'يجب مراجعة طبيب قلب لعمل إيكو وتخطيط قلب.',
    'وذمة رئوية': 'يُرجى زيارة قسم الطوارئ لتقييم وظائف القلب والتنفس.',
}

def build_per_finding_recommendations(finding_labels):
    lines = []
    if not finding_labels:
        return lines
        
    lines.append("📋 نصائح طبية مقترحة بناءً على النتائج:")
    for label in sorted(set(finding_labels)):
        rec = PER_FINDING_RECOMMENDATIONS_AR.get(label)
        if rec:
            lines.append(f"  • {label}: {rec}")
    if len(lines) == 1:
        lines = []
    return lines

def generate_medical_report(analysis_result, image_type, patient_message=None):
    from services.llm import get_llm_response, resolve_llm_response
    
    if "error" in analysis_result:
        return f"عذراً، حدث خطأ أثناء تحليل الصورة: {analysis_result['error']}"
        
    ai_findings = analysis_result.get("ai_findings", {})
    detections = analysis_result.get("detections", [])
    
    filtered_detections = dedup_detections_by_iou(detections, iou_thresh=0.6, conf_thresh=0.3)
    filtered_detections.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    report_lines = []
    
    # Header
    if image_type in ['dental_photo', 'dental_xray', 'dental']:
        report_lines.append("📋 **تقرير فحص الأسنان المبدئي (بواسطة AI)**")
        report_lines.append(f"• نوع الصورة: {'أشعة بانوراما' if image_type == 'dental_xray' else 'صورة فموية مباشرة'}")
    elif image_type == 'bone':
        report_lines.append("📋 **تقرير فحص العظام المبدئي (بواسطة AI)**")
        report_lines.append("• نوع الصورة: أشعة عظام")
    elif image_type == 'chest':
        report_lines.append("📋 **تقرير فحص الصدر المبدئي (بواسطة AI)**")
        report_lines.append("• نوع الصورة: أشعة صدر (X-ray)")
    else:
        report_lines.append("📋 **تقرير طبي مبدئي (بواسطة AI)**")
        
    report_lines.append("")
    report_lines.append("🔍 **النتائج:**")
    
    patient_friendly_labels = []
    
    if filtered_detections:
        for i, det in enumerate(filtered_detections, 1):
            label = det.get('label') or det.get('class_name') or 'Finding'
            conf = det.get('confidence')
            
            if image_type in ['dental', 'dental_xray', 'dental_photo']:
                ar_label = normalize_oral_label_ar(label)
            elif image_type == 'chest':
                ar_label = normalize_chest_label_ar(label)
            else:
                ar_label = label
                
            patient_friendly_labels.append(ar_label)
            conf_str = confidence_to_percent(conf)
            report_lines.append(f"{i}. {ar_label} (ثقة: {conf_str})")
    elif ai_findings:
        for k, v in ai_findings.items():
            if isinstance(v, list) and v:
                report_lines.append(f"• {k}: {', '.join(map(str, v))}")
                patient_friendly_labels.extend([str(x) for x in v])
            elif v and not isinstance(v, dict):
                report_lines.append(f"• {k}: {v}")
                patient_friendly_labels.append(str(v))
    else:
        report_lines.append("✅ لم يتم رصد أي مؤشرات غير طبيعية بوضوح في الصورة.")
        report_lines.append("ومع ذلك، يرجى استشارة الطبيب للتشخيص النهائي.")

    report_lines.append("")
    
    rec_lines = build_per_finding_recommendations(patient_friendly_labels)
    if rec_lines:
        report_lines.extend(rec_lines)
        report_lines.append("")
        
    base_report = "\n".join(report_lines)
    
    # System prompt for LLM processing
    system_prompt = (
        "You are ClinIQ AI, a highly empathetic and professional bilingual medical assistant. "
        "Your task is to take the raw medical detection report below and produce a well-formatted, patient-friendly response.\n\n"
        "Guidelines:\n"
        "1. Write predominantly in Arabic unless the user requested English.\n"
        "2. Add a brief, empathetic opening.\n"
        "3. Present the findings clearly using bullet points.\n"
        "4. DO NOT invent medical findings that are not in the raw report.\n"
        "5. ALWAYS include this disclaimer at the end: 'هذا تحليل ذكاء اصطناعي مبدئي ولا يغني عن استشارة الطبيب المتخصص.'\n"
    )
    if patient_message:
        system_prompt += f"\n6. The user also asked this specific question about the scan: '{patient_message}'. Answer this question in your report contextually."
    
    system_prompt += f"\n\nRaw Report:\n{base_report}"
    
    llm_output = resolve_llm_response(get_llm_response("Please format this medical report for the patient.", system_message=system_prompt))
    return llm_output

def generate_followup_for_uploaded_file(image_type, language="ar"):
    tips = FOLLOW_UP_TIPS.get(image_type) or FOLLOW_UP_TIPS["dental_photo"]
    return tips.get(language, tips["ar"])
