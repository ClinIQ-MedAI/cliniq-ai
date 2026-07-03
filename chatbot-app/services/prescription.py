import re

def build_prescription_report(analysis_result):
    """Build a structured Arabic report for the prescription parser output.

    Skips the LLM since the VLM already produced verified structured data.
    """
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
