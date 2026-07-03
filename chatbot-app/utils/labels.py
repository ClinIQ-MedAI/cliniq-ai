import re

def normalize_oral_label_ar(label):
    """Normalize technical oral labels to patient-friendly Arabic terms."""
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
    """Translate chest class labels to Arabic-friendly names."""
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
