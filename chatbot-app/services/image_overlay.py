import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def _bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def dedup_detections_by_iou(detections, iou_thresh=0.6, conf_thresh=0.0):
    filtered = [d for d in detections if d.get('confidence', 1.0) >= conf_thresh]
    filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    kept = []
    for d in filtered:
        box = d.get('bbox')
        if not box or len(box) != 4:
            kept.append(d)
            continue
        overlap = False
        for k in kept:
            kbox = k.get('bbox')
            if kbox and len(kbox) == 4 and d.get('label') == k.get('label'):
                if _bbox_iou(box, kbox) > iou_thresh:
                    overlap = True
                    break
        if not overlap:
            kept.append(d)
    return kept

def confidence_to_percent(conf):
    if conf is None:
        return "?"
    try:
        val = float(conf)
        if val <= 1.0:
            return f"{int(val * 100)}%"
        return f"{int(val)}%"
    except (ValueError, TypeError):
        return "?"

def parse_detections_for_overlay(analysis_result, image_type):
    from services.medical_api import fetch_detection_overlay_detections
    if 'detections' in analysis_result and isinstance(analysis_result['detections'], list):
        return [
            {
                "bbox": det.get('bbox'),
                "label": det.get('class_name') or det.get('finding') or "Finding",
                "confidence": det.get('confidence')
            }
            for det in analysis_result['detections']
            if isinstance(det, dict)
        ]
    return []

def prepare_detections_for_overlay(detections, image_type):
    from config import Config
    from utils.labels import normalize_oral_label_ar
    valid = []
    for d in detections:
        if not isinstance(d, dict):
            continue
        box = d.get("bbox")
        label = d.get("label") or d.get("finding") or d.get("class_name")
        conf = d.get("confidence", 0.0)
        
        if not box or not isinstance(box, list) or len(box) != 4:
            continue
            
        try:
            box = [float(b) for b in box]
        except (ValueError, TypeError):
            continue
            
        valid.append({"bbox": box, "label": label, "confidence": float(conf)})

    valid = dedup_detections_by_iou(valid, iou_thresh=0.7)
    
    max_d = Config.OVERLAY_MAX_DETECTIONS_BY_IMAGE_TYPE.get(image_type, Config.OVERLAY_DEFAULT_MAX_DETECTIONS)
    valid.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    valid = valid[:max_d]
    
    # Normalize labels
    for d in valid:
        original = str(d.get('label', ''))
        if image_type in ['dental', 'dental_xray', 'dental_photo']:
            d['display_label'] = normalize_oral_label_ar(original)
        else:
            d['display_label'] = original
            
    return valid

def draw_detections_on_image(image_bytes, detections):
    if not detections:
        return base64.b64encode(image_bytes).decode('utf-8')
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # High bit-depth fix
        if img.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            try:
                np_img = np.array(img, dtype=np.float32)
                min_val, max_val = np_img.min(), np_img.max()
                if max_val > min_val:
                    np_img = (np_img - min_val) / (max_val - min_val) * 255.0
                else:
                    np_img = np_img * 0
                img = Image.fromarray(np_img.astype(np.uint8))
            except Exception as e:
                print(f"16-bit to 8-bit normalization failed: {e}")
                
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        draw = ImageDraw.Draw(img, "RGBA")
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            
        colors = [
            (255, 65, 54),   # Red
            (46, 204, 64),   # Green
            (0, 116, 217),   # Blue
            (255, 133, 27),  # Orange
            (177, 13, 201),  # Purple
            (57, 204, 204)   # Teal
        ]
        
        label_colors = {}
        for d in detections:
            bbox = d.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
                
            label = str(d.get('display_label') or d.get('label', ''))
            conf = d.get('confidence')
            
            if label not in label_colors:
                label_colors[label] = colors[len(label_colors) % len(colors)]
            color = label_colors[label]
            
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            fill_color = color + (60,)
            draw.rectangle([x1, y1, x2, y2], fill=fill_color)
            
            text = label
            if conf is not None:
                text += f" {confidence_to_percent(conf)}"
                
            try:
                bbox = draw.textbbox((x1, max(0, y1 - 25)), text, font=font)
                draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
            except AttributeError:
                tw, th = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (100, 20)
                draw.rectangle([x1, max(0, y1 - 25), x1 + tw + 4, max(0, y1 - 25) + th + 4], fill=color)
                
            draw.text((x1 + 2, max(0, y1 - 25)), text, fill=(255, 255, 255), font=font)
            
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    except Exception as e:
        print(f"Error drawing overlay: {str(e)}")
        return base64.b64encode(image_bytes).decode('utf-8')
