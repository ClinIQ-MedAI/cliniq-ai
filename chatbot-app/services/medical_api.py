import requests
from config import Config

def analyze_image_with_api(image_bytes, image_type):
    """Send image to appropriate medical API for analysis."""
    try:
        if image_type == 'bone':
            api_url = f"{Config.BONE_DETECT_API}/predict_for_llm"
        elif image_type == 'dental_photo':
            # Intraoral photos -> ConvNeXt classifier with GradCAM-derived bbox
            api_url = f"{Config.ORAL_CLASSIFY_API}/predict_for_llm"
        elif image_type in ['dental', 'dental_xray']:
            # Panoramic dental X-rays -> YOLO detection
            api_url = f"{Config.ORAL_DETECT_API}/predict_for_llm"
        elif image_type == 'chest':
            api_url = f"{Config.CHEST_XRAY_API}/predict_for_llm?include_gradcam=true"
        elif image_type == 'prescription':
            # Handwritten prescription -> Qwen2-VL VLM + RapidFuzz against Egyptian drugs DB
            api_url = f"{Config.PRESCRIPTION_API}/predict_for_llm"
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

def fetch_detection_overlay_detections(image_bytes, image_type):
    """Fetch bbox detections from detection APIs for overlay if missing."""
    if image_type == 'bone':
        api_url = f"{Config.BONE_DETECT_API}/predict"
    elif image_type == 'dental_photo':
        api_url = f"{Config.ORAL_CLASSIFY_API}/predict_for_llm"
    elif image_type in ['dental', 'dental_xray']:
        api_url = f"{Config.ORAL_DETECT_API}/predict"
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
