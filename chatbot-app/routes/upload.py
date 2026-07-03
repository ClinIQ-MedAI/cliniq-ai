import logging
import os
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from utils.db import save_chat_message
from services.medical_api import analyze_image_with_api
from services.image_overlay import parse_detections_for_overlay, prepare_detections_for_overlay, draw_detections_on_image
from services.report_builder import generate_medical_report, generate_followup_for_uploaded_file
from services.prescription import build_prescription_report
from utils.language import resolve_response_language
from routes.schemas import UploadRequest, ImageType, ChatRole
from config import Config

logger = logging.getLogger(__name__)
upload_bp = Blueprint('upload', __name__)

def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in Config.ALLOWED_IMAGE_EXTENSIONS or ext in Config.ALLOWED_PDF_EXTENSIONS

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.warning("Upload rejected: No file part")
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.warning("Upload rejected: No selected file")
        return jsonify({"error": "No selected file"}), 400
        
    if len(file.filename) > 255:
        logger.warning("Upload rejected: Filename too long")
        return jsonify({"error": "Filename is too long"}), 400

    if not allowed_file(file.filename):
        logger.warning(f"Upload rejected: Invalid file extension for {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400
        
    try:
        data = UploadRequest(**request.form.to_dict())
    except ValidationError as e:
        logger.warning(f"Upload form validation error: {e.errors()}")
        return jsonify({"error": "Invalid form data", "details": e.errors()}), 400

    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    if file_length > Config.MAX_CONTENT_LENGTH:
        logger.warning(f"Upload rejected: File size ({file_length}) exceeds MAX_CONTENT_LENGTH")
        return jsonify({"error": "File exceeds maximum allowed size"}), 413
    file.seek(0)
    
    file_bytes = file.read()
    
    logger.info(f"Processing upload for patient_id={data.patient_id}, image_type={data.image_type.value}, size={len(file_bytes)}")
    
    # Route to medical API
    analysis_result = analyze_image_with_api(file_bytes, data.image_type.value)
    if "error" in analysis_result:
        logger.error(f"Medical API error: {analysis_result['error']}")
        return jsonify({"error": analysis_result["error"]}), 500

    overlay_base64 = None
    target_lang = resolve_response_language(data.user_message, data.language_preference)

    # 1. Overlay generation (for image types)
    if data.image_type != ImageType.PRESCRIPTION:
        detections = parse_detections_for_overlay(analysis_result, data.image_type.value)
        if detections:
            valid_detections = prepare_detections_for_overlay(detections, data.image_type.value)
            overlay_base64 = draw_detections_on_image(file_bytes, valid_detections)
            
    # 2. Report generation
    if data.image_type == ImageType.PRESCRIPTION:
        report_md = build_prescription_report(analysis_result)
        if data.user_message:
            report_md += f"\n\nأجاب النظام عن تقريرك، لكن بخصوص سؤالك: '{data.user_message}' - يُرجى سؤال المساعد في رسالة نصية منفصلة لاحقًا."
    else:
        report_md = generate_medical_report(analysis_result, data.image_type.value, patient_message=data.user_message)
        
    # Save to chat history
    save_chat_message(data.patient_id, ChatRole.BOT.value, report_md)

    logger.info(f"Upload processed successfully for patient_id={data.patient_id}")
    response_data = {
        "success": True,
        "message": "File analyzed successfully",
        "report": report_md,
        "image_type": data.image_type.value
    }
    
    if overlay_base64:
        response_data['overlay_image'] = f"data:image/jpeg;base64,{overlay_base64}"
        
    return jsonify(response_data)

@upload_bp.route('/prescription/status', methods=['GET'])
def get_prescription_status():
    logger.info("Prescription status checked")
    return jsonify({"status": "completed"})
