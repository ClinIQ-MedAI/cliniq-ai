#!/usr/bin/env python3
"""ClinIQ chatbot — Flask app + HTTP routes. All logic lives in core/*."""
from flask import render_template, request, jsonify, Response, stream_with_context

from core.settings import *
from core.db import *
from core.textutil import *
from core.llm import *
from core.chat import *
from core.medical import *

# `import *` skips underscore-prefixed names, so pull this one in explicitly.
from core.medical import _screen_upload

init_chat_db()


@app.route('/')
def index():
    # Serve the main chatbot page.
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    # Handle chat messages with conversation memory per patient (Streaming).
    data = request.json or {}
    user_message = data.get('message', '').strip()
    patient_id = data.get('patient_id', 'anonymous')  # Default to 'anonymous' if not provided
    preferred_language = normalize_language_code(data.get('language_preference', 'ar'))
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    # Get recent history from persistent storage (last 20 messages).
    history = get_chat_messages(patient_id, limit=HISTORY_WINDOW_MESSAGES)
    
    # Classify the query
    query_type = classify_query(user_message)
    response_language = resolve_response_language(user_message, preferred_language)
    
    def generate():
        # Route to appropriate handler with history
        if query_type == "upload":
            # For upload prompt, we just return the string immediately
            gen_response = build_upload_guidance_message(response_language)
        elif query_type == "health":
            gen_response = handle_health_question(user_message, history, target_language=response_language)
        elif query_type == "appointment":
            gen_response = handle_appointment_request(user_message, history, target_language=response_language)
        elif query_type == "availability":
            gen_response = handle_availability_query(user_message, history, target_language=response_language)
        elif query_type == "faq":
            gen_response = handle_faq(user_message, history, target_language=response_language)
        else:
            if response_language == 'en':
                gen_response = "I'm not sure how to help with that. Try asking about symptoms, booking an appointment, or general clinic information."
            else:
                gen_response = "غير متأكد كيف أساعدك في ذلك الآن. يمكنك السؤال عن الأعراض، أو حجز موعد، أو معلومات العيادة العامة."
        
        full_response_text = ""
        
        # Stream the response
        try:
            if isinstance(gen_response, str):
                full_response_text = gen_response
                # Yield single chunk for static text
                yield json.dumps({"chunk": gen_response}) + "\n"
            else:
                # Iterate generator for streaming content
                for chunk in gen_response:
                    full_response_text += chunk
                    yield json.dumps({"chunk": chunk}) + "\n"
        except Exception as e:
            error_chunk = f"\nError generating response: {str(e)}"
            full_response_text += error_chunk
            yield json.dumps({"chunk": error_chunk}) + "\n"
        
        # Store the exchange in persistent history.
        save_chat_message(patient_id, "user", user_message)
        save_chat_message(patient_id, "assistant", full_response_text)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        # Final detailed JSON with done=true
        yield json.dumps({
            "done": True,
            "response": full_response_text,
            "query_type": query_type,
            "patient_id": patient_id,
            "show_upload": query_type == "upload"
        }) + "\n"

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


@app.route('/api/doctors', methods=['GET'])
def get_doctors():
    # Get list of all doctors.
    doctors = load_json(DOCTORS_FILE)
    return jsonify(doctors)


@app.route('/api/appointments/book', methods=['POST'])
def book_appointment():
    # Book an appointment.
    data = request.json
    
    appointment = {
        "id": len(load_json(APPOINTMENTS_FILE)) + 1,
        "patient_id": data.get('patient_id'),  # Link to patient
        "patient_name": data.get('patient_name'),
        "doctor_id": data.get('doctor_id'),
        "date": data.get('date'),
        "time": data.get('time'),
        "booked_at": datetime.now().isoformat()
    }
    
    appointments = load_json(APPOINTMENTS_FILE)
    appointments.append(appointment)
    save_json(APPOINTMENTS_FILE, appointments)
    
    return jsonify({
        "success": True,
        "message": "Appointment booked successfully",
        "appointment": appointment
    })


@app.route('/api/appointments/queue', methods=['GET'])
def get_queue():
    # Get queue information for a specific doctor and time.
    doctor_id = request.args.get('doctor_id', type=int)
    date = request.args.get('date')
    time = request.args.get('time')
    
    appointments = load_json(APPOINTMENTS_FILE)
    
    queue_count = len([
        apt for apt in appointments
        if apt['doctor_id'] == doctor_id and apt['date'] == date and apt['time'] == time
    ])
    
    return jsonify({
        "queue_position": queue_count + 1,
        "people_before_you": queue_count
    })


@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    # Clear conversation history for a patient.
    data = request.json
    patient_id = data.get('patient_id', 'anonymous')

    clear_chat_messages(patient_id)
    
    return jsonify({
        "success": True,
        "message": "Conversation history cleared",
        "patient_id": patient_id
    })


@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    # Get conversation history for a patient.
    patient_id = request.args.get('patient_id', 'anonymous')

    history = get_chat_messages(patient_id)
    
    return jsonify({
        "patient_id": patient_id,
        "history": history,
        "message_count": len(history)
    })


@app.route('/api/patient/<patient_id>/appointments', methods=['GET'])
def get_patient_appointments(patient_id):
    # Get appointments for a specific patient.
    appointments = load_json(APPOINTMENTS_FILE)
    
    patient_appointments = [
        apt for apt in appointments
        if apt.get('patient_id') == patient_id
    ]
    
    return jsonify({
        "patient_id": patient_id,
        "appointments": patient_appointments,
        "count": len(patient_appointments)
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Handle file uploads (images and PDFs) for medical analysis.
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "لم يتم اختيار ملف"}), 400
    
    patient_id = request.form.get('patient_id', 'anonymous')
    user_message = request.form.get('user_message', '').strip()
    image_type = request.form.get('image_type', 'dental')  # 'dental' or 'bone'
    language_preference = normalize_language_code(request.form.get('language_preference', 'ar'))
    response_language = resolve_response_language(user_message, language_preference)
    
    filename = file.filename.lower()
    file_bytes = file.read()

    # DICOM (.dcm) uploads are transcoded to a display-ready PNG once, up front,
    # so every downstream step (API analysis, overlay, report) sees a normal
    # image. dicom_meta carries the scan's header (modality, body part, ...).
    dicom_meta = {}
    if DICOM_SUPPORT:
        try:
            file_bytes, dicom_meta = normalize_medical_image(file_bytes, filename)
        except DicomError as exc:
            error_msg = (f"Could not read the DICOM file: {exc}" if response_language == 'en'
                         else f"تعذّر قراءة ملف DICOM: {exc}")
            return jsonify({"error": error_msg}), 400
        if dicom_meta:
            filename = "upload.png"  # route through the normal image pipeline below

    # Determine file type and process
    if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        if not IMAGE_SUPPORT:
            return jsonify({"error": "دعم الصور غير متاح"}), 500

        # OOD / input gate: reject uploads that clearly aren't the selected scan
        # type (e.g. a colour selfie sent as a "bone X-ray") before any model runs.
        gate = _screen_upload(file_bytes, image_type)
        if gate is not None and not gate.passed:
            reject_report = (
                f"⚠️ This image doesn't look like a valid {image_type.replace('_', ' ')} "
                f"scan ({gate.reason}). Please upload the correct, clear image."
                if response_language == 'en' else
                f"⚠️ الصورة دي شكلها مش مناسب لنوع الفحص المختار ({image_type}). "
                f"من فضلك ارفع صورة صحيحة وواضحة للفحص المطلوب."
            )
            save_chat_message(patient_id, "user", "[محاولة رفع صورة غير صالحة]")
            save_chat_message(patient_id, "assistant", reject_report)
            return jsonify({
                "response": reject_report,
                "input_rejected": True,
                "gate": gate.to_dict(),
                "severity": "low",
            })

        # Analyze image with appropriate API
        analysis_result = analyze_image_with_api(file_bytes, image_type)

        # Prefer API-provided annotated image (rendered with cv2 on the server,
        # handles 16-bit X-rays correctly and includes class labels + confidences).
        result_image = analysis_result.get('annotated_image_base64') or \
                       analysis_result.get('annotated_image')

        # Fallback: render overlay locally if API did not provide one.
        if not result_image and image_type in {'bone', 'dental', 'dental_photo', 'dental_xray'}:
            detections = parse_detections_for_overlay(analysis_result)
            if not detections:
                detections = fetch_detection_overlay_detections(file_bytes, image_type)
            result_image = draw_detections_on_image(file_bytes, detections, image_type=image_type)
        
        # Generate report
        report = resolve_llm_response(
            generate_medical_report(analysis_result, 'image', image_type, target_language=response_language)
        )
        report = clean_markdown(report)

        # If the user attached a text prompt with the uploaded image,
        # answer it as a second section after the standard analysis report.
        if user_message:
            followup = generate_followup_for_uploaded_file(
                user_message=user_message,
                base_report=report,
                image_type=image_type,
                target_language=response_language,
            )
            if followup:
                followup_heading = "💬 Reply to your message:" if response_language == 'en' else "💬 رد على رسالتك:"
                report = f"{report}\n\n{followup_heading}\n{followup}"
        
        # Store in persistent conversation history
        user_msg = f"[تم رفع صورة أشعة {'عظام' if image_type == 'bone' else 'أسنان'}]"
        if user_message:
            user_msg += f"\n[رسالة مرفقة]: {user_message}"
        save_chat_message(patient_id, "user", user_msg)
        save_chat_message(patient_id, "assistant", report)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        # Detect severity from analysis result
        severity = "low"
        urgency = str(analysis_result.get("urgency", "")).lower()
        if urgency in ["high", "critical", "مرتفع", "حرج"]:
            severity = "high"
        elif urgency in ["moderate", "medium", "متوسط"]:
            severity = "medium"
        else:
            ai_findings = analysis_result.get("ai_findings")
            if isinstance(ai_findings, list):
                severities = [str(item.get("severity", "")).upper() for item in ai_findings if isinstance(item, dict)]
                if any(s in ["HIGH", "CRITICAL"] for s in severities):
                    severity = "high"
                elif any(s in ["MODERATE", "MEDIUM"] for s in severities):
                    severity = "medium"
            elif isinstance(ai_findings, dict):
                sev = str(ai_findings.get("severity", "")).upper()
                if sev in ["HIGH", "CRITICAL"]:
                    severity = "high"
                elif sev in ["MODERATE", "MEDIUM"]:
                    severity = "medium"
            else:
                confidence_raw = analysis_result.get('confidence', 0)
                predicted_class = str(analysis_result.get('predicted_class', '')).lower()
                try:
                    if isinstance(confidence_raw, str):
                        confidence = float(confidence_raw.replace('%', '').strip())
                    else:
                        confidence = float(confidence_raw)
                except (ValueError, TypeError):
                    confidence = 0.0

                high_severity_conditions = ['caries', 'تسوس', 'fracture', 'كسر', 'gingivitis', 'التهاب']
                is_severe_condition = any(cond in predicted_class for cond in high_severity_conditions)

                if confidence > 80 and is_severe_condition:
                    severity = "high"
                elif confidence > 50:
                    severity = "medium"
        
        # Add follow-up suggestion based on severity
        # Note: booking question is handled by the frontend via suggest_booking flag
        # Only add care tips for medium severity (not booking prompt to avoid duplicates)
        if severity == "medium":
            if response_language == 'en':
                follow_up = "💡 Care tips: Maintain good oral hygiene, use mouthwash, and follow up with a doctor soon."
            else:
                follow_up = "💡 نصائح للعناية: حافظ على نظافة الأسنان واستخدم غسول الفم. تابع مع طبيب في أقرب وقت مناسب."
            report += f"\n\n{follow_up}"
        
        # Allow GradCAM visualization for chest X-ray only.
        gradcam_image = analysis_result.get('gradcam_image_base64', None) if image_type == 'chest' else None
        
        return jsonify({
            "success": True,
            "file_type": "image",
            "image_type": image_type,
            "analysis": analysis_result,
            "report": report,
            "patient_id": patient_id,
            "severity": severity,
            "suggest_booking": severity == "high",
            "gradcam_image": gradcam_image,
            "result_image": result_image
        })
    
    elif allowed_file(filename, ALLOWED_PDF_EXTENSIONS):
        # Extract text from PDF
        extraction_result = extract_pdf_text(file_bytes)
        
        if 'error' in extraction_result:
            return jsonify(extraction_result), 400
        
        # Generate summary report
        report = resolve_llm_response(
            generate_medical_report(extraction_result, 'pdf', target_language=response_language)
        )
        report = clean_markdown(report)

        if user_message:
            followup = generate_followup_for_uploaded_file(
                user_message=user_message,
                base_report=report,
                image_type=None,
                target_language=response_language,
            )
            if followup:
                followup_heading = "💬 Reply to your message:" if response_language == 'en' else "💬 رد على رسالتك:"
                report = f"{report}\n\n{followup_heading}\n{followup}"
        
        # Store in persistent conversation history
        user_msg = f"[تم رفع ملف PDF: {extraction_result.get('pages', 0)} صفحات]"
        if user_message:
            user_msg += f"\n[رسالة مرفقة]: {user_message}"
        save_chat_message(patient_id, "user", user_msg)
        save_chat_message(patient_id, "assistant", report)
        trim_chat_messages(patient_id, MAX_HISTORY_MESSAGES)
        
        return jsonify({
            "success": True,
            "file_type": "pdf",
            "pages": extraction_result.get('pages', 0),
            "text_preview": extraction_result.get('text', '')[:500] + "...",
            "report": report,
            "patient_id": patient_id
        })
    
    else:
        return jsonify({
            "error": f"نوع ملف غير مدعوم. الأنواع المدعومة: {', '.join(ALLOWED_IMAGE_EXTENSIONS | ALLOWED_PDF_EXTENSIONS)}"
        }), 400


@app.route('/api/prescription/status', methods=['GET'])
def prescription_status():
    # Proxy to the prescription parser /status so the UI can poll progress.
    try:
        r = requests.get(f"{PRESCRIPTION_API}/status", timeout=5)
        if r.status_code == 200:
            return jsonify(r.json())
        return jsonify({"stage": "unknown", "message": f"HTTP {r.status_code}"}), 200
    except Exception as e:
        return jsonify({"stage": "unreachable", "message": str(e)}), 200


@app.route('/api/capabilities', methods=['GET'])
def get_capabilities():
    # Get chatbot capabilities and available features.
    return jsonify({
        "image_support": IMAGE_SUPPORT,
        "pdf_support": PDF_SUPPORT,
        "llm_available": LLM_AVAILABLE,
        "bone_detect_api": BONE_DETECT_API,
        "oral_detect_api": ORAL_DETECT_API,
        "oral_classify_api": ORAL_CLASSIFY_API,
        "chest_xray_api": CHEST_XRAY_API,
        "allowed_image_types": list(ALLOWED_IMAGE_EXTENSIONS),
        "allowed_pdf_types": list(ALLOWED_PDF_EXTENSIONS),
        "max_file_size_mb": 64
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
