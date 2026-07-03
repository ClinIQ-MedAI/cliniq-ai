import logging
import json
from flask import Blueprint, request, jsonify, Response
from pydantic import ValidationError
from utils.db import save_chat_message, get_chat_messages, clear_chat_messages, trim_chat_messages
from utils.language import resolve_response_language, get_language_policy, build_upload_guidance_message
from services.classifier import classify_query
from services.llm import get_llm_response
from utils.json_store import get_faqs, get_doctors
from routes.schemas import ChatRequest, QueryType, ChatRole

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = ChatRequest(**(request.json or {}))
    except ValidationError as e:
        logger.warning(f"Chat request validation error: {e.errors()}")
        return jsonify({"error": "Invalid request payload", "details": e.errors()}), 400

    user_message = data.message.strip()
    patient_id = data.patient_id
    language_preference = data.language_preference

    target_lang = resolve_response_language(user_message, language_preference)
    query_type_str = classify_query(user_message)
    
    # Try to map the string to Enum safely, fallback to HEALTH
    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        logger.warning(f"Unknown query type returned by classifier: {query_type_str}")
        query_type = QueryType.HEALTH

    if user_message:
        logger.debug(f"Saving user message for patient_id={patient_id}")
        save_chat_message(patient_id, ChatRole.USER.value, user_message)

    if query_type == QueryType.UPLOAD:
        bot_reply = build_upload_guidance_message(target_lang)
        save_chat_message(patient_id, ChatRole.BOT.value, bot_reply)
        return jsonify({
            "response": bot_reply,
            "query_type": query_type.value
        })

    def generate_response():
        history = get_chat_messages(patient_id)
        
        system_message = (
            "You are ClinIQ AI, an empathetic and professional bilingual medical assistant for the ClinIQ platform. "
            "Your main goals are answering health queries, providing clinic info, and assisting with appointments.\n"
            f"{get_language_policy(target_lang)}\n"
        )
        
        if query_type == QueryType.FAQ:
            faqs = get_faqs()
            faq_text = "\n".join([f"Q: {f['question']} A: {f['answer']}" for f in faqs])
            system_message += f"\nHere is the clinic FAQ:\n{faq_text}"
        elif query_type in (QueryType.APPOINTMENT, QueryType.AVAILABILITY):
            doctors = get_doctors()
            doctors_info = []
            for d in doctors:
                sched = ", ".join([f"{day} ({', '.join(times)})" for day, times in d['availability'].items()])
                doctors_info.append(f"Dr. {d['name']} ({d['specialty']}): {sched}")
            system_message += f"\nClinic Doctors:\n" + "\n".join(doctors_info)
            system_message += "\nTell the user they can use the 'Book Appointment' button or ask you to book to open the booking menu."

        full_response = ""
        try:
            for chunk in get_llm_response(user_message, system_message=system_message, history=history):
                if chunk:
                    full_response += chunk
                    payload = json.dumps({"chunk": chunk, "done": False})
                    yield f"{payload}\n"
        except Exception as e:
            logger.error(f"Error streaming LLM response for patient_id={patient_id}: {str(e)}")
            payload = json.dumps({"chunk": " [Error generating response] ", "done": True, "error": str(e)})
            yield f"{payload}\n"
            return
            
        save_chat_message(patient_id, ChatRole.BOT.value, full_response)
        trim_chat_messages(patient_id)
        
        logger.info(f"Completed chat response for patient_id={patient_id}, query_type={query_type.value}")
        final_payload = json.dumps({
            "chunk": "",
            "done": True,
            "response": full_response,
            "query_type": query_type.value
        })
        yield f"{final_payload}\n"

    logger.info(f"Starting chat stream for patient_id={patient_id}, query_type={query_type.value}")
    return Response(generate_response(), mimetype='application/x-ndjson')

@chat_bp.route('/chat/history', methods=['GET'])
def get_history():
    patient_id = request.args.get('patient_id', 'anonymous')
    try:
        patient_id = ChatRequest(patient_id=patient_id).patient_id
    except ValidationError:
        pass
    
    logger.info(f"Fetching chat history for patient_id={patient_id}")
    messages = get_chat_messages(patient_id)
    return jsonify({"history": messages})

@chat_bp.route('/chat/clear', methods=['POST'])
def clear_chat():
    data = request.json or {}
    patient_id = data.get('patient_id', 'anonymous')
    try:
        patient_id = ChatRequest(patient_id=patient_id).patient_id
    except ValidationError:
        pass
        
    logger.info(f"Clearing chat history for patient_id={patient_id}")
    clear_chat_messages(patient_id)
    return jsonify({"success": True})
