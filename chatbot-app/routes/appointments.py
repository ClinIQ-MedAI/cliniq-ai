import logging
from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from utils.json_store import get_doctors, get_appointments, save_json
from config import Config
from routes.schemas import BookAppointmentRequest, QueueCheckParams, AppointmentStatus
import uuid
import datetime

logger = logging.getLogger(__name__)
appointments_bp = Blueprint('appointments', __name__)

@appointments_bp.route('/doctors', methods=['GET'])
def list_doctors():
    logger.info("Fetching doctors list")
    doctors = get_doctors()
    return jsonify(doctors)

@appointments_bp.route('/appointments/queue', methods=['GET'])
def check_queue():
    try:
        params = QueueCheckParams(**request.args.to_dict())
    except ValidationError as e:
        logger.warning(f"Queue check validation error: {e.errors()}")
        return jsonify({"error": "Invalid parameters", "details": e.errors()}), 400
        
    appointments = get_appointments()
    count = 0
    for appt in appointments:
        # handle legacy records where 'date' might be used instead of 'day'
        appt_day = appt.get('day') or appt.get('date')
        if (str(appt.get('doctor_id')) == str(params.doctor_id) and 
            appt_day == params.day.value and 
            appt.get('time') == params.time):
            count += 1
            
    available = Config.MAX_SLOTS_PER_TIME - count
    
    logger.info(f"Queue checked for doctor_id={params.doctor_id} day={params.day.value} time={params.time}, available={available}")
    return jsonify({
        "queue_position": count + 1,
        "is_available": available > 0,
        "available_slots": available
    })

@appointments_bp.route('/appointments/book', methods=['POST'])
def book_appointment():
    try:
        data = BookAppointmentRequest(**(request.json or {}))
    except ValidationError as e:
        logger.warning(f"Appointment booking validation error: {e.errors()}")
        return jsonify({"error": "Invalid request payload", "details": e.errors()}), 400
        
    doctors = get_doctors()
    doctor = next((d for d in doctors if str(d['id']) == str(data.doctor_id)), None)
    if not doctor:
        logger.warning(f"Booking attempt for non-existent doctor_id={data.doctor_id}")
        return jsonify({"error": "Doctor not found"}), 404
        
    if data.day.value not in doctor['availability']:
        logger.warning(f"Doctor {data.doctor_id} not available on {data.day.value}")
        return jsonify({"error": "Doctor not available on this day"}), 400
        
    if data.time not in doctor['availability'][data.day.value]:
        logger.warning(f"Time slot {data.time} not available for doctor {data.doctor_id} on {data.day.value}")
        return jsonify({"error": "Time slot not available"}), 400
        
    # Queue check
    appointments = get_appointments()
    count = sum(1 for a in appointments 
               if str(a.get('doctor_id')) == str(data.doctor_id) 
               and (a.get('day') or a.get('date')) == data.day.value
               and a.get('time') == data.time)
               
    if count >= Config.MAX_SLOTS_PER_TIME:
        logger.warning(f"Slot {data.day.value} {data.time} for doctor {data.doctor_id} is full")
        return jsonify({"error": "Time slot is fully booked"}), 400
        
    new_appt = {
        "id": str(uuid.uuid4()),
        "patient_id": data.patient_id,
        "patient_name": data.patient_name,
        "doctor_id": data.doctor_id,
        "doctor_name": doctor['name'],
        "day": data.day.value,
        "time": data.time,
        "status": AppointmentStatus.CONFIRMED.value,
        "created_at": datetime.datetime.now().isoformat()
    }
    
    appointments.append(new_appt)
    save_json(Config.APPOINTMENTS_FILE, appointments)
    
    logger.info(f"Appointment booked: {new_appt['id']} for patient={data.patient_name} (id={data.patient_id}) with doctor_id={data.doctor_id}")
    return jsonify({
        "success": True,
        "message": "Appointment booked successfully",
        "appointment": new_appt
    })

@appointments_bp.route('/patient/<patient_id>/appointments', methods=['GET'])
def get_patient_appointments(patient_id):
    logger.info(f"Fetching appointments for patient_id={patient_id}")
    appointments = get_appointments()
    patient_appts = [a for a in appointments if a.get('patient_id') == patient_id]
    return jsonify(patient_appts)

@appointments_bp.route('/appointments/<appt_id>/cancel', methods=['PATCH'])
def cancel_appointment(appt_id):
    logger.info(f"Cancel request for appt_id={appt_id}")
    appointments = get_appointments()
    for appt in appointments:
        if str(appt.get('id')) == appt_id:
            appt['status'] = AppointmentStatus.CANCELLED.value
            save_json(Config.APPOINTMENTS_FILE, appointments)
            logger.info(f"Appointment {appt_id} cancelled successfully.")
            return jsonify({"success": True, "message": "Appointment cancelled."})
            
    logger.warning(f"Appointment {appt_id} not found for cancellation.")
    return jsonify({"error": "Appointment not found"}), 404
