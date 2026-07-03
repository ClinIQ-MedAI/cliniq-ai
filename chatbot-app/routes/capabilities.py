import logging
from flask import Blueprint, jsonify
from services.llm import LLM_AVAILABLE
from routes.schemas import ImageType

logger = logging.getLogger(__name__)
capabilities_bp = Blueprint('capabilities', __name__)

@capabilities_bp.route('/capabilities', methods=['GET'])
def get_capabilities():
    logger.info("Fetching capabilities")
    return jsonify({
        "llm_available": LLM_AVAILABLE,
        "services": {
            "bone_detection": True,
            "oral_detection": True,
            "chest_xray": True,
            "oral_classification": True,
            "prescription_parsing": True
        }
    })
