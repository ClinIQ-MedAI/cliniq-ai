from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional
import re
import bleach

class QueryType(str, Enum):
    HEALTH = "health"
    FAQ = "faq"
    APPOINTMENT = "appointment"
    AVAILABILITY = "availability"
    UPLOAD = "upload"

class ChatRole(str, Enum):
    USER = "user"
    BOT = "bot"

class AppointmentStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"

class ImageType(str, Enum):
    DENTAL_XRAY = "dental_xray"
    DENTAL_PHOTO = "dental_photo"
    BONE = "bone"
    CHEST_XRAY = "chest_xray"
    ORAL_CLASSIFICATION = "oral_classification"
    PRESCRIPTION = "prescription"

class WeekDay(str, Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"

class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=2000)
    patient_id: str = Field(default="anonymous", max_length=100)
    language_preference: str = Field(default="ar", max_length=10)

    @field_validator('patient_id')
    @classmethod
    def sanitize_patient_id(cls, v: str) -> str:
        if not v:
            return "anonymous"
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', str(v))[:100]
        return cleaned if cleaned else "anonymous"

class BookAppointmentRequest(BaseModel):
    patient_id: str = Field(default="anonymous", max_length=100)
    patient_name: str = Field(..., min_length=1, max_length=100)
    doctor_id: int
    day: WeekDay
    time: str = Field(..., pattern=r'^\d{2}:\d{2}$')

    @field_validator('patient_name')
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        return bleach.clean(v.strip())

    @model_validator(mode='before')
    @classmethod
    def map_date_to_day(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if 'date' in data and 'day' not in data:
                data['day'] = data['date']
        return data

class QueueCheckParams(BaseModel):
    doctor_id: int
    day: WeekDay
    time: str = Field(..., pattern=r'^\d{2}:\d{2}$')

    @model_validator(mode='before')
    @classmethod
    def map_date_to_day(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if 'date' in data and 'day' not in data:
                data['day'] = data['date']
        return data

class UploadRequest(BaseModel):
    patient_id: str = Field(default="anonymous", max_length=100)
    image_type: ImageType = Field(default=ImageType.DENTAL_XRAY)
    user_message: str = Field(default="", max_length=2000)
    language_preference: str = Field(default="ar", max_length=10)

    @field_validator('patient_id')
    @classmethod
    def sanitize_patient_id(cls, v: str) -> str:
        if not v:
            return "anonymous"
        cleaned = re.sub(r'[^a-zA-Z0-9_-]', '', str(v))[:100]
        return cleaned if cleaned else "anonymous"
