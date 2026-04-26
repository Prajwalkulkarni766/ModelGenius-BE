import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

SENSITIVE_FIELDS = {
    "password", "refresh_token", "access_token", "token", 
    "secret", "api_key", "authorization", "cookie"
}

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(LOG_DIR, "app.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
        ),
    ],
)

logger = logging.getLogger("ModelGenius-BE")
logger.setLevel(getattr(logging, LOG_LEVEL))


def mask_sensitive_fields(data):
    if data is None or not isinstance(data, (dict, list)):
        return data
    
    if isinstance(data, list):
        return [mask_sensitive_fields(item) for item in data]
    
    masked = {}
    for key, value in data.items():
        lower_key = key.lower() if isinstance(key, str) else key
        if any(field in lower_key for field in SENSITIVE_FIELDS):
            masked[key] = "[REDACTED]"
        elif isinstance(value, dict):
            masked[key] = mask_sensitive_fields(value)
        elif isinstance(value, list):
            masked[key] = mask_sensitive_fields(value)
        else:
            masked[key] = value
    return masked