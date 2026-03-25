import logging
import os
from datetime import datetime

# --- CONFIGURATION ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a unique log file for each session
LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# --- LOGGER SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_logger(name: str):
    """Returns a logger instance with the specified name."""
    return logging.getLogger(name)

if __name__ == "__main__":
    logger = get_logger("TestLogger")
    logger.info("Universal Logging System Initialized.")
