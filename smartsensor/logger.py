import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Add a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Attach handler to logger
logger.addHandler(console_handler)
