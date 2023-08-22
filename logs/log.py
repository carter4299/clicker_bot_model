
from loguru import logger

def setup_logger():
    print("setting up logger")

    logger.remove()

    log_file = 'bot.log'
    logger.add(log_file, rotation="10 MB", retention="10 days", enqueue=True)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")