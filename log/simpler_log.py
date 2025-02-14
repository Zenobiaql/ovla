import logging

def get_logger(name, file_path):
    if file_path is not None:
        f_handler = logging.FileHandler(file_path)
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if file_path is not None: 
        logger.addHandler(f_handler)
    
    return logger

if __name__ == "__main__":
    logger1 = get_logger("log1", "log1.txt")
    logger1.info("This is a log message.")
    
    logger2 = get_logger("log2", "log2.txt")
    logger2.info("This is another log message.")