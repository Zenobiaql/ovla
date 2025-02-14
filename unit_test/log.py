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

val_dataset_name = "val"
mul_avg_loss = 0.1111111
mul_avg_accuracy = 0.2222222
mul_avg_action_l1_loss = 0.33333333
val_logger = get_logger("val_logger", "val_log.txt")

print(f"On dataset {val_dataset_name}, Loss:{mul_avg_loss:.3f}, Accuracy:{mul_avg_accuracy:.3f}, L1 Loss:{mul_avg_action_l1_loss:.3f}.")
val_logger.info(f"On dataset {val_dataset_name}, Loss:{mul_avg_loss:.3f}, Accuracy:{mul_avg_accuracy:.3f}, L1 Loss:{mul_avg_action_l1_loss:.3f}.")