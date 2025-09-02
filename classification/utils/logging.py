"""
A simple CSV logger for tracking machine learning experiment metrics.
"""
import csv
import os
from pathlib import Path
from typing import List, Dict, Union

class CsvLogger:
    """
    A simple logger that saves experiment metrics to a CSV file.

    This class is designed to be used within a training loop. It handles
    creating the log file, writing the header, and appending new rows of data
    for each epoch or step.
    """
    def __init__(self, log_file_path: Union[str, Path], header: List[str]):
        """
        Initializes the CsvLogger.

        If the log file does not exist, it will be created and the header
        will be written. If it exists, the logger will append to it.

        Args:
            log_file_path (Union[str, Path]): The full path to the log file.
            header (List[str]): A list of strings representing the column names.
        """
        self.log_file_path = Path(log_file_path)
        self.header = header
        
        # Ensure the directory for the log file exists
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Open the file in append mode
        self.file = open(self.log_file_path, 'a', newline='')
        self.writer = csv.writer(self.file)

        # Check if the file is empty and write the header if so
        if os.path.getsize(self.log_file_path) == 0:
            self.writer.writerow(self.header)
            self.file.flush() # Ensure the header is written immediately

    def log(self, data: Dict[str, Union[int, float, str]]):
        """
        Logs a new row of data to the CSV file.

        Args:
            data (Dict): A dictionary where keys correspond to the header columns
                         and values are the metrics to be logged.
        """
        # Create a row of data in the same order as the header
        row = [data.get(col, '') for col in self.header]
        self.writer.writerow(row)
        self.file.flush() # Ensure data is written to disk immediately

    def close(self):
        """
        Closes the log file. Should be called at the end of training.
        """
        self.file.close()


# --- Example of how to use the CsvLogger in a training loop ---
if __name__ == '__main__':
    print("Running CsvLogger example...")

    # 1. Define the experiment parameters
    experiment_dir = Path("experiments/sample_run")
    log_file = experiment_dir / "training_log.csv"
    num_epochs = 5
    header = [
        'epoch', 'train_loss', 'train_accuracy',
        'val_loss', 'val_accuracy', 'learning_rate'
    ]

    # 2. Initialize the logger
    logger = CsvLogger(log_file_path=log_file, header=header)
    print(f"Logger initialized. Log file at: {log_file}")

    # 3. Simulate a training loop
    for epoch in range(num_epochs):
        # Simulate calculating metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': 1.0 - (epoch * 0.1),
            'train_accuracy': 85.0 + (epoch * 2.1),
            'val_loss': 0.8 - (epoch * 0.05),
            'val_accuracy': 88.0 + (epoch * 1.5),
            'learning_rate': 0.001 * (0.9 ** epoch)
        }

        # Log the metrics for the current epoch
        logger.log(metrics)
        print(f"Logged metrics for epoch {epoch + 1}")

    # 4. Close the logger at the end of the experiment
    logger.close()
    print("Logger closed.")

    # 5. Verify the output
    print("\n--- Contents of training_log.csv ---")
    with open(log_file, 'r') as f:
        print(f.read())
