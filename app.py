from concurrent.futures import ThreadPoolExecutor
import glob
import os
import logging
import shutil
import sys
import re
import json
from threading import Event
from tqdm import tqdm
from notes_generator import NotesGenerator

# Define constants
ERROR_DIR = "errors"
os.makedirs(ERROR_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Global event to stop processing
stop_processing = Event()


def load_config(config_path):
    """Load configuration from a JSON file."""
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def process_raw_to_notes(file_path, notes_generator):
    """Process a single file to generate notes."""
    if stop_processing.is_set():
        logging.info(f"Processing halted for {file_path}.")
        return

    try:
        notes_generator.process_transcript(file_path)
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        shutil.copy(file_path, os.path.join(ERROR_DIR, os.path.basename(file_path)))
        stop_processing.set()  # Signal other threads to stop if one fails


def run_raw_to_notes(model, max_threads, folder):
    """Process raw transcript files to notes."""
    notes_generator = NotesGenerator(model=model, max_tokens=4096)

    # Get all transcript files from input directory
    raw_files = glob.glob(os.path.join(folder, "**/*.txt"), recursive=True)
    filtered_raw_files = []

    # Iterate over the raw files and filter out files that already have notes
    for raw_file in raw_files:
        raw_file_name = os.path.splitext(os.path.basename(raw_file))[0]
        model_name = re.sub(r"[:/\\]", "_", model)  # sanitize model name for file naming
        notes_file = os.path.join(
            os.path.dirname(raw_file), f"{raw_file_name}.{model_name}.notes.md"
        )
        if not os.path.exists(notes_file):
            filtered_raw_files.append(raw_file)

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        with tqdm(total=len(filtered_raw_files)) as pbar:
            futures = [
                executor.submit(process_raw_to_notes, file, notes_generator)
                for file in filtered_raw_files
            ]
            for future in futures:
                try:
                    future.result()  # Wait for each future to complete
                except Exception as e:
                    logging.error(f"Error in transcript to notes thread: {e}")
                finally:
                    pbar.update(1)


def main(pipeline, model, max_threads, folder):
    """Main entry point to select pipeline and start processing."""
    if pipeline == "raw_to_notes":
        run_raw_to_notes(model, max_threads, folder)
    else:
        logging.error(f"Unknown pipeline: {pipeline}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <pipeline> <max_threads> <model> <folder>")
        print("pipeline: 'raw_to_notes'")
        sys.exit(1)

    pipeline = sys.argv[1]
    max_threads = int(sys.argv[2])
    model = sys.argv[3]
    folder = sys.argv[4]

    logging.info("Starting note generation process.")
    main(pipeline, model, max_threads, folder)
    logging.info("Completed note generation process.")
