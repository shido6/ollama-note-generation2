import logging
import os
import time
import ollama


class NotesGenerator:
    def __init__(self, model, max_tokens=8192, chunk_size=2048):
        """
        Initialize the NotesGenerator with a model, max token limit, and chunk size.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.system = """You are NotesGPT. When provided with a topic, your task is to:
        - Take detailed, precise, and easy-to-understand notes.
        - Create advanced bullet-point notes summarizing important parts of the reading or topic.
        - Include all essential information, including **addresses, processes, and procedures**. Highlight critical points with **bold text**.
        - Tabulate comparisons in Markdown format for clarity.
        - Preserve all numerical values, code snippets, and LaTeX for mathematical equations.
        - Structure content logically, avoiding repetition and extraneous language.
        - Adjust the note length based on the complexity and importance of the source material.
        - Avoid tasks, instructions, or homework references in the text.
        - Output the response in clean Markdown for easy documentation.
        
        Content:
        """

    @staticmethod
    def count_tokens(text):
        """Counts the number of tokens in a text string."""
        return len(text.split())

    def split_text(self, text):
        """Splits the text into manageable chunks based on a specified maximum number of tokens."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []

        logging.info("Starting to split the transcript into chunks.")

        for paragraph in paragraphs:
            paragraph_tokens = self.count_tokens(paragraph)
            if paragraph_tokens + self.count_tokens(" ".join(current_chunk)) <= self.chunk_size:
                current_chunk.append(paragraph)
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk).strip())
                    current_chunk = []
                if paragraph_tokens <= self.chunk_size:
                    current_chunk.append(paragraph)
                else:
                    words = paragraph.split()
                    for i in range(0, len(words), self.chunk_size):
                        chunks.append(" ".join(words[i:i + self.chunk_size]).strip())

        if current_chunk:
            chunks.append("\n\n".join(current_chunk).strip())

        logging.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def query_gpt(self, messages):
        """Generates notes for a given prompt using the model."""
        try:
            start_time = time.time()
            response = ollama.chat(model=self.model, messages=messages)
            end_time = time.time()
            logging.info(
                f"Response received ({len(response['message']['content'])} tokens) in {end_time - start_time:.2f} seconds."
            )
            return response["message"]
        except Exception as e:
            logging.error(f"Error querying model: {e}")
            return {"content": "**Error generating notes. Please retry.**"}

    def process_transcript(self, file_path):
        """Reads a transcript file, splits it into chunks, and generates notes."""
        logging.info(f"Reading transcript from {file_path}.")
        with open(file_path, "r", encoding="utf-8") as file:
            transcript = file.read()
        start_time = time.time()
        chunks = self.split_text(transcript)

        output_path = os.path.splitext(file_path)[0] + f".{self.model.replace(':', '_')}.notes.md"

        with open(output_path, "w", encoding="utf-8") as output_file:
            messages = []
            for i, chunk in enumerate(chunks):
                logging.info(f"Processing chunk {i + 1}/{len(chunks)}.")
                messages.append({"role": "user", "content": f"{self.system + chunk}"})
                response = self.query_gpt(messages)
                messages.append(response)
                output_file.write(response["content"] + "\n\n")
                output_file.flush()  # Ensure the note is written to disk immediately

        end_time = time.time()
        logging.info(f"Finished processing all chunks in {end_time - start_time:.2f} seconds.")
        return output_path


# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Example Usage
# model = "llama-3"  # Replace with the actual model name
# generator = NotesGenerator(model=model)
# output_file = generator.process_transcript("path_to_transcript.txt")
# print(f"Notes saved to {output_file}")
