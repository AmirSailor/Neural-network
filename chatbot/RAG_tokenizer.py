import tiktoken
import json
from config import Tokenizer_encoder_name

class RAGTokenizer:
    """
    A tokenizer class designed for RAG systems that works with specific
    JSON data structures and uses the tiktoken library for efficiency.

    This class helps in converting text from a JSON object into tokens and
    counting them, which is essential for managing document chunking and
    LLM context window limits in a RAG pipeline.
    """

    def __init__(self, encoding_name: str = Tokenizer_encoder_name):
        """
        Initializes the tokenizer with a specific encoding.

        Args:
            encoding_name (str): The name of the encoding to use.
                                 'cl100k_base' is the standard for models
                                 like gpt-4, gpt-3.5-turbo, and text-embedding-ada-002.
        """
        try:
            # Get the tokenizer encoding from tiktoken
            self.encoder = tiktoken.get_encoding(encoding_name)
            print(f"Tokenizer initialized with '{encoding_name}' encoding.")
        except Exception as e:
            print(f"Error initializing tokenizer: {e}")
            self.encoder = None

    def tokenize_entry(self, data_entry: dict, use_summary_if_present: bool = False) -> list[int] | None:
        """
        Tokenizes the text content of a given data entry dictionary.

        It can be configured to prioritize tokenizing the 'summary' field if it exists.

        Args:
            data_entry (dict): A dictionary containing the text data,
                               expected to have a 'text' key and an optional 'summary' key.
            use_summary_if_present (bool): If True and a 'summary' key exists,
                                           the summary will be tokenized.
                                           Otherwise, the 'text' field is used.

        Returns:
            list[int] | None: A list of token integers, or None if an error occurs
                              or no valid text field is found.
        """
        if not self.encoder:
            print("Error: Tokenizer is not initialized.")
            return None

        text_to_tokenize = ""
        # Decide which field to use based on the flag and availability
        if use_summary_if_present and "summary" in data_entry and data_entry["summary"]:
            text_to_tokenize = data_entry["summary"]
        elif "text" in data_entry and data_entry["text"]:
            text_to_tokenize = data_entry["text"]
        else:
            # Handle cases where expected keys are missing
            print("Warning: No valid 'text' or 'summary' field found in data_entry.")
            return []

        # Perform the tokenization
        try:
            tokens = self.encoder.encode(text_to_tokenize)
            return tokens
        except Exception as e:
            print(f"Error during tokenization: {e}")
            return None

    def count_tokens_in_entry(self, data_entry: dict, use_summary_if_present: bool = False) -> int:
        """
        Counts the number of tokens in the text content of a given data entry.

        This is a convenience method that is often more efficient than generating
        the full token list if you only need the count.

        Args:
            data_entry (dict): A dictionary containing the text data.
            use_summary_if_present (bool): If True, counts tokens in the 'summary',
                                           otherwise counts tokens in the 'text'.

        Returns:
            int: The total number of tokens. Returns 0 if no text is found.
        """
        tokens = self.tokenize_entry(data_entry, use_summary_if_present)
        return len(tokens) if tokens is not None else 0

# --- Example Usage ---
if __name__ == "__main__":
    # To run this example, you need to install tiktoken:
    # pip install tiktoken

    # 1. Initialize our custom tokenizer
    tokenizer = RAGTokenizer(encoding_name="cl100k_base")

    # 2. Sample data entries based on your structure
    page_url = "https://example.com/article"
    title = "The Future of AI"
    text = "The field of artificial intelligence is evolving at an incredible pace. " \
           "Large language models, or LLMs, are at the forefront of this revolution, " \
           "enabling new applications in various domains."
    
    # Mock summary function for the example
    def generate_summary(input_text: str) -> str:
        return "AI is changing fast, with LLMs leading the way for new apps."

    # --- Scenario 1: Data entry without a summary ---
    print("\n--- Scenario 1: Processing entry without summary ---")
    data_entry_no_summary = {
        "url": page_url,
        "title": title,
        "text": text,
    }
    
    # Count tokens in the main text
    token_count = tokenizer.count_tokens_in_entry(data_entry_no_summary)
    print(f"Token count for the main text: {token_count}")

    # Get the actual tokens
    tokens = tokenizer.tokenize_entry(data_entry_no_summary)
    print(f"First 10 tokens: {tokens[:10] if tokens else 'N/A'}")
    
    # --- Scenario 2: Data entry with a summary ---
    print("\n--- Scenario 2: Processing entry with summary ---")
    data_entry_with_summary = {
        "url": page_url,
        "title": title,
        "text": text,
        "summary": generate_summary(text)
    }

    # a) Count tokens in the main text
    token_count_full = tokenizer.count_tokens_in_entry(data_entry_with_summary, use_summary_if_present=False)
    print(f"Token count for the main text: {token_count_full}")
    
    # b) Count tokens in the summary
    token_count_summary = tokenizer.count_tokens_in_entry(data_entry_with_summary, use_summary_if_present=True)
    print(f"Token count for the summary: {token_count_summary}")

    # Get the tokens for the summary
    summary_tokens = tokenizer.tokenize_entry(data_entry_with_summary, use_summary_if_present=True)
    print(f"Summary tokens: {summary_tokens}")

    print("\n" + "="*50)
    print("This tokenizer can now be used in your Langchain pipeline, for example,")
    print("with a TextSplitter that uses token count for chunking.")
    print("="*50)

