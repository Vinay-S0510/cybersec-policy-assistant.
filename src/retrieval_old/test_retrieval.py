import os
from dotenv import load_dotenv

from .vector_store import test_retrieval


load_dotenv()

if __name__ == "__main__":
    query = "What security controls should be used to protect administrator accounts?"
    test_retrieval(query, n_results=5)
