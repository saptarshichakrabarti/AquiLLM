
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# Global SentenceTransformer model for embeddings
_embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")


def get_embedding(query: str, input_type: str = 'search_query'):
    """
    Return an embedding vector for the given text using a local
    SentenceTransformer model (BAAI/bge-large-en-v1.5).

    The input_type parameter is kept for backward compatibility with the
    previous Cohere-based implementation but is not used by this model.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    embedding = _embedding_model.encode(query)
    # pgvector expects a plain list of floats, not a NumPy array
    return embedding.tolist()
