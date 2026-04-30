from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch
import gc

# Explicit batch size to prevent embedding large document sets from OOM-ing
EMBED_BATCH_SIZE = 32


def setup_advanced_text_processing():
    device = _select_device()
    print(f"Embedding device: {device}")

    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=EMBED_BATCH_SIZE,
        device=device,
    )

    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    return embed_model


def create_node_parser():
    return SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
    )


def _select_device():
    """Pick CUDA if available, fall back to CPU. Frees GPU cache before loading."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        return "cuda"
    return "cpu"


def release_embed_model(embed_model):
    """Explicitly release embedding model after index build to reclaim RAM/VRAM."""
    del embed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
