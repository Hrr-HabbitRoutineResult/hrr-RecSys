from typing import List
from sentence_transformers import SentenceTransformer
from app import config

_model: SentenceTransformer | None = None

def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(
                config.MODEL_PATH,
                device=config.DEVICE,
                tokenizer_kwargs={"fix_mistral_regex": True},
            )
        except TypeError:
            _model = SentenceTransformer(config.MODEL_PATH, device=config.DEVICE)
    return _model

def _ensure_passage_prefix(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("passage:") else f"passage: {t}"

def embed_text(text: str) -> List[float]:
    """
    /embedding에서 사용:
    - 항상 passage 스타일로 임베딩 생성
    """
    model = load_model()
    ptext = _ensure_passage_prefix(text)
    emb = model.encode(ptext, normalize_embeddings=True)
    return emb.tolist()
