"""Memory management utilities"""

import os
from typing import List

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from sentence_transformers import SentenceTransformer

_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_emb_dim = None


def get_embedding_model() -> SentenceTransformer:
    global _model, _emb_dim
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(_EMB_MODEL_NAME, device=device)
        _emb_dim = _model.get_sentence_embedding_dimension()
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_embedding_model()
    if not texts:

        return np.zeros((0, _emb_dim), dtype=np.float32)

    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

def _select_novel_lessons(
    cur_texts,
    new_texts,
    sim_threshold: float = 0.8,
    max_new_per_session: int = 3,
    max_capacity: int = 150,
):


    available_slots = max(0, max_capacity - len(cur_texts))
    if available_slots == 0:
        return []


    candidate_texts = [t for t in new_texts if t and t not in cur_texts]

    if not candidate_texts:
        return []


    if not cur_texts:
        session_limit = min(max_new_per_session, available_slots)
        return candidate_texts[:session_limit]


    cur_emb = embed_texts(cur_texts)          # shape: [N_cur, D]
    new_emb = embed_texts(candidate_texts)    # shape: [N_new, D]


    sims = cosine_sim_matrix(new_emb, cur_emb)

    scored = []
    for i, text in enumerate(candidate_texts):

        max_sim = float(sims[i].max())
        scored.append((max_sim, text))


    scored.sort(key=lambda x: x[0])

    novel = []

    session_limit = min(max_new_per_session, available_slots)

    for max_sim, text in scored:
        if max_sim < sim_threshold and len(novel) < session_limit:
            novel.append(text)
    return novel

def get_strategy_memory(target_model_name: str, T: int = 12):
    """
    Returns a limited chat history containing the summary and the first T lessons.
    """
    path = f"strategy_memory/{target_model_name}/memory.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    

    file_history = FileChatMessageHistory(path)
    

    limited_history = InMemoryChatMessageHistory()

    for msg in file_history.messages:
        content = msg.content
        

        if content.startswith("Positive Lessons:") or content.startswith("Negative Lessons:"):
            lines = content.split('\n')
            msg.content = "\n".join(lines[:T+1])
        

        limited_history.add_message(msg)
    return limited_history



def remove_messages_with_prefix(chat_memory, prefix: str):
    remaining = [msg for msg in chat_memory.messages if not msg.content.startswith(prefix)]
    chat_memory.clear()
    for msg in remaining:
        chat_memory.add_message(msg)
    
def _read_bucket(chat_memory, prefix):
    for m in chat_memory.messages:
        if m.content.startswith(prefix):
            return [l.strip("- ").strip()
                    for l in m.content[len(prefix):].split("\n") if l.strip()]
    return []

def _write_bucket(chat_memory, prefix, items):
    remove_messages_with_prefix(chat_memory, prefix)
    if items:
        payload = prefix + "\n".join(f"- {x}" for x in items)
        chat_memory.add_message(AIMessage(content=payload))

def save_summary_to_memory(model_name: str, result: dict):

    chat_history = get_strategy_memory(model_name)
    SUMM_PFX     = "Updated Summary:\n"
    POS_PFX, NEG_PFX = "Positive Lessons:\n", "Negative Lessons:\n"
    MAX_LESSONS_PER_BUCKET=150

    # current memory -----------------------------------------------------------
    cur_sum  = _read_bucket(chat_history, SUMM_PFX)  # list[str] or []
    cur_pos  = _read_bucket(chat_history, POS_PFX)   # list[str]
    cur_neg  = _read_bucket(chat_history, NEG_PFX)   # list[str]

    # incoming -----------------------------------------------------------------
    new_sum  = result.get("summary", "").strip()
    new_pos_raw  = [x.strip() for x in result.get("positive_lessons", []) if x.strip()]
    new_neg_raw  = [x.strip() for x in result.get("negative_lessons", []) if x.strip()]

    # summary ------------------------------------------------------------------
    if new_sum and (not cur_sum or new_sum != cur_sum[0]):
        _write_bucket(chat_history, SUMM_PFX, [new_sum])

    # novelty-based selection for lessons --------------------------------------
    new_pos = _select_novel_lessons(
        cur_pos,
        new_pos_raw,
        sim_threshold=0.8,
        max_new_per_session=3,
        max_capacity=MAX_LESSONS_PER_BUCKET,
    )

    new_neg = _select_novel_lessons(
        cur_neg,
        new_neg_raw,
        sim_threshold=0.8,
        max_new_per_session=3,
        max_capacity=MAX_LESSONS_PER_BUCKET,
    )

    # buckets (dedupe, keep order, clip to capacity) ---------------------------
    def merge_clip(old: List[str], new: List[str]) -> List[str]:
        merged = old + [x for x in new if x not in old]
        return merged[-MAX_LESSONS_PER_BUCKET:]

    final_pos = merge_clip(cur_pos, new_pos)
    final_neg = merge_clip(cur_neg, new_neg)

    _write_bucket(chat_history, POS_PFX, final_pos)
    _write_bucket(chat_history, NEG_PFX, final_neg)
    print(
    f"[LTM] {model_name} -> POS: {len(cur_pos)} -> {len(final_pos)}, "
    f"NEG: {len(cur_neg)} -> {len(final_neg)}, "
    f"added_pos={len(new_pos)}, added_neg={len(new_neg)}"
    )
    return("")