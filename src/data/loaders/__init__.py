"""Source-specific dataset loaders.

Each loader exposes:
    load(splits: list[str] | None = None) -> Iterator[EmotionExample]

The unified output schema is `src.data.schema.EmotionExample`.
"""

from .base import LoaderError
from .go_emotions import load as load_go_emotions
from .semeval2018 import load as load_semeval2018
from .daily_dialog import load as load_daily_dialog
from .isear import load as load_isear
from .emobank import load as load_emobank

LOADERS = {
    "go_emotions": load_go_emotions,
    "semeval2018": load_semeval2018,
    "daily_dialog": load_daily_dialog,
    "isear": load_isear,
    "emobank": load_emobank,
}

__all__ = ["LOADERS", "LoaderError"]
