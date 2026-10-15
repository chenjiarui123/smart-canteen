"""Pure normalization compatibility helper; no filesystem repository."""
import re
import unicodedata


def normalize_dish_name(value):
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = text.translate(str.maketrans({"（": "(", "）": ")", "【": "[", "】": "]"}))
    return re.sub(r"\s+", " ", text)
