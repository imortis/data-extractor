"""PP-OCRv6 text reader, used as an evidence source independent of the vision model.

The model loads on first use because importing Paddle takes several seconds and many
requests (text documents, follow-up questions) never need it.
"""

import base64
import io
import os
import threading

import numpy as np

# PP-OCRv6's tiers are tiny, small and medium; "small" is its mobile-deployment tier.
# Medium was about 3x slower with no accuracy gain on the sample scans, so it's opt-in.
OCR_MODEL_TIER = os.environ.get("OCR_MODEL_TIER", "small")
OCR_DETECTION_MODEL = f"PP-OCRv6_{OCR_MODEL_TIER}_det"
OCR_RECOGNITION_MODEL = f"PP-OCRv6_{OCR_MODEL_TIER}_rec"

_lock = threading.Lock()
_ocr = None


def _load_ocr():
    global _ocr
    if _ocr is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddleocr import PaddleOCR

        _ocr = PaddleOCR(
            text_detection_model_name=OCR_DETECTION_MODEL,
            text_recognition_model_name=OCR_RECOGNITION_MODEL,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,  # oneDNN crashes on these models in Paddle 3.3.1 on CPU
        )
    return _ocr


def read_image_b64(image_b64: str) -> list[dict]:
    """Returns [{text, score, box: [x1, y1, x2, y2]}] for one base64-encoded image."""
    from PIL import Image

    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    bgr = np.array(image)[:, :, ::-1]

    with _lock:  # the Paddle predictor isn't safe to share across threads
        result = _load_ocr().predict(bgr)[0]

    data = result.json
    data = data.get("res", data)
    return [
        {"text": text, "score": float(score), "box": [float(v) for v in box]}
        for text, score, box in zip(data["rec_texts"], data["rec_scores"], data["rec_boxes"])
    ]
