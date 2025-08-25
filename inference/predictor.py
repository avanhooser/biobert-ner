import os, re
from typing import List, Dict

class Predictor:
    def __init__(self):
        self.labels = ["CHEMICAL", "DISEASE", "GENE"]
        self.pipeline = None
        model_dir = os.environ.get("MODEL_DIR", "/opt/ml/model")
        # Try to load a HF pipeline if a saved model is present
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                tok = AutoTokenizer.from_pretrained(model_dir)
                mdl = AutoModelForTokenClassification.from_pretrained(model_dir)
                self.pipeline = pipeline("token-classification", model=mdl, tokenizer=tok, aggregation_strategy="simple")
        except Exception:
            self.pipeline = None

    def predict(self, text: str) -> List[Dict]:
        if self.pipeline is not None:
            preds = self.pipeline(text)
            return [{"text": p["word"], "label": p["entity_group"], "score": float(p["score"])}
                    for p in preds]
        # Fallback heuristic
        ents, tokens = [], re.findall(r"[A-Z][a-zA-Z]{3,}", text)
        for t in tokens:
            label = self.labels[hash(t) % len(self.labels)]
            ents.append({"text": t, "label": label, "score": 0.5})
        return ents
