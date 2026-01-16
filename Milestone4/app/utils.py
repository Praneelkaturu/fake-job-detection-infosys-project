# app/utils.py

def predict_job(text: str):
    """
    Dummy fake-job prediction function.
    Replace with real ML logic later.
    """
    if "money" in text.lower() or "earn" in text.lower():
        return {
            "label": "Fake",
            "confidence": 0.92
        }
    return {
        "label": "Real",
        "confidence": 0.88
    }
