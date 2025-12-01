import os
import requests
import json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
print("🔑 OPENROUTER_API_KEY =", OPENROUTER_API_KEY)  # Log pour vérifier la clé

# Intents en FR et EN
PROMPTS = {
    "fr": """
Tu es un moteur d'intentions très strict.
Analyse le texte utilisateur et retourne UNE intention parmi :

- inscription
- connexion
- ouvrir
- paiement
- information
- résiliation
- retour
- moins_35
- entre_35_50
- plus_50
- homme
- femme
- autre

Réponds uniquement par un mot.
""",
    "en": """
You are a very strict intent classifier.
Analyze the user text and return ONE of the following intents:

- signup
- enroll
- login
- payment
- info
- back
- under_35
- between_35_50
- over_50
- man
- woman
- previously
- cancel
- other

Respond with only one word.
"""
}

def detect_intent_openrouter(text: str, lang="fr") -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set in environment variables!")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "FlaskAudioAssistant",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": PROMPTS.get(lang, PROMPTS["fr"])},
            {"role": "user", "content": text}
        ]
    }

    print("🔹 OpenRouter headers:", headers)
    print("🔹 OpenRouter payload:", json.dumps(payload, indent=2))

    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()

    intent = data["choices"][0]["message"]["content"].strip().lower()
    print("🔹 OpenRouter detected intent:", intent)
    return intent
