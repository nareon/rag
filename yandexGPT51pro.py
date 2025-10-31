import os
import requests
from dotenv import load_dotenv

# === Загрузка .env ===
load_dotenv()

BASE = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

YC_API_KEY = os.getenv("YC_API_KEY")
YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_MODEL_URI = os.getenv("YC_MODEL_URI")

if not all([YC_API_KEY, YC_FOLDER_ID, YC_MODEL_URI]):
    raise RuntimeError("Ошибка: не найдены обязательные переменные YC_API_KEY, YC_FOLDER_ID, YC_MODEL_URI")

headers = {
    "Authorization": f"Api-Key {YC_API_KEY}",
    "x-folder-id": YC_FOLDER_ID,
    "Content-Type": "application/json",
}

payload = {
    "modelUri": YC_MODEL_URI,
    "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": 600},
    "messages": [
        {"role": "system", "text": "Ты ассистент по Rasa. Отвечай кратко по-русски."},
        {"role": "user", "text": "Как подключить Telegram к Rasa?"},
    ],
}

r = requests.post(BASE, headers=headers, json=payload, timeout=60)
r.raise_for_status()

data = r.json()
if "result" in data and data["result"].get("alternatives"):
    print(data["result"]["alternatives"][0]["message"]["text"])
else:
    print("Ответ от модели пустой или некорректный:", data)

