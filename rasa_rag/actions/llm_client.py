"""
Унифицированный клиент Chat Completions.
Работает с локальным сервером (совместимым с OpenAI API) или с OpenAI.

Все параметры берутся только из .env:
  OPENAI_BASE_URL=http://192.168.1.66:8000/v1
  OPENAI_MODEL=Qwen/Qwen2.5-3B-Instruct
  OPENAI_API_KEY=
  LLM_TIMEOUT=60
  LLM_RETRIES=2
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, time, requests
from dotenv import load_dotenv
from pathlib import Path

# ---------------- Загрузка окружения ----------------
# Подтягивает .env из корня проекта
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# ---------------- Класс клиента ----------------
class LLMClient:
    """Обёртка над /v1/chat/completions."""

    def __init__(self):
        # Все параметры только из окружения
        self.base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "")
        self.timeout = float(os.getenv("LLM_TIMEOUT", "60"))
        self.retries = int(os.getenv("LLM_RETRIES", "2"))

        if not self.base_url or not self.model:
            raise RuntimeError("Не заданы OPENAI_BASE_URL или OPENAI_MODEL в .env")

        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # Клиент статeless, поэтому переиспользовать один экземпляр в рамках
        # запроса безопасно.  Requests сам управляет HTTP-сессией.

    # -------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: float = 0.2,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Делает POST-запрос к /chat/completions и возвращает ответ модели."""

        # Метод принимает список сообщений «как есть», чтобы повторно
        # использовать всю инфраструктуру промптов из Rasa actions.

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if extra:
            payload.update(extra)

        err = None
        for _ in range(self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                err = e
                time.sleep(0.3)

        # Если ни одна попытка не удалась, оборачиваем исходную ошибку в
        # RuntimeError, чтобы вызывающая сторона могла показать человекочитаемое
        # сообщение.
        raise RuntimeError(f"LLM error: {err}")
