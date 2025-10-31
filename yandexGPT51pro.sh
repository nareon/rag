#!/bin/bash
set -a
source .env
set +a

curl -s https://llm.api.cloud.yandex.net/foundationModels/v1/completion   -H "Authorization: Api-Key $YC_API_KEY"   -H "x-folder-id: $YC_FOLDER_ID"   -H "Content-Type: application/json"   -d '{
        "modelUri": "'"$YC_MODEL_URI"'",
        "completionOptions": { "stream": false, "temperature": 0.2, "maxTokens": 600 },
        "messages": [
          { "role": "system", "text": "Отвечай кратко на русском." },
          { "role": "user",   "text": "Как подключить Telegram к Rasa?" }
        ]
      }' | jq -r '.result.alternatives[0].message.text'