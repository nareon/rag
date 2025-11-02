#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: setup_rasa_nginx.sh WEBCHAT_ROOT [RASA_REST_URL]

WEBCHAT_ROOT   Absolute path to the directory containing the built webchat assets.
RASA_REST_URL  Optional URL for the Rasa REST webhook endpoint.
               Defaults to http://127.0.0.1:5005/webhooks/rest/

The script creates the nginx site configuration `/etc/nginx/sites-available/rasa_webchat`,
links it into `sites-enabled`, tests the configuration, and reloads nginx.
Run this script with root privileges.
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# > 2 ]]; then
  usage
  exit 1
fi

WEBCHAT_ROOT="$1"
RASA_REST_URL="${2:-http://127.0.0.1:5005/webhooks/rest/}"

if [[ ! -d "$WEBCHAT_ROOT" ]]; then
  echo "Error: WEBCHAT_ROOT '$WEBCHAT_ROOT' does not exist or is not a directory." >&2
  exit 1
fi

if [[ $(id -u) -ne 0 ]]; then
  echo "Error: this script must be run as root to modify nginx configuration." >&2
  exit 1
fi

CONFIG_PATH="/etc/nginx/sites-available/rasa_webchat"
ENABLED_PATH="/etc/nginx/sites-enabled/rasa_webchat"

cat <<'NOTICE'
Creating nginx configuration with the following settings:
NOTICE
printf '  Webchat root: %s\n' "$WEBCHAT_ROOT"
printf '  Rasa REST URL: %s\n' "$RASA_REST_URL"


server {
    listen 80;
    server_name _;

    root $WEBCHAT_ROOT;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    location /webhooks/rest/ {
        proxy_pass $RASA_REST_URL;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
CONFIG



ln -sf "$CONFIG_PATH" "$ENABLED_PATH"
echo "Symlinked $CONFIG_PATH to $ENABLED_PATH"

if [[ -f "/etc/nginx/sites-enabled/default" ]]; then
  echo "Removing default nginx site symlink to avoid conflicts."
  rm -f /etc/nginx/sites-enabled/default
fi

nginx -t
systemctl reload nginx

echo "nginx configuration reloaded."

echo "Setup complete. Ensure that the Rasa server is running with:"
echo "  rasa run --enable-api --cors \"*\""

echo "Then open http://localhost to verify the webchat."
