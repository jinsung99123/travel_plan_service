#!/bin/sh
# Generates .streamlit/secrets.toml from environment variables at container startup,
# keeping secrets out of the Docker image layer.
set -e

mkdir -p .streamlit

cat > .streamlit/secrets.toml << EOF
OPENAI_API_KEY = "${OPENAI_API_KEY}"
AUTH_API_BASE  = "${AUTH_API_BASE:-http://auth-service:8000}"
EOF

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
