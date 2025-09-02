#!/bin/sh

# This script creates a runtime config.js based on environment variables
# It can be used as a Docker ENTRYPOINT script for dynamic configuration

# Default values
BASE_URL=${BASE_URL:-'http://20.64.145.29:8010'}
CUSTOM_MODELS_URL=${CUSTOM_MODELS_URL:-'http://localhost:8000'}
DEBUG=${DEBUG:-'false'}

# Generate config.js
cat > /usr/share/nginx/html/config.js <<EOF
window.RUNTIME_CONFIG = {
  API_CONFIG: {
    BASE_URL: "${BASE_URL}",
    CUSTOM_MODELS_URL: "${CUSTOM_MODELS_URL}",
    DEBUG: ${DEBUG}
  }
};
EOF

echo "Generated runtime config.js with:"
echo "BASE_URL: ${BASE_URL}"
echo "CUSTOM_MODELS_URL: ${CUSTOM_MODELS_URL}"
echo "DEBUG: ${DEBUG}"

# Execute the CMD (nginx)
exec "$@"
