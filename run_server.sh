#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set port from argument or default to 8000
PORT=${1:-8000}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$DIR/.venv" ]; then
    echo "Creating virtual environment..."
    uv venv "$DIR/.venv"
fi

# Activate virtual environment
source "$DIR/.venv/bin/activate"

# Install dependencies
if [ -f "$DIR/requirements.txt" ]; then
    echo "Installing dependencies..."
    uv pip install -r "$DIR/requirements.txt"
fi

# Check if port is in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Warning: Port $PORT is already in use."
fi

echo "Starting Uvicorn Server on port $PORT..."
echo "API Docs: http://localhost:$PORT/docs"

# Run Uvicorn
# execution from root directory so 'user_api.user_api' module path works
cd "$DIR"
uvicorn user_api.user_api:app --host 0.0.0.0 --port "$PORT" --reload
