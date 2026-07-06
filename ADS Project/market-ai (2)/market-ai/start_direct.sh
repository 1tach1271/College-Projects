#!/bin/bash

# Market AI System Startup
echo "Starting Market AI - GPU-Accelerated Financial Intelligence System..."

# Start API server
cd /home/arnav/market-ai
python -c "
from fastapi import FastAPI
import uvicorn
import time

app = FastAPI(title='Market AI API', version='1.0.0')

@app.get('/')
def root():
    return {'message': 'Market AI is running!', 'status': 'operational'}

@app.get('/health')
def health():
    return {'status': 'healthy', 'timestamp': time.time()}

@app.get('/docs')
def docs():
    return {'message': 'API Documentation available at /docs'}

print('Market AI API Starting...')
print('API: http://localhost:8000')
print('Docs: http://localhost:8000/docs')
uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
" &
BACKEND_PID=$!

# Wait for startup
sleep 3

# Check if running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "SUCCESS! Market AI API is running!"
    echo ""
    echo "Access points:"
    echo "   API: http://localhost:8000"
    echo "   Docs: http://localhost:8000/docs"
    echo "   Health: http://localhost:8000/health"
    echo ""
    echo "Press Ctrl+C to stop"
    
    trap "echo 'Stopping API...'; kill $BACKEND_PID; echo 'Stopped'; exit" INT
    wait
else
    echo "Failed to start API"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi
