#!/bin/bash

# Market AI Dashboard Launcher
echo "Launching Market AI Dashboard..."

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "Backend API is running!"
    echo ""
    echo "Opening Dashboard..."
    
    # Open dashboard in browser
    if command -v xdg-open > /dev/null; then
        xdg-open /home/arnav/market-ai/dashboard.html
    elif command -v gnome-open > /dev/null; then
        gnome-open /home/arnav/market-ai/dashboard.html
    else
        echo "Dashboard file: /home/arnav/market-ai/dashboard.html"
        echo "Please open this file in your browser"
    fi
    
    echo ""
    echo "Dashboard is ready with:"
    echo "  Real-time API status monitoring"
    echo "  Interactive feature cards"
    echo "  Performance charts"
    echo "  Direct API access links"
    echo ""
    echo "API Access Points:"
    echo "  API: http://localhost:8000"
    echo "  Docs: http://localhost:8000/docs"
    echo "  Health: http://localhost:8000/health"
    
else
    echo "Backend API is not running"
    echo "Please start the API first with: ./start_direct.sh"
    exit 1
fi
