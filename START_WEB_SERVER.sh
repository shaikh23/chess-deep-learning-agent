#!/bin/bash
# Quick start script for Chess AI Web App

echo "=========================================="
echo "Chess AI Web App - Local Test Server"
echo "=========================================="
echo ""

cd web

echo "Starting server on http://localhost:8000"
echo ""
echo "Open your browser and visit:"
echo "  → http://localhost:8000"
echo ""
echo "To stop the server: Press Ctrl+C"
echo ""
echo "=========================================="
echo ""

python3 -m http.server 8000
