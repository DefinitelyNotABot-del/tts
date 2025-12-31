# TTS Quick Start Script
# Launches Ollama and TTS app in separate terminals

Write-Host "🚀 Starting BARK AI TTS with Qwen2.5..." -ForegroundColor Cyan

# Check if Ollama is installed
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "✅ Ollama found" -ForegroundColor Green
    
    # Start Ollama in a new terminal
    Write-Host "🔄 Starting Ollama server..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "ollama serve"
    
    # Wait for Ollama to start
    Write-Host "⏳ Waiting 3 seconds for Ollama to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
} else {
    Write-Host "⚠️  Ollama not found - AI preprocessing will be disabled" -ForegroundColor Yellow
    Write-Host "   Install: winget install Ollama.Ollama" -ForegroundColor Gray
}

# Activate venv and start TTS app
Write-Host "🎙️  Starting TTS application..." -ForegroundColor Cyan

# Check if venv exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  No virtual environment found, using global Python" -ForegroundColor Yellow
}

# Start the app
Write-Host "`n🌐 Opening http://localhost:5000 in 5 seconds..." -ForegroundColor Cyan
Write-Host "📊 GPU scheduler enabled - will use shared memory for optimal performance`n" -ForegroundColor Green

# Start app and wait 5 seconds before opening browser
python app.py &
Start-Sleep -Seconds 5
Start-Process "http://localhost:5000"
