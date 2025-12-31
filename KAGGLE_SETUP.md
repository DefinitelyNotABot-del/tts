# Kaggle Kernel Offload Setup Guide

## What is This?

This allows you to offload TTS generation to a free Kaggle GPU kernel when your local GPU is busy or slow. Kaggle offers:
- **30 hours/week** of free GPU compute
- NVIDIA Tesla P100 or T4 GPUs
- Faster than most laptop GPUs

## Setup Steps

### 1. Create a Kaggle Account
- Go to https://www.kaggle.com/
- Sign up (free)
- Verify your phone number (required for GPU access)

### 2. Get API Credentials
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json` (contains username and API key)

### 3. Create a Kaggle Notebook
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Name it: `bark-tts-api`
4. **Enable GPU**: Click Settings → Accelerator → GPU T4 x2
5. Add this code to the notebook:

```python
# Install dependencies
!pip install flask flask-cors scipy numpy torch torchaudio bark

# Create API server
from flask import Flask, request, jsonify, send_file
from bark import generate_audio, SAMPLE_RATE
import numpy as np
from scipy.io.wavfile import write as write_wav
import io

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'v2/en_speaker_6')
    
    # Generate audio
    audio_array = generate_audio(text, history_prompt=voice)
    audio_array = np.clip(audio_array, -1, 1)
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # Save to bytes
    buffer = io.BytesIO()
    write_wav(buffer, SAMPLE_RATE, audio_int16)
    buffer.seek(0)
    
    return send_file(buffer, mimetype='audio/wav')

if __name__ == '__main__':
    from flask_ngrok import run_with_ngrok
    run_with_ngrok(app)  # Start ngrok when app is run
    app.run()
```

6. Run the notebook
7. Copy the **ngrok URL** (e.g., `https://abc123.ngrok.io`)

### 4. Configure Local TTS App

Set environment variables before starting:

```powershell
# PowerShell
$env:KAGGLE_API_URL = "https://YOUR_NGROK_URL.ngrok.io/generate"
$env:KAGGLE_API_KEY = "your-kaggle-api-key"

# Then start app
python app.py
```

OR create a `.env` file:
```
KAGGLE_API_URL=https://YOUR_NGROK_URL.ngrok.io/generate
KAGGLE_API_KEY=your-kaggle-api-key
```

### 5. Usage

The app will automatically:
- Use local GPU when available
- Offload to Kaggle when local GPU is busy
- Fall back to CPU if both fail

## Monitoring

Watch the console for:
```
📊 Using Kaggle offload - Local GPU busy
✅ Kaggle generation complete in 2.3s
```

## Limitations

- 30 hours/week GPU quota
- Notebook must stay running
- Internet latency (~1-2s extra per request)
- ngrok URL changes when notebook restarts

## Alternative: Permanent Kaggle API

For production use, deploy a persistent Flask app on Kaggle instead of ngrok.

## Troubleshooting

**Connection failed**: Kaggle notebook might be stopped - restart it
**Quota exceeded**: You've used 30 hours this week - wait for reset
**Slow**: Network latency - check your internet connection
