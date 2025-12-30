# 🎙️ BARK AI Text-to-Speech

Premium neural text-to-speech application powered by Suno's BARK AI. Features emotional AI voices, GPU acceleration, and advanced playback controls.

## ✨ Features

- **🎭 Emotional AI Voices** - Natural emotions, laughter, sighs, and singing
- **⚡ GPU Accelerated** - Optimized for NVIDIA RTX GPUs (tested on RTX 3060)
- **🎮 Advanced Playback Controls** - Line-by-line navigation with rewind
- **📝 Code Symbol Reading** - Properly reads brackets, operators, and special characters
- **🎨 10 Premium Voices** - Multiple voice presets with unique characteristics
- **💾 Smart Caching** - Saves generated audio to avoid re-processing
- **🌐 Web Interface** - Modern, intuitive dark-themed UI
- **🔓 Completely Free** - No API keys, no subscriptions, unlimited use

## 🎯 Special Features

### Emotion Tags
BARK supports special tags for realistic expressions:
- `[laughter]` - Natural laughter
- `[sighs]` - Sighing sound
- `[gasps]` - Gasp sound
- `[clears throat]` - Throat clearing
- `♪ text ♪` - Singing
- `CAPITALS` - Emphasis/louder speech
- `...` - Natural pause/hesitation

### Code Reading
Automatically converts programming symbols to spoken words:
- `{` → "open curly brace"
- `}` → "close curly brace"
- `==` → "equals equals"
- `=>` → "arrow"
- `&&` → "and and"
- And many more!

## 🚀 Quick Start

### Requirements
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended, RTX 3060 or better)
- 8GB+ VRAM recommended for best performance
- ~13GB disk space for AI models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/DefinitelyNotABot-del/tts.git
cd tts
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Open in browser**
Navigate to http://localhost:5000

### First Run
On first launch, the app will automatically download ~13GB of AI models:
- `text_2.pt` (5.35GB) - Semantic understanding
- `coarse_2.pt` (3.93GB) - Voice generation
- `fine_2.pt` (3.74GB) - Audio quality refinement

**Models are downloaded once and cached locally** - subsequent runs start instantly.

## 🎮 Usage

### Basic Workflow
1. **Enter Text** - Paste or type your text in the input area
2. **Select Voice** - Choose from 10 premium voice presets
3. **Load Text** - Click "Load Text" to process into lines
4. **Play** - Press Play or Spacebar to start
5. **Navigate** - Use arrow keys or buttons to control playback

### Keyboard Shortcuts
- `Space` - Play/Pause
- `←` - Previous line
- `→` - Next line
- `R` - Rewind current line (press twice to go to previous line)
- `S` - Stop playback

### Rewind Behavior
The rewind button has smart behavior:
- **First press**: Restarts current line from beginning
- **Second press** (while on same line): Goes to previous line

## 📁 Project Structure

```
tts/
├── app.py                 # Flask backend with BARK integration
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Web interface
├── static/
│   ├── app.js            # Frontend JavaScript
│   └── styles.css        # UI styling
└── audio_cache/          # Generated audio cache (auto-created)
```

## 🔧 Configuration

### GPU & CPU Settings
Edit `app.py` or set environment variables to modify configuration:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # GPU index
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # Keep models on GPU by default
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # Use full quality models
```

You can force the app to use more CPU when GPU is busy:
- Set `BARK_CPU_THREADS` environment variable (defaults to number of cores - 1)
- In the UI, enable **Aggressive CPU mode** to force CPU-only generation for current requests

Example (PowerShell):

```powershell
$env:BARK_CPU_THREADS = 12
python app.py
```

### Voice Presets
Available voices in `app.py`:
- Emma (Female - Warm)
- James (Male - Deep)
- Sophia (Female - Clear)
- Michael (Male - Friendly)
- And 6 more...

## 💡 Tips & Tricks

1. **Best Results**: Keep text segments under 180 characters for optimal quality
2. **GPU Performance**: RTX 3060 6GB generates ~3 seconds of audio per second
3. **Caching**: Previously generated audio is cached - identical text plays instantly
4. **Emotion Tags**: Mix emotion tags with regular text for realistic expressions
5. **Symbol Reading**: Enable "Read Code Symbols" for technical content

## 🐛 Troubleshooting

### Models Not Loading
- Ensure stable internet connection (13GB download)
- Check available disk space (~15GB free recommended)
- Models are cached in `~/.cache/huggingface/hub/`

### GPU Not Detected
- Install CUDA Toolkit 12.1 or compatible version
- Verify with: `torch.cuda.is_available()`
- CPU fallback is automatic but slower

### Slow Generation
- Check GPU memory usage - other apps may be using VRAM
- Try closing GPU-intensive applications
- Consider using smaller text segments

### Audio Quality Issues
- Avoid very long sentences (>200 chars)
- Use proper punctuation for natural pauses
- Try different voice presets

## 📦 Technology Stack

- **Backend**: Flask 3.1.2, Python 3.11
- **AI Model**: Suno BARK (Advanced Neural TTS)
- **GPU**: PyTorch 2.5.1 with CUDA 12.1
- **Audio**: scipy, numpy
- **Frontend**: Vanilla JavaScript, CSS3

## 🎓 Credits

- **BARK AI** by [Suno AI](https://github.com/suno-ai/bark) - The amazing neural TTS model
- **PyTorch** - Deep learning framework
- **Flask** - Web framework

## 📄 License

This project is for educational and personal use. BARK AI model is subject to its own license terms.

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 🌟 Star This Repo

If you find this useful, please consider starring the repository!

---

**Made with ❤️ for the AI community**
