"""
Premium Text-to-Speech Application - BARK AI Edition
Uses Suno's BARK - The most advanced free neural TTS with emotions
Features: Natural emotions, laughs, sighs, pauses, line-by-line playback, rewind
GPU Accelerated for RTX 3060 (6GB VRAM)
"""

import os
import re
import hashlib
import threading
import numpy as np
import requests
import json
from scipy.io.wavfile import write as write_wav
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS

# GPU Configuration - RTX 3060 6GB
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # Keep on GPU
os.environ["SUNO_USE_SMALL_MODELS"] = "False"  # Full quality models

import torch
# Configure CPU thread usage for heavy CPU fallback (can be overridden with BARK_CPU_THREADS env var)
CPU_THREADS = int(os.environ.get('BARK_CPU_THREADS', max(1, (os.cpu_count() or 2) - 1)))
try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, int(CPU_THREADS/2)))
except Exception:
    pass
print(f"🧮 CPU threads set to: {CPU_THREADS}")
print(f"🎮 PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🎮 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from bark import SAMPLE_RATE, generate_audio, preload_models

app = Flask(__name__)
CORS(app)

# Audio cache directory
AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'audio_cache')
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Track if models are loaded
models_loaded = False
loading_status = {"status": "not_started", "message": ""}

# Qwen2.5 configuration (assuming local Ollama or similar API)
QWEN_API_URL = os.environ.get('QWEN_API_URL', 'http://localhost:11434/api/generate')
QWEN_MODEL = os.environ.get('QWEN_MODEL', 'qwen2.5:latest')

def ai_preprocess_text(text):
    """
    Use Qwen2.5 to analyze and normalize text before TTS:
    - Adds proper punctuation and spacing
    - Fixes structure for better prosody
    - Understands context to improve readability
    """
    try:
        prompt = f"""You are a text preprocessing assistant for a text-to-speech system. Your job is to analyze the following text and improve it for natural speech generation.

Rules:
1. Add proper punctuation (periods, commas, question marks) where missing
2. Fix spacing issues (add spaces between words if missing)
3. Break very long sentences into shorter ones with commas or periods
4. Preserve code structure but add pauses (...) after code statements
5. Keep the original meaning and content intact
6. Do NOT add explanations, just return the improved text

Original text:
{text}

Improved text for speech:"""

        payload = {
            "model": QWEN_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 2048
            }
        }

        response = requests.post(QWEN_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            improved_text = result.get('response', '').strip()
            
            # If AI returned something reasonable, use it
            if improved_text and len(improved_text) > 10:
                return improved_text
            else:
                return text
        else:
            print(f"⚠️ Qwen API error: {response.status_code}")
            return text
            
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Qwen connection failed: {e}")
        return text
    except Exception as e:
        print(f"⚠️ AI preprocessing error: {e}")
        return text

# BARK Speaker presets - Neural voice embeddings with unique characteristics
VOICE_PRESETS = {
    # English Voices - Natural & Expressive
    "Emma (Female - Warm)": "v2/en_speaker_9",
    "James (Male - Deep)": "v2/en_speaker_6", 
    "Sophia (Female - Clear)": "v2/en_speaker_1",
    "Michael (Male - Friendly)": "v2/en_speaker_3",
    "Olivia (Female - Soft)": "v2/en_speaker_0",
    "William (Male - Professional)": "v2/en_speaker_7",
    "Ava (Female - Energetic)": "v2/en_speaker_2",
    "Benjamin (Male - Calm)": "v2/en_speaker_4",
    "Isabella (Female - Expressive)": "v2/en_speaker_5",
    "Ethan (Male - Narrator)": "v2/en_speaker_8",
    # Multi-lingual voices  
    "Hans (German)": "v2/de_speaker_3",
    "Marie (French)": "v2/fr_speaker_1",
    "Carlos (Spanish)": "v2/es_speaker_6",
    "Yuki (Japanese)": "v2/ja_speaker_1",
    "Wei (Chinese)": "v2/zh_speaker_4",
}

# Special audio prompts for emotions/effects
EMOTION_TAGS = """
BARK SPECIAL TAGS (add to your text for effects):
[laughter] - Natural laughter
[laughs] - Short laugh  
[sighs] - Sighing sound
[gasps] - Gasp sound
[clears throat] - Throat clearing
... - Natural pause/hesitation
♪ text ♪ - Singing
CAPITALS - Emphasis/louder
"""

def preprocess_text_for_speech(text, read_symbols=True):
    """
    Preprocess text for BARK. Converts code symbols to spoken words.
    Preserves BARK emotion tags like [laughter], [sighs], etc.
    """
    if not read_symbols:
        return text
    
    # Preserve BARK special tags
    bark_tags = ['[laughter]', '[laughs]', '[sighs]', '[gasps]', '[clears throat]', '♪']
    preserved = {}
    for i, tag in enumerate(bark_tags):
        placeholder = f"__BARK_{i}__"
        text = text.replace(tag, placeholder)
        preserved[placeholder] = tag
    
    # Symbol pronunciation dictionary
    symbol_map = {
        '{': ' open curly brace ',
        '}': ' close curly brace ',
        '[': ' open bracket ',
        ']': ' close bracket ',
        '(': ' open parenthesis ',
        ')': ' close parenthesis ',
        '<': ' less than ',
        '>': ' greater than ',
        '<=': ' less than or equal to ',
        '>=': ' greater than or equal to ',
        '==': ' equals equals ',
        '===': ' triple equals ',
        '!=': ' not equal to ',
        '!==': ' not strictly equal to ',
        '=>': ' arrow ',
        '->': ' arrow ',
        '::': ' double colon ',
        '&&': ' and and ',
        '||': ' or or ',
        '++': ' plus plus ',
        '--': ' minus minus ',
        '+=': ' plus equals ',
        '-=': ' minus equals ',
        '*=': ' times equals ',
        '/=': ' divide equals ',
        '**': ' power ',
        '//': ' double slash ',
        '/*': ' slash star ',
        '*/': ' star slash ',
        '...': ' spread operator ',
        '?.': ' optional chaining ',
        '??': ' nullish coalescing ',
        '@': ' at symbol ',
        '#': ' hash ',
        '$': ' dollar sign ',
        '%': ' percent ',
        '^': ' caret ',
        '&': ' ampersand ',
        '*': ' asterisk ',
        '~': ' tilde ',
        '`': ' backtick ',
        '|': ' pipe ',
        '\\': ' backslash ',
        '/': ' slash ',
        ';': ' semicolon ',
        ':': ' colon ',
        '"': ' quote ',
        "'": ' single quote ',
        '_': ' underscore ',
        '=': ' equals ',
        '+': ' plus ',
        '-': ' minus ',
        '!': ' exclamation ',
        '?': ' question mark ',
    }
    
    # Sort by length (longer patterns first) to avoid partial replacements
    sorted_symbols = sorted(symbol_map.keys(), key=len, reverse=True)
    
    processed = text
    
    # Handle multi-character symbols first
    for symbol in sorted_symbols:
        if len(symbol) > 1:
            processed = processed.replace(symbol, symbol_map[symbol])
    
    # Handle single character symbols (but be smart about context)
    for symbol in sorted_symbols:
        if len(symbol) == 1:
            # Don't replace symbols that are part of words or numbers
            pattern = re.escape(symbol)
            processed = re.sub(
                rf'(?<![a-zA-Z0-9]){pattern}(?![a-zA-Z0-9])',
                symbol_map[symbol],
                processed
            )
    
    # Clean up multiple spaces
    processed = re.sub(r'\s+', ' ', processed)
    
    # Handle camelCase and PascalCase - add spaces between words
    processed = re.sub(r'([a-z])([A-Z])', r'\1 \2', processed)
    
    # Handle snake_case - replace underscores with spaces for readability
    processed = re.sub(r'_([a-zA-Z])', r' \1', processed)

    # Replace dot between alphanumeric chars (e.g., db.students) with ' dot ' for code clarity
    processed = re.sub(r'(?<=\w)\.(?=\w)', ' dot ', processed)

    # Restore BARK tags
    for placeholder, tag in preserved.items():
        processed = processed.replace(placeholder, tag)

    return processed.strip()

def split_into_lines(text):
    """
    Split text into chunks optimal for BARK (~100-150 chars for best quality).
    Also merges very short lines to avoid prosody issues and appends a period to short fragments
    so the TTS doesn't invent filler sounds.
    """
    MIN_LEN = 45  # Merge lines shorter than this with the next
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    result = []
    
    for line in lines:
        # BARK works best with shorter segments
        if len(line) > 180:
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(sentence) > 180:
                    parts = sentence.split(', ')
                    current = ""
                    for part in parts:
                        if len(current) + len(part) < 150:
                            current += (", " if current else "") + part
                        else:
                            if current:
                                result.append(current)
                            current = part
                    if current:
                        result.append(current)
                elif sentence:
                    result.append(sentence)
        else:
            result.append(line)
    
    # Merge very short lines with the following line to provide context
    merged = []
    i = 0
    while i < len(result):
        cur = result[i]
        if len(cur) < MIN_LEN and i + 1 < len(result):
            # Merge with next line
            merged_line = f"{cur} {result[i+1]}"
            merged.append(merged_line)
            i += 2
        else:
            merged.append(cur)
            i += 1

    # Ensure short fragments get terminal punctuation to reduce filler noises
    final = []
    for line in merged:
        line = line.strip()
        # If the line looks like code (contains braces, arrows, dot notation), add a small pause '...'
        if re.search(r'[{}()=<>]|\w+\.\w+|=>|->', line):
            if not re.search(r'[\.\!\?]$', line):
                line = line + ' ...'
        elif len(line) < 60 and not re.search(r'[\.\!\?]$', line):
            line = line + '.'
        final.append(line)

    return final

def get_cache_filename(text, voice):
    """Generate unique cache filename."""
    content = f"{text}_{voice}"
    hash_val = hashlib.md5(content.encode()).hexdigest()
    return os.path.join(AUDIO_CACHE_DIR, f"{hash_val}.wav")

def load_bark_models():
    """Load BARK models to GPU."""
    global models_loaded, loading_status
    
    loading_status = {"status": "loading", "message": "Loading AI models to GPU..."}
    try:
        print("\n🔄 Loading BARK neural models to RTX 3060...")
        print("   First run downloads ~5GB of models.")
        
        preload_models()
        
        models_loaded = True
        loading_status = {"status": "ready", "message": "Models loaded on GPU!"}
        print("\n✅ BARK models loaded to GPU successfully!")
        if torch.cuda.is_available():
            print(f"   VRAM used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
    except Exception as e:
        loading_status = {"status": "error", "message": str(e)}
        print(f"\n❌ Error loading models: {e}")

def generate_bark_audio(text, voice_preset, output_file, force_cpu=False):
    """Generate audio using BARK. Supports optional CPU-only generation by setting
    environment flags and increasing CPU thread usage temporarily."""
    if not models_loaded:
        raise Exception("Models not loaded yet")

    # Save current env settings to restore later
    prev_offload = os.environ.get('SUNO_OFFLOAD_CPU')
    prev_small = os.environ.get('SUNO_USE_SMALL_MODELS')

    try:
        if force_cpu:
            # Force CPU-heavy mode
            os.environ['SUNO_OFFLOAD_CPU'] = 'True'
            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
            try:
                torch.set_num_threads(CPU_THREADS)
                torch.set_num_interop_threads(max(1, int(CPU_THREADS/2)))
            except Exception:
                pass

        # Generate audio (BARK handles device placement)
        audio_array = generate_audio(text, history_prompt=voice_preset)

        # Normalize and convert to int16
        audio_array = np.clip(audio_array, -1, 1)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Save as WAV
        write_wav(output_file, SAMPLE_RATE, audio_int16)

        return output_file

    finally:
        # Restore env flags
        if prev_offload is None:
            os.environ.pop('SUNO_OFFLOAD_CPU', None)
        else:
            os.environ['SUNO_OFFLOAD_CPU'] = prev_offload
        if prev_small is None:
            os.environ.pop('SUNO_USE_SMALL_MODELS', None)
        else:
            os.environ['SUNO_USE_SMALL_MODELS'] = prev_small
        try:
            # Reset threads to configured default
            torch.set_num_threads(CPU_THREADS)
            torch.set_num_interop_threads(max(1, int(CPU_THREADS/2)))
        except Exception:
            pass

@app.route('/', methods=['GET'])
def index():
    """Serve the main page - only accept GET requests."""
    return render_template('index.html')

@app.route('/', methods=['POST', 'PUT', 'DELETE', 'PATCH'])
def reject_root_post():
    """Reject POST and other methods to root - prevents accidental spam."""
    return jsonify({'error': 'Method not allowed on root path'}), 405

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get model loading status."""
    return jsonify({
        "models_loaded": models_loaded,
        "loading_status": loading_status,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    })

@app.route('/api/voices', methods=['GET'])
def get_voices():
    """Return available voice presets."""
    return jsonify(VOICE_PRESETS)

@app.route('/api/emotion-tags', methods=['GET'])
def get_emotion_tags():
    """Return available emotion tags."""
    return jsonify({"info": EMOTION_TAGS})

@app.route('/api/process-text', methods=['POST'])
def process_text():
    """Process text and return lines for TTS."""
    data = request.json
    text = data.get('text', '')
    read_code_symbols = data.get('readCodeSymbols', True)
    use_ai_preprocessing = data.get('useAiPreprocessing', False)
    
    # Step 1: AI preprocessing if enabled
    if use_ai_preprocessing:
        text = ai_preprocess_text(text)
    
    # Step 2: Split into lines
    lines = split_into_lines(text)
    
    # Step 3: Apply symbol reading if enabled
    if read_code_symbols:
        processed_lines = [preprocess_text_for_speech(line, True) for line in lines]
    else:
        processed_lines = lines
    
    return jsonify({
        'original_lines': lines,
        'processed_lines': processed_lines,
        'total_lines': len(lines),
        'ai_enhanced': use_ai_preprocessing
    })

@app.route('/api/generate-line-audio', methods=['POST'])
def generate_line_audio():
    """Generate audio for a single line using BARK."""
    if not models_loaded:
        return jsonify({'error': 'Models still loading. Please wait...'}), 503

    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'v2/en_speaker_6')
    hybrid = data.get('hybrid', True)
    force_cpu = data.get('force_cpu', False)

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    cache_file = get_cache_filename(text, voice)

    if not os.path.exists(cache_file):
        try:
            # If force_cpu is requested, attempt CPU-only generation
            if force_cpu:
                generate_bark_audio(text, voice, cache_file, force_cpu=True)
            else:
                # If hybrid mode is requested, attempt to detect low GPU memory and switch to small models
                reverted_small_models = False
                if hybrid and torch.cuda.is_available():
                    try:
                        total = torch.cuda.get_device_properties(0).total_memory
                        used = torch.cuda.memory_allocated(0)
                        free = total - used
                        # If less than ~1 GB free, request smaller models to avoid OOM
                        if free < 1 * 1024**3:
                            os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
                            reverted_small_models = True
                    except Exception:
                        pass

                generate_bark_audio(text, voice, cache_file)

                # Revert small models env var if changed
                if reverted_small_models:
                    os.environ['SUNO_USE_SMALL_MODELS'] = 'False'

        except RuntimeError as e:
            # GPU OOM or other runtime errors - try fallback with small models
            try:
                os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
                generate_bark_audio(text, voice, cache_file)
                os.environ['SUNO_USE_SMALL_MODELS'] = 'False'
            except Exception as e2:
                return jsonify({'error': str(e2)}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return send_file(cache_file, mimetype='audio/wav')

@app.route('/api/generate-full-audio', methods=['POST'])
def generate_full_audio():
    """Generate audio for full text."""
    if not models_loaded:
        return jsonify({'error': 'Models still loading'}), 503

    data = request.json
    text = data.get('text', '')
    voice = data.get('voice', 'v2/en_speaker_6')
    read_code_symbols = data.get('readCodeSymbols', True)
    hybrid = data.get('hybrid', True)

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    if read_code_symbols:
        processed_text = preprocess_text_for_speech(text, True)
    else:
        processed_text = text

    cache_file = get_cache_filename(processed_text, voice)

    if not os.path.exists(cache_file):
        try:
            reverted_small_models = False
            if hybrid and torch.cuda.is_available():
                try:
                    total = torch.cuda.get_device_properties(0).total_memory
                    used = torch.cuda.memory_allocated(0)
                    free = total - used
                    if free < 1 * 1024**3:
                        os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
                        reverted_small_models = True
                except Exception:
                    pass

            generate_bark_audio(processed_text, voice, cache_file)

            if reverted_small_models:
                os.environ['SUNO_USE_SMALL_MODELS'] = 'False'

        except RuntimeError as e:
            # Try fallback with small models
            try:
                os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
                generate_bark_audio(processed_text, voice, cache_file)
                os.environ['SUNO_USE_SMALL_MODELS'] = 'False'
            except Exception as e2:
                return jsonify({'error': str(e2)}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return send_file(cache_file, mimetype='audio/wav', as_attachment=True,
                     download_name='bark_tts_output.wav')

@app.route('/api/clear-cache', methods=['POST'])
def clear_cache():
    """Clear audio cache."""
    try:
        count = 0
        for file in os.listdir(AUDIO_CACHE_DIR):
            os.remove(os.path.join(AUDIO_CACHE_DIR, file))
            count += 1
        return jsonify({'success': True, 'message': f'Cleared {count} cached files'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*65)
    print("🎙️  BARK AI TEXT-TO-SPEECH - GPU ACCELERATED")
    print("="*65)
    print("Features:")
    print("  ✓ Neural AI voices with real emotions")
    print("  ✓ Supports [laughter], [sighs], ♪singing♪")
    print("  ✓ RTX 3060 GPU acceleration")
    print("  ✓ Line-by-line playback with rewind")
    print("  ✓ Unlimited local use!")
    print("="*65)
    
    # Start model loading in background
    print("\n⏳ Loading models to GPU in background...")
    loader_thread = threading.Thread(target=load_bark_models, daemon=True)
    loader_thread.start()
    
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("="*65 + "\n")
    
    app.run(debug=False, port=5000, threaded=True)
