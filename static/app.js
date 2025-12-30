/**
 * Premium Text-to-Speech Application - BARK AI
 * Full-featured audio player with line-by-line control
 * GPU Accelerated Neural TTS
 */

class TTSPlayer {
    constructor() {
        // State
        this.lines = [];
        this.processedLines = [];
        this.currentLineIndex = 0;
        this.isPlaying = false;
        this.isPaused = false;
        this.audioCache = new Map();
        this.modelsLoaded = false;
        
        // Settings
        this.selectedVoice = 'v2/en_speaker_6';
        this.readCodeSymbols = true;
        this.useFullGen = false; // single-file generation toggle
        this.hybridMode = true; // auto GPU/CPU fallback
        
        // Playback tracking
        this.currentLineStarted = false;
        this.currentChunkSize = 1; // how many lines are in the current audio chunk
        this.playGenerateLock = false; // prevent overlapping generation requests
        this.rewindPressedOnce = false;
        this.rewindTimer = null;
        
        // DOM Elements
        this.initElements();
        this.initEventListeners();
        this.loadVoices();
        this.checkModelStatus();
    }
    
    initElements() {
        // Text input
        this.textInput = document.getElementById('textInput');
        this.charCount = document.getElementById('charCount');
        this.lineCount = document.getElementById('lineCount');
        
        // Buttons
        this.clearBtn = document.getElementById('clearBtn');
        this.pasteBtn = document.getElementById('pasteBtn');
        this.loadTextBtn = document.getElementById('loadTextBtn');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.prevLineBtn = document.getElementById('prevLineBtn');
        this.nextLineBtn = document.getElementById('nextLineBtn');
        this.rewindBtn = document.getElementById('rewindBtn');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.clearCacheBtn = document.getElementById('clearCacheBtn');
        
        // Additional settings
        this.useFullGenCheckbox = document.getElementById('useFullGen');
        this.hybridModeCheckbox = document.getElementById('hybridMode');
        
        // Icons
        this.playPauseIcon = document.getElementById('playPauseIcon');
        
        // Settings
        this.voiceSelect = document.getElementById('voiceSelect');
        this.readCodeSymbolsCheckbox = document.getElementById('readCodeSymbols');
        
        // Player display
        this.currentLineNum = document.getElementById('currentLineNum');
        this.totalLines = document.getElementById('totalLines');
        this.currentLineText = document.getElementById('currentLineText');
        this.progressFill = document.getElementById('progressFill');
        this.progressBar = document.getElementById('progressBar');
        
        // Line list
        this.lineList = document.getElementById('lineList');
        
        // Status
        this.statusMessage = document.getElementById('statusMessage');
        
        // Loading
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.loadingText = document.getElementById('loadingText');
        
        // Audio
        this.audioPlayer = document.getElementById('audioPlayer');
    }
    
    initEventListeners() {
        // Text input events
        this.textInput.addEventListener('input', () => this.updateTextStats());
        this.textInput.addEventListener('paste', (e) => {
            // Allow the paste to complete, then update
            setTimeout(() => this.updateTextStats(), 0);
        });
        
        // Button events
        this.clearBtn.addEventListener('click', () => this.clearText());
        this.pasteBtn.addEventListener('click', () => this.pasteFromClipboard());
        this.loadTextBtn.addEventListener('click', () => this.loadText());
        this.playPauseBtn.addEventListener('click', () => this.togglePlayPause());
        this.stopBtn.addEventListener('click', () => this.stop());
        this.prevLineBtn.addEventListener('click', () => this.previousLine());
        this.nextLineBtn.addEventListener('click', () => this.nextLine());
        this.rewindBtn.addEventListener('click', () => this.rewindLine());
        this.downloadBtn.addEventListener('click', () => this.downloadAudio());
        this.clearCacheBtn.addEventListener('click', () => this.clearCache());
        
        // Settings toggles
        this.useFullGenCheckbox.addEventListener('change', (e) => { this.useFullGen = e.target.checked; });
        this.hybridModeCheckbox.addEventListener('change', (e) => { this.hybridMode = e.target.checked; });
        this.forceCpuCheckbox = document.getElementById('forceCpuMode');
        this.useCpuAggressive = false;
        this.forceCpuCheckbox?.addEventListener('change', (e) => { this.useCpuAggressive = e.target.checked; });
        
        // Audio events
        this.audioPlayer.addEventListener('timeupdate', () => this.updateProgress());
        this.audioPlayer.addEventListener('ended', () => this.onLineEnded());
        this.audioPlayer.addEventListener('play', () => {
            this.isPlaying = true;
            this.isPaused = false;
            this.playPauseIcon.className = 'fas fa-pause';
        });
        this.audioPlayer.addEventListener('pause', () => {
            this.isPaused = true;
            this.playPauseIcon.className = 'fas fa-play';
        });
        this.audioPlayer.addEventListener('error', (e) => {
            console.error('Audio element error', e);
            this.setStatus('Playback engine error', 'error');
        });
        
        // Settings events
        this.voiceSelect.addEventListener('change', (e) => {
            this.selectedVoice = e.target.value;
            this.setStatus(`Voice changed to ${e.target.options[e.target.selectedIndex].text}`);
        });
        
        this.readCodeSymbolsCheckbox.addEventListener('change', (e) => {
            this.readCodeSymbols = e.target.checked;
        });
        
        // Progress bar click
        this.progressBar.addEventListener('click', (e) => {
            if (!this.audioPlayer.duration) return;
            const rect = this.progressBar.getBoundingClientRect();
            const percent = (e.clientX - rect.left) / rect.width;
            this.audioPlayer.currentTime = percent * this.audioPlayer.duration;
        });
        
        // Audio events
        this.audioPlayer.addEventListener('timeupdate', () => this.updateProgress());
        this.audioPlayer.addEventListener('ended', () => this.onLineEnded());
        this.audioPlayer.addEventListener('error', (e) => {
            console.error('Audio error:', e);
            this.setStatus('Audio playback error', 'error');
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }
    
    handleKeyboard(e) {
        // Ignore if typing in textarea
        if (e.target === this.textInput) return;
        
        switch(e.code) {
            case 'Space':
                e.preventDefault();
                this.togglePlayPause();
                break;
            case 'KeyR':
                e.preventDefault();
                this.rewindLine();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.previousLine();
                break;
            case 'ArrowRight':
                e.preventDefault();
                this.nextLine();
                break;
            case 'KeyS':
                e.preventDefault();
                this.stop();
                break;
        }
    }
    
    async loadVoices() {
        try {
            const response = await fetch('/api/voices');
            const voices = await response.json();
            
            this.voiceSelect.innerHTML = '';
            for (const [name, value] of Object.entries(voices)) {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = name;
                if (value === this.selectedVoice) {
                    option.selected = true;
                }
                this.voiceSelect.appendChild(option);
            }
        } catch (error) {
            console.error('Failed to load voices:', error);
            this.setStatus('Failed to load voices', 'error');
        }
    }
    
    async checkModelStatus() {
        let statusCheckCount = 0;
        const MAX_STATUS_CHECKS = 300; // Stop after 10 minutes (300 * 2 seconds)
        
        const checkStatus = async () => {
            // Prevent infinite status checking
            if (statusCheckCount >= MAX_STATUS_CHECKS) {
                this.setStatus('Model loading timeout - please refresh page', 'error');
                this.hideLoading();
                return;
            }
            
            statusCheckCount++;
            
            try {
                const response = await fetch('/api/status');
                
                // Only process if response is OK
                if (!response.ok) {
                    console.warn('Status check failed:', response.status);
                    setTimeout(checkStatus, 5000); // Slower retry on error
                    return;
                }
                
                const data = await response.json();
                this.modelsLoaded = data.models_loaded;
                
                if (data.models_loaded) {
                    this.setStatus(`✅ BARK AI ready on ${data.gpu}`, 'success');
                    this.hideLoading();
                    // Stop checking once loaded
                    return;
                } else if (data.loading_status.status === 'loading') {
                    this.showLoading(data.loading_status.message);
                    this.setStatus(`⏳ ${data.loading_status.message}`, 'info');
                    setTimeout(checkStatus, 3000); // Check every 3 seconds while loading
                } else if (data.loading_status.status === 'error') {
                    this.setStatus(`❌ ${data.loading_status.message}`, 'error');
                    this.hideLoading();
                    return;
                } else {
                    setTimeout(checkStatus, 3000);
                }
            } catch (error) {
                console.warn('Status check error:', error);
                // Slower retry on network errors
                setTimeout(checkStatus, 5000);
            }
        };
        
        checkStatus();
    }
    
    updateTextStats() {
        const text = this.textInput.value;
        this.charCount.textContent = text.length.toLocaleString();
        const lines = text.split('\n').filter(l => l.trim()).length;
        this.lineCount.textContent = lines.toLocaleString();
    }
    
    clearText() {
        this.textInput.value = '';
        this.updateTextStats();
        this.lines = [];
        this.processedLines = [];
        this.renderLineList();
        this.updatePlayerDisplay();
        this.setStatus('Text cleared');
    }
    
    async pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            this.textInput.value = text;
            this.updateTextStats();
            this.setStatus('Text pasted from clipboard', 'success');
        } catch (error) {
            console.error('Clipboard access denied:', error);
            this.setStatus('Clipboard access denied - paste manually with Ctrl+V', 'error');
        }
    }
    
    async loadText() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            this.setStatus('Please enter or paste some text first', 'error');
            return;
        }
        
        this.showLoading('Processing text...');
        
        try {
            const response = await fetch('/api/process-text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    readCodeSymbols: this.readCodeSymbols
                })
            });
            
            const data = await response.json();
            
            this.lines = data.original_lines;
            this.processedLines = data.processed_lines;
            this.currentLineIndex = 0;
            this.currentLineStarted = false;
            
            this.renderLineList();
            this.updatePlayerDisplay();
            this.hideLoading();
            
            this.setStatus(`Loaded ${this.lines.length} lines - Press Play to start`, 'success');
        } catch (error) {
            console.error('Failed to process text:', error);
            this.hideLoading();
            this.setStatus('Failed to process text', 'error');
        }
    }
    
    renderLineList() {
        if (this.lines.length === 0) {
            this.lineList.innerHTML = '<p class="empty-message">Load text to see lines</p>';
            return;
        }
        
        this.lineList.innerHTML = this.lines.map((line, index) => `
            <div class="line-item ${index === this.currentLineIndex ? 'active' : ''} ${index < this.currentLineIndex ? 'played' : ''}" 
                 data-index="${index}" 
                 onclick="ttsPlayer.jumpToLine(${index})">
                <span class="line-item-num">${index + 1}</span>
                <span class="line-item-text" title="${this.escapeHtml(line)}">${this.escapeHtml(line)}</span>
            </div>
        `).join('');
        
        // Scroll active line into view
        const activeItem = this.lineList.querySelector('.line-item.active');
        if (activeItem) {
            activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    updatePlayerDisplay() {
        this.currentLineNum.textContent = this.lines.length > 0 ? this.currentLineIndex + 1 : 0;
        this.totalLines.textContent = this.lines.length;
        
        if (this.lines.length > 0 && this.currentLineIndex < this.lines.length) {
            this.currentLineText.textContent = this.lines[this.currentLineIndex];
        } else {
            this.currentLineText.textContent = 'No text loaded';
        }
        
        // Update line list highlighting
        this.lineList.querySelectorAll('.line-item').forEach((item, index) => {
            item.classList.toggle('active', index === this.currentLineIndex);
            item.classList.toggle('played', index < this.currentLineIndex);
        });
        
        // Scroll active line into view
        const activeItem = this.lineList.querySelector('.line-item.active');
        if (activeItem) {
            activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    
    updateProgress() {
        if (this.audioPlayer.duration) {
            const percent = (this.audioPlayer.currentTime / this.audioPlayer.duration) * 100;
            this.progressFill.style.width = `${percent}%`;
        }
    }
    
    async togglePlayPause() {
        if (!this.modelsLoaded) {
            this.setStatus('Please wait for AI models to load...', 'error');
            return;
        }
        
        if (this.lines.length === 0) {
            this.setStatus('Please load text first', 'error');
            return;
        }
        
        if (this.isPlaying && !this.isPaused) {
            // Pause
            this.audioPlayer.pause();
            this.isPaused = true;
            this.playPauseIcon.className = 'fas fa-play';
            this.setStatus('Paused');
        } else if (this.isPaused) {
            // Resume
            this.audioPlayer.play();
            this.isPaused = false;
            this.playPauseIcon.className = 'fas fa-pause';
            this.setStatus(`Playing line ${this.currentLineIndex + 1}...`);
        } else {
            // Start playing
            await this.playCurrentLine();
        }
    }
    
    async playCurrentLine() {
        if (this.currentLineIndex >= this.lines.length) {
            this.stop();
            this.setStatus('Playback complete!', 'success');
            return;
        }
        
        this.showLoading('🧠 BARK AI generating speech...');
        
        try {
            // Avoid overlapping generation requests
            if (this.playGenerateLock) return;
            this.playGenerateLock = true;

            // Determine if we should generate full file
            if (this.useFullGen) {
                this.showLoading('🧠 Generating full audio (this may take longer)...');
                const response = await fetch('/api/generate-full-audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: this.textInput.value.trim(),
                        voice: this.selectedVoice,
                        readCodeSymbols: this.readCodeSymbols,
                        hybrid: this.hybridMode,
                        force_cpu: this.useCpuAggressive
                    })
                });

                if (!response.ok) throw new Error('Failed to generate full audio');

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                this.audioCache.set(`full_${this.selectedVoice}`, audioUrl);

                this.currentChunkSize = this.lines.length; // full file covers all lines
                this.audioPlayer.src = audioUrl;
                this.audioPlayer.load();
                this.audioPlayer.currentTime = 0;
                await this.audioPlayer.play();

                this.currentLineStarted = true;
                this.isPlaying = true;
                this.isPaused = false;
                this.playPauseIcon.className = 'fas fa-pause';

                this.hideLoading();
                this.updatePlayerDisplay();
                this.setStatus('🎵 Playing full audio...');
                this.playGenerateLock = false;
                return;
            }

            // Context-aware generation: if the current line is very short, merge next line(s)
            const MIN_LEN = 45;
            let chunkText = this.processedLines[this.currentLineIndex];
            this.currentChunkSize = 1;
            while (chunkText.length < MIN_LEN && (this.currentLineIndex + this.currentChunkSize) < this.processedLines.length && this.currentChunkSize < 3) {
                chunkText = chunkText + ' ' + this.processedLines[this.currentLineIndex + this.currentChunkSize];
                this.currentChunkSize += 1;
            }

            const cacheKey = `${chunkText}_${this.selectedVoice}`;
            let audioUrl = this.audioCache.get(cacheKey);

            if (!audioUrl) {
                const response = await fetch('/api/generate-line-audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        text: chunkText,
                        voice: this.selectedVoice,
                        hybrid: this.hybridMode,
                        force_cpu: this.useCpuAggressive
                    })
                });

                if (!response.ok) {
                    throw new Error('Failed to generate audio');
                }

                const blob = await response.blob();
                audioUrl = URL.createObjectURL(blob);
                this.audioCache.set(cacheKey, audioUrl);
            }

            // Play
            // Pause any existing playback and reset position before loading new audio
            try { this.audioPlayer.pause(); } catch(e){}
            try { this.audioPlayer.currentTime = 0; } catch(e){}
            this.audioPlayer.playbackRate = 1.0; // Ensure normal speed

            this.audioPlayer.src = audioUrl;
            this.audioPlayer.load();

            // Wait for metadata to ensure duration is valid (avoid silent files)
            const metadataPromise = new Promise((resolve, reject) => {
                const onMeta = () => { cleanup(); resolve(true); };
                const onErr = () => { cleanup(); reject(new Error('Audio failed to load metadata')); };
                const cleanup = () => {
                    this.audioPlayer.removeEventListener('loadedmetadata', onMeta);
                    this.audioPlayer.removeEventListener('error', onErr);
                };
                this.audioPlayer.addEventListener('loadedmetadata', onMeta);
                this.audioPlayer.addEventListener('error', onErr);
                // Timeout after 8s
                setTimeout(() => { cleanup(); reject(new Error('Timed out waiting for audio metadata')); }, 8000);
            });

            await metadataPromise;

            if (!this.audioPlayer.duration || isNaN(this.audioPlayer.duration) || this.audioPlayer.duration < 0.01) {
                throw new Error('Generated audio appears empty');
            }

            this.audioPlayer.currentTime = 0;
            await this.audioPlayer.play();

            this.isPlaying = true;
            this.isPaused = false;
            this.currentLineStarted = true;
            this.playPauseIcon.className = 'fas fa-pause';

            this.hideLoading();
            this.updatePlayerDisplay();
            const endLine = Math.min(this.lines.length, this.currentLineIndex + this.currentChunkSize);
            this.setStatus(`🎵 Playing lines ${this.currentLineIndex+1}-${endLine} of ${this.lines.length}...`);

            // Preload next line/chunk
            this.preloadNextLine();
            this.playGenerateLock = false;

        } catch (error) {
            console.error('Playback error:', error);
            this.hideLoading();
            this.setStatus('Playback error - retrying...', 'error');
            this.playGenerateLock = false;

            // Retry once after a short delay
            setTimeout(() => this.playCurrentLine(), 1000);
        }
    }
    
    async preloadNextLine() {
        const nextIndex = this.currentLineIndex + 1;
        if (nextIndex >= this.lines.length) return;
        
        const text = this.processedLines[nextIndex];
        const cacheKey = `${text}_${this.selectedVoice}`;
        
        if (this.audioCache.has(cacheKey)) return;
        
        try {
            const response = await fetch('/api/generate-line-audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    voice: this.selectedVoice
                })
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                this.audioCache.set(cacheKey, audioUrl);
            }
        } catch (error) {
            // Silent fail for preload
        }
    }
    
    onLineEnded() {
        // Advance by the number of lines covered in the current chunk
        this.currentLineIndex += Math.max(1, this.currentChunkSize);
        this.currentLineStarted = false;
        this.currentChunkSize = 1;
        this.progressFill.style.width = '0%';

        if (this.currentLineIndex < this.lines.length) {
            this.updatePlayerDisplay();
            this.playCurrentLine();
        } else {
            // All lines complete
            this.stop();
            this.setStatus('Playback complete!', 'success');
        }
    }
    
    stop() {
        this.audioPlayer.pause();
        this.audioPlayer.currentTime = 0;
        this.isPlaying = false;
        this.isPaused = false;
        this.currentLineIndex = 0;
        this.currentLineStarted = false;
        this.progressFill.style.width = '0%';
        this.playPauseIcon.className = 'fas fa-play';
        this.updatePlayerDisplay();
        this.setStatus('Stopped');
    }
    
    rewindLine() {
        if (this.lines.length === 0) return;

        // Use a double-press detection with a short timeout
        if (!this.rewindPressedOnce) {
            // First press: restart current line
            this.rewindPressedOnce = true;
            clearTimeout(this.rewindTimer);
            this.rewindTimer = setTimeout(() => { this.rewindPressedOnce = false; }, 1400);

            this.audioPlayer.currentTime = 0;
            this.progressFill.style.width = '0%';

            if (this.isPaused) {
                this.setStatus(`Rewound line ${this.currentLineIndex + 1} - Press play to start`);
            } else if (this.isPlaying) {
                this.playCurrentLine();
            }

        } else {
            // Second press within timeout: go to previous line
            clearTimeout(this.rewindTimer);
            this.rewindPressedOnce = false;
            this.previousLine();
        }
    }
    
    previousLine() {
        if (this.lines.length === 0) return;

        if (this.currentLineIndex > 0) {
            // Stop current playback cleanly
            try { this.audioPlayer.pause(); } catch(e){}
            try { this.audioPlayer.currentTime = 0; } catch(e){}

            // Move back one line (consistent with user expectation)
            this.currentLineIndex = Math.max(0, this.currentLineIndex - 1);
            this.currentChunkSize = 1; // Reset chunk size when user manually navigates
            this.currentLineStarted = false;
            this.progressFill.style.width = '0%';
            this.updatePlayerDisplay();

            if (this.isPlaying && !this.isPaused) {
                // Start playing the newly selected line
                this.playCurrentLine();
            } else {
                this.setStatus(`Moved to line ${this.currentLineIndex + 1}`);
            }
        } else {
            try { this.audioPlayer.currentTime = 0; } catch(e){}
            this.progressFill.style.width = '0%';
            this.setStatus('Already at the first line');
        }
    }
    
    nextLine() {
        if (this.lines.length === 0) return;
        
        if (this.currentLineIndex < this.lines.length - 1) {
            this.currentLineIndex++;
            this.currentLineStarted = false;
            this.progressFill.style.width = '0%';
            this.updatePlayerDisplay();
            
            if (this.isPlaying && !this.isPaused) {
                this.playCurrentLine();
            } else {
                this.setStatus(`Moved to line ${this.currentLineIndex + 1}`);
            }
        } else {
            this.setStatus('Already at the last line');
        }
    }
    
    jumpToLine(index) {
        if (index < 0 || index >= this.lines.length) return;
        
        this.currentLineIndex = index;
        this.currentLineStarted = false;
        this.progressFill.style.width = '0%';
        this.updatePlayerDisplay();
        
        if (this.isPlaying && !this.isPaused) {
            this.playCurrentLine();
        } else {
            this.setStatus(`Jumped to line ${index + 1}`);
        }
    }
    
    async downloadAudio() {
        const text = this.textInput.value.trim();
        
        if (!text) {
            this.setStatus('Please enter or paste some text first', 'error');
            return;
        }
        
        this.showLoading('Generating full audio file...');
        
        try {
            const response = await fetch('/api/generate-full-audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    voice: this.selectedVoice,
                    readCodeSymbols: this.readCodeSymbols
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate audio');
            }
            
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = 'tts_output.mp3';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            this.hideLoading();
            this.setStatus('Audio downloaded!', 'success');
            
        } catch (error) {
            console.error('Download error:', error);
            this.hideLoading();
            this.setStatus('Download failed', 'error');
        }
    }
    
    async clearCache() {
        try {
            // Clear local cache
            this.audioCache.forEach(url => URL.revokeObjectURL(url));
            this.audioCache.clear();
            
            // Clear server cache
            await fetch('/api/clear-cache', { method: 'POST' });
            
            this.setStatus('Cache cleared', 'success');
        } catch (error) {
            console.error('Clear cache error:', error);
            this.setStatus('Failed to clear cache', 'error');
        }
    }
    
    showLoading(message = 'Loading...') {
        this.loadingText.textContent = message;
        this.loadingOverlay.classList.remove('hidden');
    }
    
    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }
    
    setStatus(message, type = 'info') {
        const icon = type === 'success' ? 'check-circle' : 
                     type === 'error' ? 'exclamation-circle' : 
                     'info-circle';
        
        this.statusMessage.className = `status-message ${type}`;
        this.statusMessage.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
    }
}

// Initialize the app
let ttsPlayer;
document.addEventListener('DOMContentLoaded', () => {
    ttsPlayer = new TTSPlayer();
});
