import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
import torch
import logging
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    
    # Download required NLTK data if not available
    try:
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer available")
    except LookupError:
        logger.warning("⚠️ NLTK punkt tokenizer not found - downloading...")
        try:
            nltk.download('punkt', quiet=True)
            NLTK_AVAILABLE = True
            logger.info("✅ NLTK punkt tokenizer downloaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to download NLTK punkt: {e}")
            NLTK_AVAILABLE = False
            logger.warning("⚠️ Will use simple text splitting instead")
    
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
TTS_GENERATED_DIR = Path("/tts_generated")
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Chunking and processing class
class TTSProcessor:
    def __init__(self, model: ChatterboxTTS, voice_profile_path: str, pause_ms: int = 100, max_chars: int = 500):
        """
        Initializes the TTSProcessor with a TTS model and a voice profile.

        :param model: The ChatterboxTTS model instance
        :param voice_profile_path: Path to the voice profile (.npy)
        :param pause_ms: Milliseconds of silence between chunks
        :param max_chars: Maximum number of characters per chunk
        """
        self.model = model
        self.voice_profile = voice_profile_path
        self.pause_ms = pause_ms
        self.max_chars = max_chars

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits input text into sentence-aligned chunks based on max character count.

        :param text: Full story text
        :return: List of text chunks
        """
        if NLTK_AVAILABLE:
            try:
                # Use NLTK for proper sentence tokenization
                sentences = sent_tokenize(text)
                logger.debug(f"📝 NLTK tokenization: {len(sentences)} sentences")
            except Exception as e:
                logger.warning(f"⚠️ NLTK tokenization failed: {e} - using fallback")
                sentences = self._simple_sentence_split(text)
        else:
            # Fallback to simple sentence splitting
            sentences = self._simple_sentence_split(text)
        
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self.max_chars:
                current += (" " + sent).strip()
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = sent
        if current.strip():
            chunks.append(current.strip())
        
        logger.info(f"📦 Text chunking: {len(sentences)} sentences → {len(chunks)} chunks")
        return chunks
    
    def _simple_sentence_split(self, text: str) -> List[str]:
        """
        Simple sentence splitting fallback when NLTK is not available.
        
        :param text: Text to split into sentences
        :return: List of sentences
        """
        # Clean up the text
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Split on sentence endings
        sentence_endings = ['.', '!', '?']
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in sentence_endings:
                sentence = current.strip()
                if sentence:
                    sentences.append(sentence)
                current = ""
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        # Filter out empty sentences
        sentences = [s for s in sentences if s.strip()]
        
        return sentences

    def generate_chunks(self, chunks: List[str]) -> List[str]:
        """
        Generates speech from each text chunk and stores temporary WAV files.

        :param chunks: List of text chunks
        :return: List of file paths to temporary WAV files
        """
        wav_paths = []
        
        # Prepare voice profile once
        self.model.prepare_conditionals_with_voice_profile(self.voice_profile, exaggeration=0.6)
        
        for i, chunk in enumerate(chunks):
            logger.info(f"🔄 Generating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            # Retry logic for each chunk
            chunk_success = False
            retry_count = 0
            max_retries = 2  # 2 retries = 3 total attempts
            
            while not chunk_success and retry_count <= max_retries:
                try:
                    # Clear GPU cache before each attempt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate audio tensor
                    audio_tensor = self.model.generate(chunk, temperature=0.7)
                    
                    # Save to temporary file
                    temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{i}.wav", delete=False)
                    torchaudio.save(temp_wav.name, audio_tensor, self.model.sr)
                    wav_paths.append(temp_wav.name)
                    
                    logger.info(f"✅ Chunk {i+1} generated | Shape: {audio_tensor.shape}")
                    chunk_success = True
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.warning(f"⚠️ Chunk {i+1} failed (attempt {retry_count}/{max_retries + 1}): {e}")
                        logger.info(f"🔄 Retrying chunk {i+1}...")
                    else:
                        # Final failure - stop processing
                        logger.error(f"❌ Chunk {i+1} failed after {max_retries + 1} attempts: {e}")
                        logger.error(f"❌ Stopping TTS processing due to chunk failure")
                        
                        # Clean up any successfully generated chunks
                        self.cleanup(wav_paths)
                        
                        # Raise exception to stop processing
                        raise RuntimeError(f"Chunk {i+1} failed after {max_retries + 1} attempts: {e}")
        
        return wav_paths

    def stitch_and_normalize(self, wav_paths: List[str], output_path: str) -> float:
        """
        Stitches WAV chunks together with pause and normalizes audio levels.

        :param wav_paths: List of temporary WAV file paths
        :param output_path: Final path to export the combined WAV file
        :return: Total duration of the final audio in seconds
        """
        if PYDUB_AVAILABLE:
            # Use pydub for professional audio processing
            final = AudioSegment.empty()
            for p in wav_paths:
                seg = AudioSegment.from_wav(p)
                final += seg + AudioSegment.silent(self.pause_ms)
            normalized = effects.normalize(final)
            normalized.export(output_path, format="wav")
            return len(normalized) / 1000.0  # Convert ms to seconds
        else:
            # Fallback to torchaudio concatenation
            audio_chunks = []
            for wav_path in wav_paths:
                audio_tensor, sample_rate = torchaudio.load(wav_path)
                audio_chunks.append(audio_tensor)
                
                # Add silence between chunks
                silence_duration = int(self.pause_ms * sample_rate / 1000)
                silence = torch.zeros(1, silence_duration)
                audio_chunks.append(silence)
            
            # Concatenate all chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=-1)
                torchaudio.save(output_path, final_audio, sample_rate)
                return final_audio.shape[-1] / sample_rate
            else:
                raise RuntimeError("No audio chunks to concatenate")

    def cleanup(self, wav_paths: List[str]):
        """
        Deletes temporary WAV files to clean up disk space.

        :param wav_paths: List of paths to temporary WAV files
        """
        for path in wav_paths:
            try:
                os.remove(path)
                logger.debug(f"🧹 Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {path} — {e}")

    def process(self, text: str, output_path: str) -> Dict:
        """
        Full TTS pipeline: chunk → generate → stitch → clean.

        :param text: Input text to synthesize
        :param output_path: Path to save the final audio file
        :return: Dictionary with metadata including duration, chunk count, and file paths
        """
        logger.info(f"🎵 Starting TTS processing for {len(text)} characters")
        
        chunks = self.chunk_text(text)
        logger.info(f"📦 Split into {len(chunks)} chunks")
        
        wav_paths = self.generate_chunks(chunks)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Stitching {len(wav_paths)} audio chunks...")
        total_duration = self.stitch_and_normalize(wav_paths, output_path)
        
        self.cleanup(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        
        return {
            "chunk_count": len(chunks),
            "output_path": output_path,
            "duration_sec": total_duration,
            "successful_chunks": len(wav_paths)
        }

def initialize_model():
    global model
    
    logger.info("🔧 ===== MODEL INITIALIZATION =====")
    
    if model is not None:
        logger.info("✅ Model already initialized")
        logger.info(f"✅ Model type: {type(model)}")
        return model
    
    logger.info("🔄 Initializing ChatterboxTTS model for TTS generation...")
    
    # Check CUDA availability
    logger.info("🔍 Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    logger.info(f"🔍 CUDA available: {cuda_available}")
    
    if not cuda_available:
        logger.error("❌ CUDA is required but not available")
        raise RuntimeError("CUDA is required but not available")
    
    logger.info(f"✅ CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"✅ CUDA device name: {torch.cuda.get_device_name(0)}")
    logger.info(f"✅ CUDA device capability: {torch.cuda.get_device_capability(0)}")
    
    try:
        # Debug: Check which chatterbox repository is being used
        import chatterbox
        import os
        
        logger.info(f"📦 chatterbox module loaded from: {chatterbox.__file__}")
        
        # Enhanced Debug: Log chatterbox installation details with dependency analysis
        repo_path = os.path.dirname(chatterbox.__file__)
        git_path = os.path.join(repo_path, '.git')
        
        # Check if it's a git repository
        if os.path.exists(git_path):
            logger.info(f"📁 chatterbox installed as git repo: {repo_path}")
            try:
                import subprocess
                commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD']).decode().strip()
                logger.info(f"🔢 Git commit: {commit_hash}")
                remote_url = subprocess.check_output(['git', '-C', repo_path, 'remote', 'get-url', 'origin']).decode().strip()
                logger.info(f"🌐 Git remote: {remote_url}")
                
                # Check if it's the forked repository
                if 'chrijaque/chatterbox_embed' in remote_url:
                    logger.info("✅ This is the CORRECT forked repository!")
                else:
                    logger.error("❌ This is NOT the forked repository - using wrong repo!")
            except Exception as e:
                logger.warning(f"⚠️ Could not get git info: {e}")
        else:
            logger.error(f"📁 chatterbox not installed as git repo (no .git directory found)")
            logger.error(f"❌ This indicates PyPI package installation instead of git repo")
        
        # Check pip installation details
        try:
            import subprocess
            pip_info = subprocess.check_output(['pip', 'show', 'chatterbox-tts']).decode().strip()
            logger.info(f"📋 Pip package info:\n{pip_info}")
        except Exception as e:
            logger.warning(f"⚠️ Could not get pip info: {e}")
        
        # Check for dependency conflicts
        try:
            import subprocess
            deps = subprocess.check_output(['pip', 'list']).decode().strip()
            chatterbox_deps = [line for line in deps.split('\n') if 'chatterbox' in line.lower()]
            if chatterbox_deps:
                logger.info(f"🔍 Found chatterbox-related packages:\n" + '\n'.join(chatterbox_deps))
            else:
                logger.info("🔍 No chatterbox-related packages found in pip list")
        except Exception as e:
            logger.warning(f"⚠️ Could not check dependencies: {e}")
        
        logger.info("🔄 Loading ChatterboxTTS model...")
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully on CUDA device")
        logger.info(f"✅ Model type: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")

        # Additional model introspection logs
        import inspect
        logger.info(f"📦 Model class: {model.__class__}")
        logger.info(f"📁 Model module: {model.__class__.__module__}")
        logger.info(f"📂 Loaded model from file: {inspect.getfile(model.__class__)}")
        logger.info(f"🧠 Model dir(): {dir(model)}")
        logger.info(f"🔎 Has method load_voice_profile: {hasattr(model, 'load_voice_profile')}")

        # List all methods that contain 'voice' or 'profile'
        voice_methods = [method for method in dir(model) if 'voice' in method.lower() or 'profile' in method.lower()]
        logger.info(f"🔍 Voice/Profile related methods: {voice_methods}")

        # Fast-fail check for required method
        assert hasattr(model, 'load_voice_profile'), "🚨 Loaded model is missing `load_voice_profile`. Wrong class?"

        # Check model capabilities
        logger.info("🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_profile: {hasattr(model, 'load_voice_profile')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_profile: {hasattr(model, 'save_voice_profile')}")
        
    except Exception as e:
        logger.error("❌ Failed to initialize model")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error message: {str(e)}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        raise

def list_files_for_debug():
    """List files in our directories for debugging"""
    logger.info("📂 Directory contents:")
    for directory in [VOICE_PROFILES_DIR, TTS_GENERATED_DIR]:
        if directory.exists():
            files = list(directory.glob("*"))
            logger.info(f"  {directory}: {[f.name for f in files]} ({len(files)} files)")
        else:
            logger.info(f"  {directory}: [DIRECTORY NOT FOUND]")

def handler(event, responseFormat="base64"):
    """Handle TTS generation requests using saved voice embeddings"""
    global model
    
    logger.info("🚀 ===== TTS HANDLER STARTED =====")
    logger.info(f"📥 Received event: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input = event.get('input', {})
    logger.info(f"📥 Input type: {type(input)}")
    logger.info(f"📥 Input keys: {list(input.keys()) if isinstance(input, dict) else 'Not a dict'}")
    
    # Extract TTS parameters
    text = input.get('text')
    voice_id = input.get('voice_id')
    profile_base64 = input.get('profile_base64')  # New: voice profile data
    responseFormat = input.get('responseFormat', 'base64')
    
    logger.info(f"📋 Extracted parameters:")
    logger.info(f"  - text: {text[:50]}{'...' if text and len(text) > 50 else ''} (length: {len(text) if text else 0})")
    logger.info(f"  - voice_id: {voice_id}")
    logger.info(f"  - has_profile_base64: {bool(profile_base64)}")
    logger.info(f"  - profile_size: {len(profile_base64) if profile_base64 else 0}")
    logger.info(f"  - responseFormat: {responseFormat}")
    
    if not text or not voice_id or not profile_base64:
        logger.error("❌ Missing required parameters")
        logger.error(f"  - text provided: {bool(text)}")
        logger.error(f"  - voice_id provided: {bool(voice_id)}")
        logger.error(f"  - profile_base64 provided: {bool(profile_base64)}")
        return {"status": "error", "message": "text, voice_id, and profile_base64 are required"}
    
    logger.info(f"🎤 TTS request validated: voice_id={voice_id}, text_length={len(text)}")
    
    try:
        logger.info("🔍 ===== VOICE EMBEDDING PROCESSING =====")
        
        # Check if model is initialized
        if model is None:
            logger.error("❌ Model not initialized")
            return {"status": "error", "message": "Model not initialized"}
        
        logger.info(f"✅ Model is initialized: {type(model)}")
        logger.info(f"✅ Model device: {getattr(model, 'device', 'Unknown')}")
        logger.info(f"✅ Model sample rate: {getattr(model, 'sr', 'Unknown')}")
        
        # Decode the voice profile data
        logger.info("🔄 Decoding voice profile data...")
        try:
            profile_data = base64.b64decode(profile_base64)
            logger.info(f"✅ Voice profile data decoded: {len(profile_data)} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to decode voice profile data: {e}")
            return {"status": "error", "message": f"Failed to decode voice profile data: {e}"}
        
        # Save the voice profile data to a temporary file
        logger.info("🔄 Saving voice profile data to temporary file...")
        temp_profile_path = TEMP_VOICE_DIR / f"{voice_id}_temp.npy"
        
        try:
            with open(temp_profile_path, 'wb') as f:
                f.write(profile_data)
            logger.info(f"✅ Temporary profile file created: {temp_profile_path}")
            logger.info(f"✅ File size: {temp_profile_path.stat().st_size} bytes")
        except Exception as e:
            logger.error(f"❌ Failed to save temporary profile file: {e}")
            return {"status": "error", "message": f"Failed to save temporary profile file: {e}"}
        
        # Check if model has the required method
        logger.info(f"🔍 Checking model capabilities:")
        logger.info(f"  - has load_voice_profile: {hasattr(model, 'load_voice_profile')}")
        logger.info(f"  - has generate: {hasattr(model, 'generate')}")
        logger.info(f"  - has save_voice_profile: {hasattr(model, 'save_voice_profile')}")
        
        # Load the profile using the forked repository method
        if hasattr(model, 'load_voice_profile'):
            logger.info("🔄 Loading profile using load_voice_profile method...")
            profile = model.load_voice_profile(str(temp_profile_path))
            logger.info(f"✅ Voice profile loaded successfully")
            logger.info(f"✅ Profile type: {type(profile)}")
            if hasattr(profile, 'shape'):
                logger.info(f"✅ Profile shape: {profile.shape}")
            if hasattr(profile, 'dtype'):
                logger.info(f"✅ Profile dtype: {profile.dtype}")
        else:
            logger.error("❌ Model doesn't have load_voice_profile method")
            logger.error("❌ This suggests the forked repository features are not available")
            return {"status": "error", "message": "Voice profile support not available"}
        
        # Generate speech using the profile
        logger.info(f"🎵 TTS: {voice_id} | Text length: {len(text)}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        start_time = time.time()
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = TTS_GENERATED_DIR / f"tts_{voice_id}_{timestamp}.wav"
        
        try:
            logger.info("🔄 Starting TTS processing...")
            
            # Use TTSProcessor for robust chunking and generation
            processor = TTSProcessor(
                model=model,
                voice_profile_path=str(temp_profile_path),
                pause_ms=150,  # Slightly longer pause for better flow
                max_chars=600   # Conservative chunk size to avoid CUDA errors
            )
            
            # Process the text
            result = processor.process(text, str(tts_filename))
            
            generation_time = time.time() - start_time
            logger.info(f"✅ TTS generated in {generation_time:.2f}s")
            logger.info(f"📊 Processing stats: {result}")
            
            # Load the generated audio for base64 conversion
            audio_tensor, sample_rate = torchaudio.load(str(tts_filename))
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Failed to generate TTS after {generation_time:.2f}s")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error message: {str(e)}")
            logger.error(f"❌ Error details: {e}")
            
            # Try with even smaller chunks if it's a CUDA error
            if "CUDA error" in str(e) and len(text) > 300:
                logger.info("🔄 Retrying with smaller chunks due to CUDA error...")
                try:
                    processor = TTSProcessor(
                        model=model,
                        voice_profile_path=str(temp_profile_path),
                        pause_ms=100,
                        max_chars=300  # Even smaller chunks
                    )
                    result = processor.process(text, str(tts_filename))
                    audio_tensor, sample_rate = torchaudio.load(str(tts_filename))
                    generation_time = time.time() - start_time
                    logger.info(f"✅ TTS generated with smaller chunks in {generation_time:.2f}s")
                except Exception as retry_error:
                    logger.error(f"❌ Retry also failed: {retry_error}")
                    return {"status": "error", "message": f"Failed to generate TTS even with chunking: {retry_error}"}
            else:
                return {"status": "error", "message": f"Failed to generate TTS: {e}"}
        
        # Convert to base64
        try:
            audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
            logger.info(f"📤 Base64: {len(audio_base64)} chars")
        except Exception as e:
            logger.error(f"❌ Failed to convert audio to base64: {e}")
            return {"status": "error", "message": f"Failed to convert audio to base64: {e}"}
        
        # Create response
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "voice_id": voice_id,
                "voice_name": voice_id.replace('voice_', ''),  # Extract name from ID
                "text_input": text,
                "generation_time": generation_time,
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape),
                "tts_file": str(tts_filename),
                "timestamp": timestamp
            }
        }
        
        logger.info(f"📤 Response ready | Time: {generation_time:.2f}s | File: {tts_filename.name}")
        
        # Clean up temporary profile file
        try:
            if temp_profile_path.exists():
                os.unlink(temp_profile_path)
        except Exception as cleanup_error:
            logger.warning(f"⚠️ Failed to clean up temp file: {cleanup_error}")
        
        logger.info("🎉 TTS completed successfully")
        return response
        
    except Exception as e:
        logger.error("💥 ===== TTS HANDLER FAILED =====")
        logger.error(f"❌ TTS request failed: {e}")
        logger.error(f"❌ Error type: {type(e)}")
        logger.error(f"❌ Error details: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            
            # Save audio tensor to temporary file
            torchaudio.save(tmp_filename, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_filename)
            
            # Encode as base64
            base64_data = base64.b64encode(audio_data).decode('utf-8')
            
            return base64_data
            
    except Exception as e:
        logger.error(f"❌ Error converting audio to base64: {e}")
        raise

if __name__ == '__main__':
    logger.info("🚀 TTS Handler starting...")
    
    try:
        logger.info("🔧 Initializing model...")
        initialize_model()
        logger.info("✅ Model ready")
        
        logger.info("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({'handler': handler })
        
    except Exception as e:
        logger.error(f"💥 TTS Handler startup failed: {e}")
        raise 