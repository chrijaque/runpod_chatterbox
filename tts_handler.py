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

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the handler from the forked repository
try:
    from chatterbox.vc import ChatterboxVC
    FORKED_HANDLER_AVAILABLE = True
    logger.info("✅ Successfully imported ChatterboxVC from forked repository")
except ImportError as e:
    FORKED_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Could not import ChatterboxVC from forked repository: {e}")

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    
    # Force download and setup NLTK punkt tokenizer
    logger.info("🔧 Setting up NLTK punkt tokenizer...")
    try:
        # Download punkt tokenizer explicitly
        nltk.download('punkt', quiet=True)

        # Verify it's available
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer downloaded and verified")

        # Test tokenization to ensure it works
        test_text = "This is a test. It has multiple sentences. Let's verify NLTK works."
        tokenizer = PunktSentenceTokenizer()
        test_sentences = tokenizer.tokenize(test_text)
        logger.info(f"✅ NLTK test successful: {len(test_sentences)} sentences tokenized")

    except Exception as e:
        logger.error(f"❌ NLTK setup failed: {e}")
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

model = None
forked_handler = None

# Local directory paths (use absolute paths for RunPod deployment)
VOICE_PROFILES_DIR = Path("/voice_profiles")
TTS_GENERATED_DIR = Path("/voice_samples")  # Use same directory as voice cloning for persistence
TEMP_VOICE_DIR = Path("/temp_voice")

# Log directory status (don't create them as they already exist in RunPod)
logger.info(f"Using existing directories:")
logger.info(f"  VOICE_PROFILES_DIR: {VOICE_PROFILES_DIR}")
logger.info(f"  TTS_GENERATED_DIR: {TTS_GENERATED_DIR}")
logger.info(f"  TEMP_VOICE_DIR: {TEMP_VOICE_DIR}")

# Chunking and processing class right
class TTSProcessor:
    def __init__(self, model: ChatterboxTTS, voice_profile_path: str, pause_ms: int = 100, max_chars: int = 500, forked_handler=None):
        """
        Initializes the TTSProcessor with a TTS model and a voice profile.

        :param model: The ChatterboxTTS model instance
        :param voice_profile_path: Path to the voice profile (.npy)
        :param pause_ms: Milliseconds of silence between chunks
        :param max_chars: Maximum number of characters per chunk
        :param forked_handler: Optional VoiceCloneHandler from forked repository
        """
        self.model = model
        self.forked_handler = forked_handler
        self.pause_ms = pause_ms
        self.max_chars = max_chars
        self.voice_profile_path = voice_profile_path  # Store the path for later use
        
        # Load the voice profile using the best available method
        try:
            if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
                # ChatterboxVC doesn't have load_voice_profile, so use model method
                if hasattr(model, 'load_voice_profile'):
                    self.voice_profile = model.load_voice_profile(voice_profile_path)
                    logger.info(f"✅ Voice profile loaded using model method from: {voice_profile_path}")
                else:
                    # Fallback: store path for old method
                    self.voice_profile = voice_profile_path
                    logger.warning(f"⚠️ Model doesn't have load_voice_profile - using path: {voice_profile_path}")
            elif hasattr(model, 'load_voice_profile'):
                # Use enhanced method from forked repository
                self.voice_profile = model.load_voice_profile(voice_profile_path)
                logger.info(f"✅ Voice profile loaded as ref_dict from: {voice_profile_path}")
            else:
                # Fallback: store path for old method
                self.voice_profile = voice_profile_path
                logger.warning(f"⚠️ Model doesn't have load_voice_profile - using path: {voice_profile_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load voice profile: {e}")
            # Fallback: store path for old method
            self.voice_profile = voice_profile_path

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits input text into sentence-aligned chunks based on max character count.

        :param text: Full story text
        :return: List of text chunks
        """
        if NLTK_AVAILABLE:
            try:
                # Use NLTK for proper sentence tokenization
                logger.info("📝 Using NLTK sentence tokenization")
                tokenizer = PunktSentenceTokenizer()
                sentences = tokenizer.tokenize(text)
                logger.info(f"📝 NLTK tokenization successful: {len(sentences)} sentences")
            except Exception as e:
                logger.warning(f"⚠️ NLTK tokenization failed: {e} - using fallback")
                sentences = self._simple_sentence_split(text)
        else:
            # Fallback to simple sentence splitting
            logger.info("📝 Using fallback sentence splitting (NLTK not available)")
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
        
        # Voice profile is already loaded in __init__
        
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
                    
                    # Generate audio tensor using the best available method
                    audio_tensor = None
                    
                    # Use the correct high-level method: ChatterboxTTS.generate() with voice_profile_path
                    try:
                        # For TTS, we'll use the standard generate method with the voice profile path
                        # Since we're in the TTSProcessor, we need to pass the voice profile path
                        # that was used to initialize this processor
                        
                        audio_tensor = self.model.generate(
                            text=chunk,
                            voice_profile_path=self.voice_profile_path,  # Use the path passed to __init__
                            temperature=0.8,
                            exaggeration=0.5,
                            cfg_weight=0.5
                        )
                        logger.info(f"✅ Chunk {i+1} generated using standard generate with voice_profile_path")
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Standard generate with voice_profile_path failed for chunk {i+1}: {e}")
                        
                        # Fallback: Use the old generate method
                        logger.info(f"🔄 Falling back to generate method for chunk {i+1}")
                        audio_tensor = self.model.generate(chunk, temperature=0.7)
                        logger.info(f"✅ Chunk {i+1} generated using fallback generate method")
                    
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

    def stitch_and_normalize(self, wav_paths: List[str], output_path: str) -> tuple:
        """
        Stitches WAV chunks together with pause and normalizes audio levels.

        :param wav_paths: List of temporary WAV file paths
        :param output_path: Final path to export the combined WAV file
        :return: Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        if PYDUB_AVAILABLE:
            # Use pydub for professional audio processing
            final = AudioSegment.empty()
            for p in wav_paths:
                seg = AudioSegment.from_wav(p)
                final += seg + AudioSegment.silent(self.pause_ms)
            normalized = effects.normalize(final)
            normalized.export(output_path, format="wav")
            
            # Load the saved file to get the tensor
            audio_tensor, sample_rate = torchaudio.load(output_path)
            duration = len(normalized) / 1000.0  # Convert ms to seconds
            return audio_tensor, sample_rate, duration
        else:
            # Fallback to torchaudio concatenation
            audio_chunks = []
            sample_rate = None
            for wav_path in wav_paths:
                audio_tensor, sr = torchaudio.load(wav_path)
                if sample_rate is None:
                    sample_rate = sr
                audio_chunks.append(audio_tensor)
                
                # Add silence between chunks
                silence_duration = int(self.pause_ms * sample_rate / 1000)
                silence = torch.zeros(1, silence_duration)
                audio_chunks.append(silence)
            
            # Concatenate all chunks
            if audio_chunks:
                final_audio = torch.cat(audio_chunks, dim=-1)
                torchaudio.save(output_path, final_audio, sample_rate)
                duration = final_audio.shape[-1] / sample_rate
                return final_audio, sample_rate, duration
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

    def process(self, text: str, output_path: str) -> tuple:
        """
        Full TTS pipeline: chunk → generate → stitch → clean.

        :param text: Input text to synthesize
        :param output_path: Path to save the final audio file
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting TTS processing for {len(text)} characters")
        
        chunks = self.chunk_text(text)
        logger.info(f"📦 Split into {len(chunks)} chunks")
        
        wav_paths = self.generate_chunks(chunks)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, output_path)
        
        self.cleanup(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        
        metadata = {
            "chunk_count": len(chunks),
            "output_path": output_path,
            "duration_sec": total_duration,
            "successful_chunks": len(wav_paths),
            "sample_rate": sample_rate
        }
        
        return audio_tensor, sample_rate, metadata

def initialize_model():
    global model, forked_handler
    
    if model is not None:
        logger.info("Model already initialized")
        return model
    
    logger.info("Initializing S3Token2Wav model...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
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
            
            # Check if it's from the forked repository
            if 'chrijaque/chatterbox_embed' in pip_info:
                logger.info("✅ Pip shows forked repository")
            else:
                logger.warning("⚠️ Pip shows original repository")
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

        # Initialize the forked repository handler if available
        if FORKED_HANDLER_AVAILABLE:
            logger.info("🔧 Initializing ChatterboxVC from forked repository...")
            try:
                # ChatterboxVC needs to be initialized with the s3gen model and device
                forked_handler = ChatterboxVC(
                    s3gen=model.s3gen,
                    device=model.device
                )
                logger.info("✅ ChatterboxVC initialized successfully")
                
                # Log handler capabilities
                handler_methods = [method for method in dir(forked_handler) if not method.startswith('_')]
                logger.info(f"📋 Available handler methods: {handler_methods}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize ChatterboxVC: {e}")
                forked_handler = None
        else:
            logger.warning("⚠️ ChatterboxVC not available - will use fallback methods")

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

        # Verify s3gen module source
        logger.info("🔍 ===== S3GEN VERIFICATION =====")
        if hasattr(model, "s3gen"):
            logger.info(f"📂 s3gen module path: {model.s3gen.__class__.__module__}")
            logger.info(f"📂 s3gen class: {model.s3gen.__class__}")
            logger.info(f"📂 s3gen class file: {model.s3gen.__class__.__module__}")
            
            # Check s3gen module file path
            try:
                import chatterbox.models.s3gen.s3gen as s3gen_module
                logger.info(f"📂 s3gen module file: {s3gen_module.__file__}")
                
                if 'chatterbox_embed' in s3gen_module.__file__:
                    logger.info("🎯 s3gen module is from FORKED repository")
                else:
                    logger.warning("⚠️ s3gen module is from ORIGINAL repository")
            except Exception as e:
                logger.warning(f"⚠️ Could not check s3gen module file: {e}")
            
            # Check if inference_from_text exists and its source
            if hasattr(model.s3gen, 'inference_from_text'):
                method = getattr(model.s3gen, 'inference_from_text')
                logger.info(f"📂 inference_from_text source: {method.__code__.co_filename}")
                logger.info(f"📂 inference_from_text line: {method.__code__.co_firstlineno}")
                
                if 'chatterbox_embed' in method.__code__.co_filename:
                    logger.info("🎯 inference_from_text is from FORKED repository")
                else:
                    logger.warning("⚠️ inference_from_text is from ORIGINAL repository")
            else:
                logger.warning("⚠️ inference_from_text method does NOT exist")
                
            # List all methods on s3gen
            s3gen_methods = [method for method in dir(model.s3gen) if not method.startswith('_')]
            logger.info(f"📋 Available s3gen methods: {s3gen_methods}")
        else:
            logger.warning("⚠️ Model does not have s3gen attribute")
        
        logger.info("🔍 ===== END S3GEN VERIFICATION =====")
        
        # Attach the T3 text‑to‑token encoder to S3Gen so that
        # s3gen.inference_from_text() works
        if hasattr(model, "s3gen") and hasattr(model, "t3"):
            model.s3gen.text_encoder = model.t3
            logger.info("📌 Attached text_encoder (model.t3) to model.s3gen")
        logger.info("✅ Model initialized on CUDA")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
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
    global model, forked_handler
    
    logger.info("🚀 ===== TTS HANDLER STARTED =====")
    logger.info(f"📥 Received event: {type(event)}")
    logger.info(f"📥 Event keys: {list(event.keys()) if isinstance(event, dict) else 'Not a dict'}")
    
    input = event.get('input', {})
    logger.info(f"📥 Input type: {type(input)}")
    logger.info(f"📥 Input keys: {list(input.keys()) if isinstance(input, dict) else 'Not a dict'}")
    
    # Check if this is a file download request
    if input.get('action') == 'download_file':
        return handle_file_download(input)
    
    # Check if this is a file listing request
    if input.get('action') == 'list_files':
        return list_available_files()
    
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
        logger.info(f"✅ Forked handler available: {forked_handler is not None}")
        
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
        logger.info(f"  - forked handler available: {forked_handler is not None}")
        
        # Load the profile using the best available method
        if forked_handler is not None and hasattr(forked_handler, 'set_target_voice'):
            logger.info("🔄 Loading profile using model method (ChatterboxVC doesn't have load_voice_profile)...")
            if hasattr(model, 'load_voice_profile'):
                profile = model.load_voice_profile(str(temp_profile_path))
                logger.info(f"✅ Voice profile loaded using model method")
            else:
                logger.error("❌ Model doesn't have load_voice_profile method")
                return {"status": "error", "message": "Voice profile support not available"}
        elif hasattr(model, 'load_voice_profile'):
            logger.info("🔄 Loading profile using model load_voice_profile method...")
            profile = model.load_voice_profile(str(temp_profile_path))
            logger.info(f"✅ Voice profile loaded successfully")
        else:
            logger.error("❌ Model doesn't have load_voice_profile method")
            logger.error("❌ This suggests the forked repository features are not available")
            return {"status": "error", "message": "Voice profile support not available"}
        
        logger.info(f"✅ Profile type: {type(profile)}")
        if hasattr(profile, 'shape'):
            logger.info(f"✅ Profile shape: {profile.shape}")
        if hasattr(profile, 'dtype'):
            logger.info(f"✅ Profile dtype: {profile.dtype}")
        
        # Generate speech using the profile
        logger.info(f"🎵 TTS: {voice_id} | Text length: {len(text)}")
        logger.info(f"🔍 DEBUG: Model loaded: {model is not None}")
        logger.info(f"🔍 DEBUG: Voice profile loaded: {profile is not None}")
        logger.info(f"🔍 DEBUG: Voice profile type: {type(profile)}")
        logger.info(f"🔍 DEBUG: Temp profile path exists: {temp_profile_path.exists()}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        start_time = time.time()
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tts_filename = TTS_GENERATED_DIR / f"TTS_{voice_id}_{timestamp}.wav"  # Use TTS_ prefix
        
        try:
            logger.info("🔄 Starting TTS processing...")
            
            # Use TTSProcessor for robust chunking and generation
            logger.info(f"🔍 DEBUG: Initializing TTSProcessor...")
            logger.info(f"🔍 DEBUG: Model type: {type(model)}")
            logger.info(f"🔍 DEBUG: Voice profile path: {temp_profile_path}")
            logger.info(f"🔍 DEBUG: Voice profile path exists: {temp_profile_path.exists()}")
            
            processor = TTSProcessor(
                model=model,
                voice_profile_path=str(temp_profile_path),
                pause_ms=150,  # Slightly longer pause for better flow
                max_chars=600,   # Conservative chunk size to avoid CUDA errors
                forked_handler=forked_handler  # Pass the forked handler
            )
            logger.info(f"🔍 DEBUG: TTSProcessor initialized successfully")
            
            # Process the text
            logger.info(f"🔍 DEBUG: Starting text processing...")
            logger.info(f"🔍 DEBUG: Input text length: {len(text)}")
            logger.info(f"🔍 DEBUG: Output filename: {tts_filename}")
            
            audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
            logger.info(f"🔍 DEBUG: Text processing completed")
            logger.info(f"🔍 DEBUG: Processing result: {result}")
            
            generation_time = time.time() - start_time
            logger.info(f"✅ TTS generated in {generation_time:.2f}s")
            logger.info(f"📊 Processing stats: {result}")
            
            # Get audio tensor directly from TTSProcessor (same as voice cloning)
            logger.info(f"🔍 DEBUG: Audio tensor shape: {audio_tensor.shape}")
            logger.info(f"🔍 DEBUG: Audio tensor dtype: {audio_tensor.dtype}")
            logger.info(f"🔍 DEBUG: Sample rate: {sample_rate}")
            
            # Save file immediately after generation (same as voice cloning)
            local_filename = TTS_GENERATED_DIR / f"TTS_{voice_id}_{timestamp}.wav"  # Use TTS_ prefix to distinguish from voice cloning
            torchaudio.save(local_filename, audio_tensor, sample_rate)
            logger.info(f"💾 Saved TTS: {local_filename.name}")
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ Failed to generate TTS after {generation_time:.2f}s")
            logger.error(f"❌ Error type: {type(e)}")
            logger.error(f"❌ Error message: {str(e)}")
            
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
                    audio_tensor, sample_rate, result = processor.process(text, str(tts_filename))
                    generation_time = time.time() - start_time
                    logger.info(f"✅ TTS generated with smaller chunks in {generation_time:.2f}s")
                except Exception as retry_error:
                    logger.error(f"❌ Retry also failed: {retry_error}")
                    return {"status": "error", "message": f"Failed to generate TTS even with chunking: {retry_error}"}
            else:
                return {"status": "error", "message": f"Failed to generate TTS: {e}"}
        
        # Check response size and handle large files
        try:
            audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)
            base64_size = len(audio_base64)
            logger.info(f"📤 Base64 size: {base64_size:,} chars ({base64_size/1024/1024:.1f} MB)")
            
            # If response is too large, return file path instead
            if base64_size > 10_000_000:  # 10MB limit
                logger.warning(f"⚠️ Response too large ({base64_size/1024/1024:.1f} MB) - returning file path only")
                response = {
                    "status": "success",
                    "audio_base64": None,  # No audio data in response
                    "file_path": str(tts_filename),
                    "file_size_mb": base64_size/1024/1024,
                    "metadata": {
                        "voice_id": voice_id,
                        "voice_name": voice_id.replace('voice_', ''),
                        "text_input": text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
                        "generation_time": generation_time,
                        "sample_rate": model.sr,
                        "audio_shape": list(audio_tensor.shape),
                        "tts_file": str(tts_filename),
                        "timestamp": timestamp,
                        "response_type": "file_path_only"
                    }
                }
            else:
                # Normal response with audio data
                response = {
                    "status": "success",
                    "audio_base64": audio_base64,
                    "metadata": {
                        "voice_id": voice_id,
                        "voice_name": voice_id.replace('voice_', ''),
                        "text_input": text[:500] + "..." if len(text) > 500 else text,  # Truncate long text
                        "generation_time": generation_time,
                        "sample_rate": model.sr,
                        "audio_shape": list(audio_tensor.shape),
                        "tts_file": str(tts_filename),
                        "timestamp": timestamp,
                        "response_type": "audio_data"
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Failed to convert audio to base64: {e}")
            return {"status": "error", "message": f"Failed to convert audio to base64: {e}"}
        
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

def handle_file_download(input):
    """Handle file download requests"""
    logger.info("📁 ===== FILE DOWNLOAD REQUEST =====")
    
    file_path = input.get('file_path')
    if not file_path:
        logger.error("❌ No file_path provided in download request")
        return {"status": "error", "message": "No file_path provided"}
    
    logger.info(f"📁 Requested file: {file_path}")
    
    # Method 1: Try direct file access
    logger.info("🔍 Method 1: Direct file system access")
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"✅ File exists: {file_path}")
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            # Read file and convert to base64
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            audio_base64 = base64.b64encode(file_data).decode('utf-8')
            logger.info(f"✅ File read successfully: {len(audio_base64):,} base64 chars")
            
            return {
                "status": "success",
                "method": "direct_file_access",
                "file_path": file_path,
                "file_size_bytes": file_size,
                "file_size_mb": file_size/1024/1024,
                "audio_base64": audio_base64,
                "message": "File downloaded via direct file system access"
            }
        else:
            logger.error(f"❌ File not found: {file_path}")
            return {"status": "error", "message": f"File not found: {file_path}"}
            
    except Exception as e:
        logger.error(f"❌ Direct file access failed: {e}")
        return {"status": "error", "message": f"Direct file access failed: {e}"}

def list_available_files():
    """List all available TTS files for debugging"""
    logger.info("📂 ===== LISTING AVAILABLE FILES =====")
    
    try:
        if TTS_GENERATED_DIR.exists():
            files = list(TTS_GENERATED_DIR.glob("TTS_*.wav"))  # Look for TTS_ prefixed files
            logger.info(f"📂 Found {len(files)} TTS files:")
            
            for file in files:
                file_size = file.stat().st_size
                logger.info(f"  📄 {file.name}: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            return {
                "status": "success",
                "files": [{"name": f.name, "size_bytes": f.stat().st_size, "size_mb": f.stat().st_size/1024/1024} for f in files]
            }
        else:
            logger.error(f"❌ TTS directory not found: {TTS_GENERATED_DIR}")
            return {"status": "error", "message": f"TTS directory not found: {TTS_GENERATED_DIR}"}
            
    except Exception as e:
        logger.error(f"❌ Failed to list files: {e}")
        return {"status": "error", "message": f"Failed to list files: {e}"}

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