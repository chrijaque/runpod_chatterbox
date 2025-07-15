import runpod
import time  
import torchaudio 
import os
import tempfile
import base64
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

model = None
output_filename = "output.wav"

def handler(event, responseFormat="base64"):
    input = event['input']    
    prompt = input.get('prompt')
    audio_data = input.get('audio_data')  # Base64 encoded audio data
    audio_format = input.get('audio_format', 'wav')  # Format of the input audio

    if not prompt or not audio_data:
        return {"status": "error", "message": "Both prompt and audio_data are required"}

    print(f"New request. Prompt: {prompt}")
    
    try:
        # Save the uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as tmp_file:
            audio_bytes = base64.b64decode(audio_data)
            tmp_file.write(audio_bytes)
            audio_file_path = tmp_file.name

        # Load and resample the audio if needed
        audio_input, sr = torchaudio.load(audio_file_path)
        if sr != 44100:  # Ensure consistent sample rate
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio_input = resampler(audio_input)

        # Prompt Chatterbox
        audio_tensor = model.generate(
            prompt,
            audio_prompt_path=audio_file_path
        )

        # Save as WAV
        torchaudio.save(output_filename, audio_tensor, model.sr)

        # Clean up the temporary input file
        os.unlink(audio_file_path)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if 'audio_file_path' in locals():
            try:
                os.unlink(audio_file_path)
            except:
                pass
        return {"status": "error", "message": str(e)}

    # Convert to base64 string
    audio_base64 = audio_tensor_to_base64(audio_tensor, model.sr)

    if responseFormat == "base64":
        # Return base64
        response = {
            "status": "success",
            "audio_base64": audio_base64,
            "metadata": {
                "sample_rate": model.sr,
                "audio_shape": list(audio_tensor.shape)
            }
        }
    elif responseFormat == "binary":
        with open(output_filename, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up the file
        os.remove(output_filename)
        
        response = audio_data  # Just return the base64 string

    return response 

def audio_tensor_to_base64(audio_tensor, sample_rate):
    """Convert audio tensor to base64 encoded WAV data."""
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            torchaudio.save(tmp_file.name, audio_tensor, sample_rate)
            
            # Read back as binary data
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            # Encode as base64
            return base64.b64encode(audio_data).decode('utf-8')
            
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        raise

def initialize_model():
    global model
    
    if model is not None:
        print("Model already initialized")
        return model
    
    print("Initializing ChatterboxTTS model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")
    print("Model initialized")

if __name__ == '__main__':
    initialize_model()
    runpod.serverless.start({'handler': handler })
