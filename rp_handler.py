import runpod
import time  
import torchaudio as ta
import yt_dlp
import os
from chatterbox.tts import ChatterboxTTS
from pathlib import Path

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    try:
        video_url = "https://www.youtube.com/shorts/jcNzoONhrmE"  # Replace with actual URL
        dl_info = download_youtube_audio(video_url, output_path="./my_audio", audio_format="wav")

        wav_file = "./my_audio/" + dl_info["title"] + ".wav"
        # wav_file, os.path.exists(wav_file)

        model = ChatterboxTTS.from_pretrained(device="cuda")

        wav = model.generate(
            prompt,
            audio_prompt_path=wav_file
        )
        ta.save("test-4.wav", wav, model.sr)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"{e}" 

    # Replace the sleep code with your Python function to generate images, text, or run any machine learning workload
    time.sleep(seconds)  
    
    return prompt 

def download_youtube_audio(url, output_path="./downloads", audio_format="mp3"):
    """
    Download audio from a YouTube video
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save the audio file
        audio_format (str): Audio format (mp3, wav, m4a, etc.)
    
    Returns:
        str: Path to the downloaded audio file, or None if download failed
    """
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best quality audio
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Output filename template
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',  # Audio quality in kbps
        }],
        'postprocessor_args': [
            '-ar', '44100'  # Set sample rate
        ],
        'prefer_ffmpeg': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            print(f"Title: {info.get('title', 'Unknown')}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print(f"Uploader: {info.get('uploader', 'Unknown')}")
            
            # Construct the expected output filename
            title = info.get('title', 'Unknown')
            # Clean the title for filename (remove invalid characters)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            expected_filepath = os.path.join(output_path, f"{safe_title}.{audio_format}")
            
            # Download the audio
            print("Downloading audio...")
            ydl.download([url])
            print("Download completed successfully!")
            
            return info
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
