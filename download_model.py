import sys
import traceback
import torch

def main():
    try:
        print('Python version:', sys.version)
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('CUDA device count:', torch.cuda.device_count())
            print('CUDA device:', torch.cuda.get_device_name(0))
        
        print('Importing ChatterboxTTS...')
        from chatterbox.tts import ChatterboxTTS
        print('Import successful.')
        
        print('Downloading model...')
        model = ChatterboxTTS.from_pretrained(device='cuda')
        print('Model downloaded and loaded successfully')
        
    except Exception as e:
        print('Error occurred:')
        print(str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 