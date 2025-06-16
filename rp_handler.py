import runpod
import time  
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def handler(event):
    print(f"Worker Start")
    input = event['input']
    
    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")
    
    model = ChatterboxTTS.from_pretrained(device="cuda")

    # text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
    wav = model.generate(prompt)
    ta.save("test-1.wav", wav, model.sr)

    # Replace the sleep code with your Python function to generate images, text, or run any machine learning workload
    time.sleep(seconds)  
    
    return prompt 

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
