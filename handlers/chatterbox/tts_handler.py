import time
import runpod

print("[BOOT] handler file loaded", flush=True)

def handler(event):
    print("[HANDLER] event received:", event, flush=True)
    return {"ok": True}

if __name__ == "__main__":
    print("[BOOT] starting runpod serverless", flush=True)
    runpod.serverless.start({"handler": handler})

    # safety: never exit
    while True:
        time.sleep(10)