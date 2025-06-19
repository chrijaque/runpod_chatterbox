# RunPod Serverless Endpoint for Voice Cloning 

## Overview
* REST API
* Call Endpoint API with YouTube link and prompt
* Downloads first 60 seconds of provided YouTube audio
* Invokes [Chatterbox TTS model](https://github.com/resemble-ai/chatterbox) 
* Returns base64 encoded WAV audio

## Deploy 

* Get a RunPod account, maybe use my [referral link](https://runpod.io?ref=3lyngjfm)

Deploy endpoint ([docs](https://docs.runpod.io/serverless/overview#runpod-hub)):
* Go to https://console.runpod.io/serverless
* -> New Endpoint
* -> GitHub Repo, choose https://github.com/geronimi73/runpod_chatterbox
* Check `Endpoint ID` of your deployed endpoint at https://console.runpod.io/serverless
* Create an `API key` at https://console.runpod.io/user/settings

## Usage 

### JavaScript 

```js
const requestConfig = {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + RP_API_KEY
  },
  body: JSON.stringify({
    "input": {
      "prompt": "Hello world",
      "yt_url": "https://www.youtube.com/shorts/jcNzoONhrmE",
    }
  })
};
const url = "https://api.runpod.ai/v2/" + RP_ENDPOINT + "/runsync";
const response = await fetch(url, requestConfig);
let data = await response.json();   
data = data.output


// audio data in data.audio_base64
```
