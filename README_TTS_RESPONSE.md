# TTS Response Format

## Overview
The TTS handler returns different response formats depending on the size of the generated audio.

## Response Types

### 1. Small Audio Files (< 10MB)
```json
{
  "status": "success",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA...",
  "metadata": {
    "voice_id": "voice_chrisrepo1",
    "voice_name": "chrisrepo1",
    "text_input": "It was a sunny afternoon in Houndville...",
    "generation_time": 407.47,
    "sample_rate": 22050,
    "audio_shape": [1, 11520000],
    "tts_file": "/tts_generated/tts_voice_chrisrepo1_20250724_144927.wav",
    "timestamp": "20250724_144927",
    "response_type": "audio_data"
  }
}
```

### 2. Large Audio Files (≥ 10MB)
```json
{
  "status": "success",
  "audio_base64": null,
  "file_path": "/tts_generated/tts_voice_chrisrepo1_20250724_144927.wav",
  "file_size_mb": 31.9,
  "metadata": {
    "voice_id": "voice_chrisrepo1",
    "voice_name": "chrisrepo1",
    "text_input": "It was a sunny afternoon in Houndville...",
    "generation_time": 407.47,
    "sample_rate": 22050,
    "audio_shape": [1, 11520000],
    "tts_file": "/tts_generated/tts_voice_chrisrepo1_20250724_144927.wav",
    "timestamp": "20250724_144927",
    "response_type": "file_path_only"
  }
}
```

## Client Handling

### For Small Files
```javascript
if (response.audio_base64) {
  // Convert base64 to audio blob
  const audioBlob = base64ToBlob(response.audio_base64, 'audio/wav');
  const audioUrl = URL.createObjectURL(audioBlob);
  
  // Play or download the audio
  const audio = new Audio(audioUrl);
  audio.play();
}
```

### For Large Files - Method 1: Direct File Download
```javascript
if (response.audio_base64 === null && response.file_path) {
  console.log(`Large audio file generated: ${response.file_path}`);
  console.log(`File size: ${response.file_size_mb} MB`);
  
  // Download the file using the TTS handler
  const downloadResponse = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      action: 'download_file',
      file_path: response.file_path
    })
  });
  
  const downloadResult = await downloadResponse.json();
  
  if (downloadResult.status === 'success') {
    // Convert base64 to audio blob
    const audioBlob = base64ToBlob(downloadResult.audio_base64, 'audio/wav');
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Play or download the audio
    const audio = new Audio(audioUrl);
    audio.play();
  } else {
    console.error('Download failed:', downloadResult.message);
  }
}
```

### For Large Files - Method 2: List Available Files
```javascript
// List all available TTS files
const listResponse = await fetch('/api/tts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    action: 'list_files'
  })
});

const listResult = await listResponse.json();

if (listResult.status === 'success') {
  console.log('Available TTS files:');
  listResult.files.forEach(file => {
    console.log(`- ${file.name}: ${file.size_mb} MB`);
  });
} else {
  console.error('Failed to list files:', listResult.message);
}
```

## File Access Options

### Option 1: RunPod File Download
If RunPod provides file download endpoints:
```javascript
const downloadUrl = `https://api.runpod.ai/v2/${endpointId}/file/${response.file_path}`;
```

### Option 2: Container File System
The file is saved in the container at the specified path and can be accessed through RunPod's file system.

### Option 3: Custom Download Endpoint
You may need to implement a custom endpoint to serve the generated files.

## Error Response
```json
{
  "status": "error",
  "message": "Failed to generate TTS: [error details]"
}
```

## Notes
- Files are saved in `/tts_generated/` directory in the container
- Large files (>10MB) avoid RunPod's response size limits
- Text input is truncated to 500 characters in metadata to reduce response size
- All processing statistics are included in metadata 