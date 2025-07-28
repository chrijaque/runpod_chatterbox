# 🏗️ Organized Storage Structure: Language-Based Voice Management

## **Overview**

This document describes the new organized storage structure for the voice AI API that supports **language-based organization** and **kids voice separation**, similar to ElevenLabs but with enhanced organization.

## **📁 Firebase Storage Structure**

```
gs://your-project-id.firebasestorage.app/audio/
├── voices/                      # Voice management
│   ├── en/                      # English voices
│   │   ├── recorded/            # User raw recordings
│   │   │   ├── voice_john_doe_recording_1_20250725_163000.wav
│   │   │   └── voice_jane_smith_recording_1_20250725_165000.wav
│   │   ├── samples/             # Generated previews
│   │   │   ├── voice_john_doe_voice_john_doe_sample_20250725_163000.wav
│   │   │   └── voice_jane_smith_voice_jane_smith_sample_20250725_165000.wav
│   │   ├── profiles/            # Voice profiles (.npy files)
│   │   │   ├── voice_john_doe_voice_john_doe_20250725_163000.npy
│   │   │   └── voice_jane_smith_voice_jane_smith_20250725_165000.npy
│   │   └── kids/                # Kids voices (English)
│   │       ├── recorded/        # Kids' raw recordings
│   │       ├── samples/         # Kids' generated previews
│   │       └── profiles/        # Kids' voice profiles
│   ├── es/                      # Spanish voices
│   │   ├── recorded/
│   │   ├── samples/
│   │   ├── profiles/
│   │   └── kids/
│   │       ├── recorded/
│   │       ├── samples/
│   │       └── profiles/
│   ├── fr/                      # French voices
│   │   ├── recorded/
│   │   ├── samples/
│   │   ├── profiles/
│   │   └── kids/
│   │       ├── recorded/
│   │       ├── samples/
│   │       └── profiles/
│   └── [other_languages]/
│       ├── recorded/
│       ├── samples/
│       ├── profiles/
│       └── kids/
│           ├── recorded/
│           ├── samples/
│           └── profiles/
└── stories/                     # TTS story generation
    ├── en/                      # English stories
    │   ├── user/                # User-generated stories
    │   │   ├── TTS_voice_john_doe_20250725_163000.wav
    │   │   └── TTS_voice_jane_smith_20250725_165000.wav
    │   └── app/                 # App-generated stories
    │       ├── TTS_voice_john_doe_20250725_163000.wav
    │       └── TTS_voice_jane_smith_20250725_165000.wav
    ├── es/                      # Spanish stories
    │   ├── user/
    │   └── app/
    ├── fr/                      # French stories
    │   ├── user/
    │   └── app/
    └── [other_languages]/
        ├── user/
        └── app/
```

## **🔄 Voice Clone Request Format**

### **Main App Request**

```json
{
  "title": "John's Professional Voice",
  "voices": [
    "UklGRiQAAABXQVZFZm10...",  // Base64 encoded WAV audio blob
    "UklGRiQAAABXQVZFZm10..."   // Additional recordings (optional)
  ],
  "visibility": "private",
  "metadata": {
    "language": "en",
    "isKidsVoice": false,
    "userId": "user_123",
    "createdAt": "2024-01-01T00:00:00Z"
  }
}
```

### **API Response**

```json
{
  "status": "success",
  "voice_id": "voice_johns_professional_voice_20250725_163000",
  "voice_name": "John's Professional Voice",
  "audio_base64": "UklGRiQAAABXQVZFZm10...",
  "profile_base64": "npy_data_here...",
  "metadata": {
    "firebase_urls": {
      "recorded": [
        "https://storage.googleapis.com/bucket/audio/voices/en/recorded/voice_johns_professional_voice_20250725_163000_recording_1_20250725_163000.wav"
      ],
      "samples": [
        "https://storage.googleapis.com/bucket/audio/voices/en/samples/voice_johns_professional_voice_20250725_163000_voice_johns_professional_voice_20250725_163000_sample_20250725_163000.wav"
      ],
      "profiles": [
        "https://storage.googleapis.com/bucket/audio/voices/en/profiles/voice_johns_professional_voice_20250725_163000_voice_johns_professional_voice_20250725_163000.npy"
      ]
    },
    "shared_access": true,
    "uploaded_at": "2025-07-25T16:30:00Z",
    "language": "en",
    "is_kids_voice": false,
    "user_id": "user_123",
    "visibility": "private",
    "title": "John's Professional Voice",
    "recordings_count": 1
  }
}
```

## **🔗 API Endpoints**

### **Voice Management**

```bash
# Create voice clone with organized storage
POST /api/voices/clone
Content-Type: application/json

# List available languages
GET /api/voices/languages

# List voices by language and type
GET /api/voices/by-language/{language}?is_kids_voice=false
GET /api/voices/by-language/{language}?is_kids_voice=true

# Get voice files with language context
GET /api/voices/{voice_id}/firebase-urls?language=en&is_kids_voice=false
GET /api/voices/{voice_id}/sample/firebase?language=en&is_kids_voice=false
```

### **TTS Generation**

```bash
# Generate TTS story with organized storage
POST /api/tts/generate
Content-Type: application/x-www-form-urlencoded

voice_id=voice_john_doe&
text=Hello, this is a test story&
language=en&
story_type=user&
is_kids_voice=false

# Generate voice clone sample
POST /api/tts/generate
Content-Type: application/x-www-form-urlencoded

voice_id=voice_john_doe&
text=Hello, this is a sample&
language=en&
story_type=sample&
is_kids_voice=false
```

## **🌍 Supported Languages**

The API supports the following languages:

- **en** - English
- **es** - Spanish  
- **fr** - French
- **de** - German
- **it** - Italian
- **pt** - Portuguese
- **ru** - Russian
- **ja** - Japanese
- **ko** - Korean
- **zh** - Chinese

## **👶 Kids Voice Support**

### **Kids Voice Structure**

```
audio/voices/en/kids/
├── recorded/      # Kids' raw recordings
├── samples/       # Kids' generated previews
└── profiles/      # Kids' voice profiles
```

### **Kids Voice Usage**

```json
{
  "title": "Little Emma's Voice",
  "voices": ["base64_audio_data"],
  "visibility": "private",
  "metadata": {
    "language": "en",
    "isKidsVoice": true,        // Mark as kids voice
    "userId": "user_123",
    "createdAt": "2024-01-01T00:00:00Z"
  }
}
```

## **📊 File Naming Convention**

### **Voice Files**

```
{voice_id}_{recording_type}_{timestamp}.{extension}
```

Examples:
- `voice_john_doe_recording_1_20250725_163000.wav`
- `voice_john_doe_voice_john_doe_sample_20250725_163000.wav`
- `voice_john_doe_voice_john_doe_20250725_163000.npy`

### **TTS Files**

```
TTS_{voice_id}_{timestamp}.wav
```

Examples:
- `TTS_voice_john_doe_20250725_163000.wav` (story generation)
- `voice_john_doe_voice_john_doe_sample_20250725_163000.wav` (voice sample)

## **🚀 Usage Examples**

### **Create Voice Clone**

```javascript
// Create a professional English voice
const response = await fetch('/api/voices/clone', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    title: "John's Professional Voice",
    voices: ["base64_audio_data"],
    visibility: "private",
    metadata: {
      language: "en",
      isKidsVoice: false,
      userId: "user_123",
      createdAt: "2024-01-01T00:00:00Z"
    }
  })
});

const result = await response.json();
console.log('Voice created:', result.voice_id);
console.log('Firebase URLs:', result.metadata.firebase_urls);
```

### **List Voices by Language**

```javascript
// List English professional voices
const englishVoices = await fetch('/api/voices/by-language/en?is_kids_voice=false');
const englishData = await englishVoices.json();

// List English kids voices
const kidsVoices = await fetch('/api/voices/by-language/en?is_kids_voice=true');
const kidsData = await kidsVoices.json();

console.log('English voices:', englishData.total_voices);
console.log('Kids voices:', kidsData.total_voices);
```

### **Get Voice Files**

```javascript
// Get all files for a specific voice
const voiceFiles = await fetch('/api/voices/voice_john_doe/firebase-urls?language=en&is_kids_voice=false');
const files = await voiceFiles.json();

console.log('Recorded files:', files.firebase_urls.recorded);
console.log('Sample files:', files.firebase_urls.samples);
console.log('Profile files:', files.firebase_urls.profiles);
```

### **Generate TTS**

```javascript
// Generate TTS story with language context
const formData = new FormData();
formData.append('voice_id', 'voice_john_doe');
formData.append('text', 'Hello, this is a test story');
formData.append('language', 'en');
formData.append('story_type', 'user');  // 'user' or 'app'
formData.append('is_kids_voice', 'false');

const ttsResponse = await fetch('/api/tts/generate', {
  method: 'POST',
  body: formData
});

const ttsResult = await ttsResponse.json();
console.log('TTS Path:', ttsResult.audio_path);

// Generate voice clone sample
const sampleFormData = new FormData();
sampleFormData.append('voice_id', 'voice_john_doe');
sampleFormData.append('text', 'Hello, this is a sample');
sampleFormData.append('language', 'en');
sampleFormData.append('story_type', 'sample');  // 'sample' for voice cloning
sampleFormData.append('is_kids_voice', 'false');

const sampleResponse = await fetch('/api/tts/generate', {
  method: 'POST',
  body: sampleFormData
});

const sampleResult = await sampleResponse.json();
console.log('Sample Path:', sampleResult.audio_path);
```

## **🔧 Implementation Details**

### **Firebase Service Methods**

```python
# Upload with language and kids voice context
firebase_service.upload_runpod_voice_sample(voice_id, filename, language, is_kids_voice)
firebase_service.upload_runpod_tts_story(voice_id, filename, language, story_type, is_kids_voice)
firebase_service.upload_voice_profile(voice_id, filename, language, is_kids_voice)
firebase_service.upload_user_recording(voice_id, filename, language, is_kids_voice)

# List files with context
firebase_service.list_voice_files(voice_id, language, is_kids_voice)
firebase_service.list_voices_by_language(language, is_kids_voice)
firebase_service.list_stories_by_language(language, story_type)
```

### **Storage Path Generation**

```python
# Voice sample paths
f"audio/voices/{language}/samples/{voice_id}_{voice_id}_sample_{timestamp}.wav"
f"audio/voices/{language}/kids/samples/{voice_id}_{voice_id}_sample_{timestamp}.wav"

# TTS story paths
f"audio/stories/{language}/{story_type}/TTS_{voice_id}_{timestamp}.wav"
```

## **📈 Benefits**

### **✅ Organization Benefits**

1. **Language Separation**: Easy to manage voices and stories by language
2. **Kids Voice Isolation**: Separate kids voices for safety and organization
3. **Story Type Separation**: User-generated vs app-generated stories clearly separated
4. **Scalable Structure**: Easy to add new languages
5. **Clear File Types**: Recorded, samples, profiles, and stories clearly separated
6. **User Context**: Each voice and story tied to specific user and metadata

### **🔄 Workflow Benefits**

1. **Automatic Organization**: Files automatically organized by language and type
2. **Easy Discovery**: Simple to find voices by language and type
3. **Shared Access**: Both apps can access the same organized structure
4. **Metadata Preservation**: All voice metadata preserved and accessible

## **🎯 Next Steps**

1. **Test the new structure** with your main app
2. **Configure language-specific settings** if needed
3. **Add more languages** as required
4. **Implement voice search** by language and type
5. **Add voice analytics** per language and type

This organized storage structure provides a **scalable, maintainable, and user-friendly** way to manage voices across multiple languages and types, perfect for a production voice AI service. 