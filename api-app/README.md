## Chatterbox API (FastAPI) — Usage Guide

Base URL (production): `https://runpod-chatterbox.fly.dev`

This API brokers between your main app, Redis queue, RunPod workers, and R2. Some endpoints are public read-only for listing, while job-creating endpoints are internal-only and require HMAC authentication. Client-facing endpoints can additionally enforce Firebase Auth and App Check (feature flags).

### Security overview
- HMAC for internal-only routes (main app → API):
  - Headers required:
    - `X-Daezend-Timestamp`: epoch milliseconds
    - `X-Daezend-Signature`: `hex(HMAC_SHA256(secret, method + "\n" + path + "\n" + timestamp + "\n" + rawBody))`
    - `X-Idempotency-Key`: unique ID per request
  - Env: `DAEZEND_API_SHARED_SECRET` must be identical in the main app and API app.
  - Toggle: `SECURITY_ENABLE_HMAC=true`
- Firebase Auth (browser/mobile): send `Authorization: Bearer <Firebase ID token>`
  - Toggle: `SECURITY_ENABLE_FIREBASE_AUTH=true`
- Firebase App Check (browser/mobile): send `X-Firebase-AppCheck: <token>`
  - Toggle: `SECURITY_ENABLE_APP_CHECK=true`
- CORS allowlist: set `ALLOW_ORIGINS="https://yourapp.com,https://staging.yourapp.com"`

Feature flags let you roll in protections progressively. Start with HMAC on internal routes, then enable Firebase Auth/App Check for client-callable routes.

### Environment variables (API)
- Core:
  - `RUNPOD_API_KEY`, `VC_CB_ENDPOINT_ID`, `TTS_CB_ENDPOINT_ID`
  - `FIREBASE_STORAGE_BUCKET` (e.g., `gs://<project>.firebasestorage.app`)
  - `RUNPOD_SECRET_Firebase` (service account JSON)
- Security:
  - `DAEZEND_API_SHARED_SECRET`
  - `SECURITY_ENABLE_HMAC`, `SECURITY_ENABLE_FIREBASE_AUTH`, `SECURITY_ENABLE_APP_CHECK`
  - `HMAC_MAX_SKEW_SECONDS` (default 300)
  - `IDEMPOTENCY_TTL_SECONDS` (default 86400)
  - `ALLOW_ORIGINS` (CSV list of allowed origins)
- Redis/Queue:
  - `REDIS_URL` (Upstash rediss URL)
  - `REDIS_NAMESPACE` (default `runpod`)
  - `REDIS_STREAM_NAME` (default `runpod:jobs`)
  - `REDIS_CONSUMER_GROUP`, `REDIS_CONSUMER_NAME`
  - `RATE_LIMIT_DEFAULT_PER_MINUTE` (60), `RATE_LIMIT_CLONE_PER_MINUTE` (5), `RATE_LIMIT_TTS_PER_MINUTE` (10)

### HMAC signing example (Node.js)
```js
import crypto from 'node:crypto'

function sign(secret, method, path, timestamp, bodyBuffer) {
  const msg = Buffer.concat([
    Buffer.from(`${method}\n${path}\n${timestamp}\n`, 'utf8'),
    bodyBuffer || Buffer.alloc(0)
  ])
  return crypto.createHmac('sha256', secret).update(msg).digest('hex')
}

// usage
const timestamp = Date.now().toString()
const body = JSON.stringify({ name: 'voice_abc', audio_data: '...', audio_format: 'wav' })
const sig = sign(process.env.DAEZEND_API_SHARED_SECRET, 'POST', '/api/voices/clone', timestamp, Buffer.from(body))
```

### Health
- `GET /api/health/health`
  - Always 200 OK when server is up.
  - Response:
    ```json
    { "status": "healthy", "service": "voice-library-api", "firebase_connected": false, "timestamp": "..." }
    ```

### Voices (client-callable)
- `GET /api/voices`
  - Lists sample voices. Auth/App Check may be required depending on flags.
  - Response: `{ status, voices: VoiceInfo[], language, is_kids_voice, total }`

- `GET /api/voices/by-language/{language}`
  - Lists voices for a specific language.

- `GET /api/voices/{voice_id}/profile`
  - Returns base64 `profile_base64` for a voice profile.

VoiceInfo schema (response shape):
```json
{
  "voice_id": "voice_john",
  "name": "John",
  "sample_file": "https://.../sample.wav",
  "embedding_file": "https://.../profile.npy",
  "created_date": 1712345678,
  "language": "en",
  "is_kids_voice": false
}
```

### Voice cloning (internal-only)
- `POST /api/voices/clone`
  - Security: HMAC required (when `SECURITY_ENABLE_HMAC=true`).
  - Body (JSON):
    ```json
    {
      "name": "voice_myid",
      "audio_data": "<base64>",
      "audio_format": "wav",
      "language": "en",
      "is_kids_voice": false,
      "model_type": "chatterbox"
    }
    ```
  - Returns either `{"status":"queued","metadata":{"job_id":"..."}}` if Redis is configured, or a success payload including Firebase paths when synchronous.
  - Headers (HMAC): `X-Daezend-Timestamp`, `X-Daezend-Signature`, `X-Idempotency-Key`

Example curl (HMAC):
```bash
BODY='{"name":"voice_test","audio_data":"<base64>","audio_format":"wav","language":"en","is_kids_voice":false,"model_type":"chatterbox"}'
TS=$(date +%s%3N)
SIG=$(python - <<'PY'
import hmac,hashlib,os,sys
secret=os.environ['DAEZEND_API_SHARED_SECRET']
method='POST'; path='/api/voices/clone'; ts=os.environ['TS']; body=os.environ['BODY'].encode()
msg=(method+'\n'+path+'\n'+ts+'\n').encode()+body
print(hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest())
PY
)
curl -X POST https://runpod-chatterbox.fly.dev/api/voices/clone \
  -H "Content-Type: application/json" \
  -H "X-Daezend-Timestamp: $TS" \
  -H "X-Daezend-Signature: $SIG" \
  -H "X-Idempotency-Key: test-voice-1" \
  -d "$BODY"
```

### TTS (client-callable listing; internal-only generation)
- `POST /api/tts/generate` (internal-only)
  - Security: HMAC required (when enabled).
  - Body (includes identifiers used by workers to update Firestore):
    ```json
    {
      "user_id": "user_123",
      "story_id": "story_abc",
      "voice_id": "voice_myid",
      "text": "Hello there",
      "profile_base64": "<base64>",
      "language": "en",
      "story_type": "user",
      "is_kids_voice": false,
      "model_type": "chatterbox"
    }
    ```
  - Returns `{"status":"queued","metadata":{"job_id":"..."}}` with Redis or a `success` payload with `audio_path` when synchronous.

- `GET /api/tts/generations`
  - Lists generated stories (flat list) from Firebase.

- `GET /api/tts/stories/{language}`
  - Lists story entries for a language (user/app, default user via query param `story_type`).

- `GET /api/tts/stories/{language}/{story_type}/{file_id}/audio`
  - Returns the public URL for a specific file.

TTSGenerateRequest (body):
```json
{
  "voice_id": "voice_x",
  "text": "...",
  "profile_base64": "...",
  "language": "en",
  "story_type": "user",
  "is_kids_voice": false,
  "model_type": "chatterbox"
}
```

### Queueing model (Redis Streams)
- If `REDIS_URL` is set, job-creating endpoints enqueue and return immediately with `{status:"queued"}` and `job_id`.
- A worker consumes jobs, calls RunPod, uploads outputs to R2 and updates Firestore docs:
  - Clone: writes `voice_profiles/{profile_id}` with status and paths
  - TTS: updates `stories/{story_id}` with `audioStatus`, `audioUrl`, and appends `audioVersions` entry
- Recommended: Use idempotency keys and track job state in Redis hashes under the configured namespace.

### Status codes & errors
- 200: success; 202: queued; 401/403: auth errors; 404: not found; 409: duplicate (idempotency); 429: rate-limited; 500: server errors.
- Error payloads include `detail` with a descriptive message.

### Access from main app (recommended patterns)
1) Internal calls (server-to-server) use HMAC and idempotency:
   - Generate timestamp and signature as described.
   - Retry on 5xx; de-duplicate with the same `X-Idempotency-Key`.
2) Client/browser calls use Firebase Auth + App Check (when enabled):
   - Include `Authorization: Bearer <ID token>` and `X-Firebase-AppCheck`.
   - Respect CORS.

### Deployment & health
- Fly hostname: `https://runpod-chatterbox.fly.dev`
- Health check: `GET /api/health/health`
- Ensure secrets are set via `flyctl secrets set ...` and CORS origins are correct.

---

For any questions about enabling the security toggles or wiring HMAC/Firebase in your main app, see the Security overview above. Enable flags gradually to avoid breaking existing integrations.

# Voice Library API - Modular FastAPI Structure

A production-ready voice cloning and TTS API built with FastAPI, featuring Firebase integration and modular architecture.

## 🏗️ Project Structure

```
api-app/
├── app/
│   ├── main.py               ← FastAPI application initialization
│   ├── api/
│   │   ├── voices.py         ← Voice management endpoints
│   │   ├── tts.py            ← TTS generation endpoints
│   │   └── health.py         ← Health and debug endpoints
│   ├── services/
│   │   ├── firebase.py       ← R2 operations
│   │   └── runpod_client.py  ← RunPod API client
│   ├── models/
│   │   └── schemas.py        ← Pydantic models and schemas
│   └── config.py             ← Application configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── start_fastapi.py
├── test_fastapi.py
└── README.md
```

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Voice Cloning**: Create and manage voice profiles using RunPod
- **TTS Generation**: Generate speech from text using cloned voices
- **Firebase Integration**: Cloud storage for voice files and TTS generations
- **Production Ready**: Docker support, health checks, and comprehensive logging
- **API Documentation**: Auto-generated OpenAPI documentation

## 📋 API Endpoints

### Voice Management
- `GET /api/voices` - List all voices
- `POST /api/voices/clone` - Create voice clone
- `POST /api/voices/save` - Save voice files locally
- `GET /api/voices/{voice_id}/sample` - Get voice sample audio
- `GET /api/voices/{voice_id}/sample/base64` - Get voice sample as base64
- `GET /api/voices/{voice_id}/profile` - Get voice profile

### TTS Generation
- `GET /api/tts/generations` - List TTS generations
- `POST /api/tts/generate` - Generate TTS
- `POST /api/tts/save` - Save TTS generation
- `GET /api/tts/generations/{file_id}/audio` - Get TTS audio file

### System
- `GET /health` - Health check
- `GET /api/debug/directories` - Debug directory status
- `GET /docs` - API documentation

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Docker (optional)
- Firebase credentials (`firebase_creds.json`)

### Local Development

1. **Clone and setup**:
   ```bash
   cd api-app
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   # For voice library display (optional - uses firebase_local_only.json automatically)
   export FIREBASE_STORAGE_BUCKET="your-project-id.firebasestorage.app"
   
   # For RunPod API access (required for voice cloning/TTS)
   export RUNPOD_API_KEY="your-runpod-api-key"
   export VC_CB_ENDPOINT_ID="your-chatterbox-voice-clone-endpoint"
   export TTS_CB_ENDPOINT_ID="your-chatterbox-tts-endpoint"
   ```

3. **Start the server**:
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Optional: Setup Firebase for voice library** (to see existing voices):
   ```bash
   chmod +x setup_local_dev.sh
   ./setup_local_dev.sh
   # Follow the instructions to add Firebase credentials to .env
   ```

**Note**: 
- **Voice cloning/TTS**: Uses RunPod's own `RUNPOD_SECRET_Firebase` secrets (no local setup needed)
- **Voice library display**: Uses `firebase_local_only.json` file automatically to show existing voices

### Docker Deployment

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Or build manually**:
   ```bash
   docker build -t voice-library-api .
   docker run -p 8000:8000 voice-library-api
   ```

## 🧪 Testing

Run the test suite:
```bash
python test_fastapi.py
```

## 📁 Directory Structure

The API manages these directories:
- `voice_profiles/` - Voice profile files (.npy, .json)
- `voice_samples/` - Voice sample audio files
- `temp_voice/` - Temporary voice processing files
- `tts_generated/` - Generated TTS audio files

## 🔧 Configuration

Key configuration options in `app/config.py`:
- `FIREBASE_STORAGE_BUCKET` - R2 bucket name
- `API_HOST` - API server host (default: 0.0.0.0)
- `API_PORT` - API server port (default: 8000)
- `CORS_ORIGINS` - Allowed CORS origins
- `MAX_AUDIO_FILE_SIZE` - Maximum audio file size (50MB)
- `MAX_PROFILE_FILE_SIZE` - Maximum profile file size (10MB)

## 🔐 Environment Variables

Required environment variables:
- `FIREBASE_STORAGE_BUCKET` - R2 bucket
- `RUNPOD_API_KEY` - RunPod API key
- `VC_CB_ENDPOINT_ID` - ChatterboxTTS Voice cloning endpoint ID
- `TTS_CB_ENDPOINT_ID` - ChatterboxTTS TTS generation endpoint ID


Optional:
- `API_HOST` - Server host (default: 0.0.0.0)
- `API_PORT` - Server port (default: 8000)
- `DEBUG` - Debug mode (default: False)
- `LOCAL_STORAGE_ENABLED` - Enable local storage (default: True)
- `FIREBASE_STORAGE_ENABLED` - Enable R2 (default: True)

## 📚 API Documentation

Once running, access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔄 Workflow

### Voice Cloning
1. Upload audio file → `POST /api/voices/clone`
2. RunPod processes the audio
3. Save results → `POST /api/voices/save`
4. Files stored locally and in Firebase

### TTS Generation
1. Select voice and text → `POST /api/tts/generate`
2. RunPod generates TTS
3. Save audio → `POST /api/tts/save`
4. Audio stored locally and in Firebase

## 🚀 Production Deployment

### Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Manual Docker
```bash
docker build -t voice-library-api .
docker run -d \
  -p 8000:8000 \
  -v ./voice_profiles:/app/voice_profiles \
  -v ./voice_samples:/app/voice_samples \
  -v ./firebase_creds.json:/app/firebase_creds.json:ro \
  -e FIREBASE_STORAGE_BUCKET=your-bucket \
  -e RUNPOD_API_KEY=your-key \
  voice-library-api
```

## 🔍 Monitoring

- **Health Check**: `GET /health`
- **Directory Status**: `GET /api/debug/directories`
- **Logs**: Check application logs for detailed information

## 🤝 Contributing

1. Follow the modular structure
2. Add tests for new features
3. Update documentation
4. Use type hints and Pydantic models

## 📄 License

This project is part of the Voice Library API system. 