# TTS Error Callback System

## Overview

The TTS Error Callback System provides end-to-end error handling for TTS (Text-to-Speech) generation failures. When TTS generation fails in the RunPod service, it automatically notifies the main application, which updates the story status in Firestore and triggers real-time frontend notifications.

## Architecture

```
┌─────────────────┐    Error Callback    ┌─────────────────┐    Firestore Update    ┌─────────────────┐
│   RunPod TTS    │ ────────────────────► │   Main API App  │ ────────────────────► │   Frontend      │
│   Handler       │                      │                 │                      │   (Real-time)    │
└─────────────────┘                      └─────────────────┘                      └─────────────────┘
        │                                         │                                         │
        │ TTS Generation Fails                   │ Story Status Updated                   │ User Sees Error
        │                                        │ audioStatus: 'failed'                  │ + Retry Button
        │                                        │ audioError: errorMessage               │
        │                                        │ audioErrorDetails: errorDetails       │
```

## Components

### 1. TTS Handler (RunPod Service)

**File**: `handlers/chatterbox/tts_handler.py`

**Key Functions**:
- `notify_error_callback()`: Sends error notifications to the main app
- Error detection in `call_tts_model_generate_tts_story()`
- Error handling in main `handler()` function

**Error Scenarios Handled**:
- TTS model not available
- Voice profile missing or corrupted
- Text processing errors
- Model generation failures
- Unexpected exceptions

### 2. Main API App

**File**: `api-app/app/api/tts.py`

**Endpoint**: `POST /api/tts/error-callback`

**Functionality**:
- Receives error callbacks from RunPod service
- Updates story status in Firestore
- Returns success/failure response

### 3. Data Models

**File**: `api-app/app/models/schemas.py`

**Schemas**:
- `TTSErrorCallbackRequest`: Error callback payload structure
- `TTSErrorCallbackResponse`: Response structure

## API Specification

### Error Callback Endpoint

**URL**: `POST /api/tts/error-callback`

**Request Payload**:
```json
{
  "story_id": "story123",
  "error": "Voice profile not found",
  "error_details": "The voice profile file was not found in storage",
  "user_id": "user456",
  "voice_id": "voice789",
  "job_id": "job_abc123",
  "metadata": {
    "language": "en",
    "story_type": "user",
    "text_length": 1500,
    "error_type": "voice_profile_missing"
  }
}
```

**Required Fields**:
- `story_id`: The ID of the story that failed audio generation
- `error`: Human-readable error message

**Optional Fields**:
- `user_id`: User ID for logging/debugging
- `voice_id`: Voice ID that was being used
- `error_details`: Technical error details for debugging
- `job_id`: Job ID from the TTS service for tracking
- `metadata`: Additional context about the failed generation

**Response**:
```json
{
  "success": true
}
```

Or on error:
```json
{
  "success": false,
  "error": "Internal server error"
}
```

## Error Callback URL Construction

The TTS handler automatically constructs the error callback URL from the success callback URL:

```python
# Handle different possible callback URL formats
if "/api/tts/callback" in callback_url:
    error_callback_url = callback_url.replace("/api/tts/callback", "/api/tts/error-callback")
elif "/api/tts/" in callback_url:
    error_callback_url = callback_url.rsplit("/", 1)[0] + "/error-callback"
else:
    base_url = callback_url.rstrip("/")
    error_callback_url = f"{base_url}/error-callback"
```

**Examples**:
- `https://api.example.com/api/tts/callback` → `https://api.example.com/api/tts/error-callback`
- `https://api.example.com/api/tts/generate` → `https://api.example.com/api/tts/error-callback`
- `https://api.example.com/callback` → `https://api.example.com/error-callback`

## Firestore Updates

When an error callback is received, the story document is updated with:

```javascript
{
  audioStatus: 'failed',
  audioError: errorMessage,
  audioErrorDetails: errorDetails,
  audioJobId: jobId,
  audioErrorMetadata: metadata,
  updatedAt: new Date()
}
```

## Frontend Behavior

Once the error callback is received and processed:

1. **Real-time Update**: Firestore listener immediately updates the story state
2. **UI Changes**:
   - Pending spinner disappears
   - Error message appears with retry button
   - Audio panel shows failed status
3. **User Action**: User can click "Try Again" to retry generation

## Error Scenarios

The TTS service calls the error callback for:

### 1. Voice Profile Issues
- Missing voice profile files
- Corrupted voice profile data
- Invalid voice profile format

### 2. Text Processing Errors
- Invalid text encoding
- Text too long or too short
- Unsupported language

### 3. Model Errors
- TTS model failures
- GPU memory issues
- Model loading failures

### 4. Storage Errors
- Failed to upload generated audio
- R2 issues
- Network connectivity problems

### 5. Resource Errors
- Insufficient memory
- Disk space issues
- CPU/GPU overload

### 6. Authentication Errors
- Invalid API keys
- Permission issues
- Service account problems

## Implementation Details

### TTS Handler Error Detection

```python
def call_tts_model_generate_tts_story(text, voice_id, profile_base64, language, story_type, is_kids_voice, api_metadata):
    # Check if TTS model is available
    if tts_model is None:
        error_msg = "TTS model not available"
        return {"status": "error", "message": error_msg}
    
    # Try TTS generation
    try:
        result = tts_model.generate_tts_story(...)
        return result
    except Exception as method_error:
        error_msg = f"generate_tts_story method failed: {method_error}"
        return {"status": "error", "message": error_msg}
```

### Error Callback Function

```python
def notify_error_callback(error_callback_url: str, story_id: str, error_message: str, **kwargs):
    payload = {
        "story_id": story_id,
        "error": error_message,
        "error_details": kwargs.get("error_details"),
        "user_id": kwargs.get("user_id"),
        "voice_id": kwargs.get("voice_id"),
        "job_id": kwargs.get("job_id"),
        "metadata": kwargs.get("metadata", {})
    }
    
    response = requests.post(
        error_callback_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=10
    )
    response.raise_for_status()
```

### Main App Error Callback Handler

```python
@router.post("/error-callback", response_model=TTSErrorCallbackResponse)
async def tts_error_callback(request: TTSErrorCallbackRequest):
    # Update story status in Firestore
    story_ref = db.collection('stories').document(request.story_id)
    
    update_data = {
        'audioStatus': 'failed',
        'audioError': request.error,
        'audioErrorDetails': request.error_details,
        'audioJobId': request.job_id,
        'updatedAt': firestore.SERVER_TIMESTAMP
    }
    
    if request.metadata:
        update_data['audioErrorMetadata'] = request.metadata
    
    story_ref.update(update_data)
    return TTSErrorCallbackResponse(success=True)
```

## Testing

### Test Script

Run the test script to verify the error callback system:

```bash
python test_error_callback.py
```

The test script includes:
- Error callback endpoint testing
- Payload validation testing
- URL construction testing

### Manual Testing

1. **Start the API server**:
   ```bash
   cd api-app
   python -m uvicorn app.main:app --reload --port 8000
   ```

2. **Test error callback endpoint**:
   ```bash
   curl -X POST http://localhost:8000/api/tts/error-callback \
     -H "Content-Type: application/json" \
     -d '{
       "story_id": "test_story_123",
       "error": "Voice profile not found",
       "user_id": "test_user_456"
     }'
   ```

3. **Expected response**:
   ```json
   {
     "success": true
   }
   ```

## Error Handling Best Practices

### 1. Graceful Degradation
- Error callbacks should not block the main TTS process
- If error callback fails, log the error but don't fail the entire operation

### 2. Comprehensive Logging
- Log all error scenarios with detailed context
- Include error types, stack traces, and metadata

### 3. User-Friendly Messages
- Provide human-readable error messages
- Include actionable information when possible

### 4. Retry Mechanisms
- Implement retry logic for transient errors
- Use exponential backoff for network issues

### 5. Monitoring and Alerting
- Monitor error callback success rates
- Set up alerts for critical error patterns

## Troubleshooting

### Common Issues

1. **Error callback not received**:
   - Check network connectivity between RunPod and main app
   - Verify callback URL construction
   - Check firewall settings

2. **Firestore update fails**:
   - Verify Firebase credentials
   - Check Firestore permissions
   - Ensure story document exists

3. **Frontend not updating**:
   - Verify Firestore listener is active
   - Check real-time connection
   - Validate story ID format

### Debug Steps

1. **Check TTS handler logs**:
   ```bash
   # Look for error callback attempts
   grep "error callback" /path/to/tts/handler/logs
   ```

2. **Check main app logs**:
   ```bash
   # Look for error callback endpoint calls
   grep "error-callback" /path/to/api/logs
   ```

3. **Verify Firestore updates**:
   ```bash
   # Check if story document was updated
   firebase firestore:get stories/story_id
   ```

## Future Enhancements

### 1. Error Classification
- Categorize errors by type (user error, system error, etc.)
- Implement different handling strategies per error type

### 2. Error Recovery
- Automatic retry for transient errors
- Fallback to different TTS models

### 3. Error Analytics
- Track error patterns and frequencies
- Generate error reports and insights

### 4. User Notifications
- Email notifications for critical errors
- In-app error history and details

### 5. Error Prevention
- Proactive validation of inputs
- Pre-flight checks before TTS generation
