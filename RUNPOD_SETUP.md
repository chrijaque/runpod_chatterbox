# RunPod Serverless Endpoint Setup Guide

This guide will help you deploy the ChatterboxTTS handlers as serverless endpoints on RunPod.

## Prerequisites

1. **RunPod Account**: Sign up at https://www.runpod.io/
2. **RunPod API Key**: Get your API key from https://www.runpod.io/console/user/settings
3. **GitHub Repository**: Your code should be pushed to GitHub (already configured: `https://github.com/chrijaque/runpod_chatterbox`)

## Step 1: Configure RunPod Secrets

Before deploying, set up your environment variables as RunPod secrets:

1. Go to https://www.runpod.io/console/user/settings
2. Navigate to **Secrets** section
3. Add the following secrets (these will be referenced in your endpoint configuration):

   - `RUNPOD_API_KEY` - Your RunPod API key
   - `MINSTRALY_API_SHARED_SECRET` - Your shared secret for API authentication
   - `APP_BASE_URL` - Your application base URL
   - `R2_ACCESS_KEY_ID` - Cloudflare R2 access key ID
   - `R2_SECRET_ACCESS_KEY` - Cloudflare R2 secret access key
   - `R2_ENDPOINT` - Cloudflare R2 endpoint URL
   - `R2_BUCKET_NAME` - Cloudflare R2 bucket name
   - `NEXT_PUBLIC_R2_PUBLIC_URL` - Public URL for R2 bucket

## Step 2: Deploy TTS Endpoint

### Option A: Using RunPod Web UI

1. **Go to Serverless Endpoints**: https://www.runpod.io/console/serverless
2. **Click "New Endpoint"**
3. **Configure the endpoint**:

   **Basic Settings:**
   - **Name**: `chatterbox-tts`
   - **Template**: Select "Custom" or "Blank Template"
   
   **Container Configuration:**
   - **Container Image**: Leave blank (we'll build from Dockerfile)
   - **Dockerfile Path**: `dockerfiles/chatterbox/Dockerfile.tts`
   - **Build Context**: `.` (root of repository)
   - **Repository**: `https://github.com/chrijaque/runpod_chatterbox`
   - **Branch**: `main`
   
   **GPU Configuration:**
   - **GPU Type**: `RTX 4090`
   - **Container Disk**: `50 GB`
   - **Volume Size**: `100 GB`
   
   **Environment Variables:**
   Add these environment variables (reference your secrets):
   - `RUNPOD_API_KEY` = `{{ RUNPOD_SECRET_RUNPOD_API_KEY }}`
   - `MINSTRALY_API_SHARED_SECRET` = `{{ RUNPOD_SECRET_MINSTRALY_API_SHARED_SECRET }}`
   - `APP_BASE_URL` = `{{ RUNPOD_SECRET_APP_BASE_URL }}`
   - `R2_ACCESS_KEY_ID` = `{{ RUNPOD_SECRET_R2_ACCESS_KEY_ID }}`
   - `R2_SECRET_ACCESS_KEY` = `{{ RUNPOD_SECRET_R2_SECRET_ACCESS_KEY }}`
   - `R2_ENDPOINT` = `{{ RUNPOD_SECRET_R2_ENDPOINT }}`
   - `R2_BUCKET_NAME` = `{{ RUNPOD_SECRET_R2_BUCKET_NAME }}`
   - `NEXT_PUBLIC_R2_PUBLIC_URL` = `{{ RUNPOD_SECRET_NEXT_PUBLIC_R2_PUBLIC_URL }}`
   
   **Ports:**
   - Port `8000` (HTTP)
   
   **Handler:**
   - The handler is already configured in the Dockerfile CMD: `python3 -u tts_handler.py`

4. **Click "Deploy"** and wait for the build to complete

### Option B: Using RunPod CLI (if available)

If you have the RunPod CLI installed:

```bash
# Install RunPod CLI (if not already installed)
# Check: https://docs.runpod.io/runpodctl/overview

# Deploy using TOML configuration
runpodctl serverless deploy --config runpod.chatterbox.tts.toml
```

## Step 3: Deploy Voice Cloning (VC) Endpoint

Repeat Step 2 with these changes:

**Configuration:**
- **Name**: `chatterbox-vc`
- **Dockerfile Path**: `dockerfiles/chatterbox/Dockerfile.vc`
- Use the same GPU, disk, and environment variable settings

Or use the TOML file:
```bash
runpodctl serverless deploy --config runpod.chatterbox.vc.toml
```

## Step 4: Deploy LLM Endpoint (Optional)

If you need the LLM endpoint:

**Configuration:**
- **Name**: `chatterbox-llm`
- **Dockerfile Path**: `dockerfiles/chatterbox/Dockerfile.llm`
- **GPU Type**: `RTX 3090` (smaller GPU, sufficient for LLM)
- **Container Disk**: `30 GB`
- **Volume Size**: `50 GB`

Or use the TOML file:
```bash
runpodctl serverless deploy --config runpod.chatterbox.llm.toml
```

## Step 5: Get Endpoint IDs

After deployment:

1. Go to https://www.runpod.io/console/serverless
2. Find your endpoints (`chatterbox-tts`, `chatterbox-vc`, `chatterbox-llm`)
3. Copy the **Endpoint ID** for each (you'll need these for your API app)

## Step 6: Update Your API App Configuration

Update your `api-app/.env` or environment variables with the endpoint IDs:

```bash
export TTS_CB_ENDPOINT_ID="your-tts-endpoint-id"
export VC_CB_ENDPOINT_ID="your-vc-endpoint-id"
export LLM_CB_ENDPOINT_ID="your-llm-endpoint-id"  # if using LLM
```

## Troubleshooting

### Build Fails Immediately
- Check that the Dockerfile path is correct: `dockerfiles/chatterbox/Dockerfile.tts`
- Verify the build context is set to `.` (repository root)
- Ensure all required files exist:
  - `requirements/chatterbox.txt`
  - `handlers/chatterbox/tts_handler.py`

### Build Fails During Docker Build
- Check the build logs in RunPod console
- Verify the GitHub repository is accessible
- Ensure the branch name is correct (`main`)

### Endpoint Not Responding
- Check that the handler file exists and is executable
- Verify port 8000 is exposed
- Check endpoint logs in RunPod console
- Ensure environment variables are correctly set

### Environment Variables Not Working
- Verify secrets are created in RunPod console
- Check that secret names match exactly (case-sensitive)
- Ensure you're using the `{{ RUNPOD_SECRET_* }}` syntax correctly

## Testing Your Endpoints

Once deployed, test your endpoints:

```bash
# Get your endpoint ID from RunPod console
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

# Test TTS endpoint
curl -X POST \
  https://api.runpod.ai/v2/${ENDPOINT_ID}/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -d '{
    "input": {
      "text": "Hello, this is a test",
      "voice_id": "test-voice",
      "profile_base64": "base64-encoded-profile"
    }
  }'
```

## Cost Optimization

- **GPU Selection**: Use RTX 3090 for LLM (cheaper), RTX 4090 for TTS/VC (faster)
- **Container Disk**: Adjust based on model sizes (50GB is usually sufficient)
- **Volume Size**: Use network volumes for persistent storage (100GB recommended)
- **Idle Timeout**: Configure appropriate idle timeout to reduce costs

## Next Steps

1. Monitor endpoint usage in RunPod console
2. Set up auto-scaling if needed
3. Configure webhooks/callbacks for async jobs
4. Set up monitoring and alerts

## Additional Resources

- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/get-started)
- [RunPod Endpoint Configuration](https://docs.runpod.io/serverless/endpoints/endpoint-configurations)
- [RunPod CLI Documentation](https://docs.runpod.io/runpodctl/overview)
