// Log environment variables to help debug
console.log('Environment variables:', {
    hasApiKey: !!process.env.NEXT_PUBLIC_RUNPOD_API_KEY,
    hasVoiceCloneEndpointId: !!process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID,
    hasTtsEndpointId: !!process.env.NEXT_PUBLIC_TTS_ENDPOINT_ID,
    voiceCloneEndpoint: `https://api.runpod.ai/v2/${process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID}/run`,
    ttsEndpoint: `https://api.runpod.ai/v2/${process.env.NEXT_PUBLIC_TTS_ENDPOINT_ID}/run`
});

export const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
export const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
export const TTS_ENDPOINT_ID = process.env.NEXT_PUBLIC_TTS_ENDPOINT_ID || '';

// RunPod API endpoints
export const API_ENDPOINT = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`;
export const TTS_API_ENDPOINT = `https://api.runpod.ai/v2/${TTS_ENDPOINT_ID}/run`;

// FastAPI local server endpoints
export const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';
export const VOICE_LIBRARY_API = `${FASTAPI_BASE_URL}/api`;
export const VOICE_API = `${VOICE_LIBRARY_API}/voices`;
export const TTS_GENERATIONS_API = `${VOICE_LIBRARY_API}/tts/generations`;

export interface ChatterboxRequest {
    prompt: string;
    audio_data: string;
    audio_format: string;
}

export interface ChatterboxResponse {
    status: string;
    audio_base64?: string;
    message?: string;
    metadata?: {
        sample_rate: number;
        audio_shape: number[];
    };
}

// FastAPI response interfaces
export interface VoiceInfo {
    voice_id: string;
    name: string;
    sample_file?: string;
    profile_file?: string;
    created_date?: number;
    has_profile: boolean;
    has_metadata: boolean;
    firebase_url?: string;
}

export interface TTSGeneration {
    file_id: string;
    voice_id: string;
    voice_name: string;
    file_path?: string;
    created_date?: number;
    timestamp: string;
    file_size?: number;
    firebase_url?: string;
}

export interface VoiceLibraryResponse {
    status: string;
    voices: VoiceInfo[];
    total_voices: number;
}

export interface TTSGenerationsResponse {
    status: string;
    total_generations: number;
    generations: TTSGeneration[];
} 