// FastAPI local server endpoints (proxy for RunPod calls)
export const FASTAPI_BASE_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || 'http://localhost:8000';
export const VOICE_LIBRARY_API = `${FASTAPI_BASE_URL}/api`;
export const VOICE_API = `${VOICE_LIBRARY_API}/voices`;
export const TTS_API = `${VOICE_LIBRARY_API}/tts`;
export const TTS_GENERATIONS_API = `${VOICE_LIBRARY_API}/tts/generations`;

// RunPod API endpoints (now routed through FastAPI backend)
export const API_ENDPOINT = `${VOICE_API}/clone`;
export const TTS_API_ENDPOINT = `${TTS_API}/generate`;

// Environment variables for debugging
console.log('API Configuration:', {
    fastapiUrl: FASTAPI_BASE_URL,
    voiceApi: VOICE_API,
    ttsApi: TTS_API,
    voiceCloneEndpoint: API_ENDPOINT,
    ttsEndpoint: TTS_API_ENDPOINT
});

export const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || '';
export const VC_CB_ENDPOINT_ID = process.env.VC_CB_ENDPOINT_ID || '';
export const TTS_CB_ENDPOINT_ID = process.env.TTS_CB_ENDPOINT_ID || '';

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