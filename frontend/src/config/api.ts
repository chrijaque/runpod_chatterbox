// Log environment variables to help debug
console.log('Environment variables:', {
    hasApiKey: !!process.env.NEXT_PUBLIC_RUNPOD_API_KEY,
    hasEndpointId: !!process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID,
    apiEndpoint: `https://api.runpod.ai/v2/${process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID}/run`
});

export const RUNPOD_API_KEY = process.env.NEXT_PUBLIC_RUNPOD_API_KEY || '';
export const RUNPOD_ENDPOINT_ID = process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID || '';
export const API_ENDPOINT = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run`;

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