'use client';

import { useEffect, useState, useRef } from 'react';
import { API_ENDPOINT, RUNPOD_API_KEY } from '@/config/api';
import Link from 'next/link';

interface Voice {
    voice_id: string;
    name: string;
    sample_file: string;
    embedding_file: string;
    created_date: number;
}

interface TTSResult {
    audio_base64: string;
    metadata: {
        voice_id: string;
        voice_name: string;
        text_input: string;
        generation_time: number;
    };
}

export default function TTSPage() {
    const [text, setText] = useState('');
    const [selectedVoice, setSelectedVoice] = useState<string>('');
    const [voiceLibrary, setVoiceLibrary] = useState<Voice[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingLibrary, setIsLoadingLibrary] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<TTSResult | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Load voice library on component mount
    useEffect(() => {
        loadVoiceLibrary();
    }, []);

    const loadVoiceLibrary = async () => {
        setIsLoadingLibrary(true);
        try {
            const response = await fetch('http://localhost:5001/api/voices', {
                method: 'GET',
            });

            const data = await response.json();
            console.log('TTS Library API response:', data);

            if (data.status === 'success') {
                setVoiceLibrary(data.voices || []);
                // Auto-select first voice if available
                if (data.voices && data.voices.length > 0 && !selectedVoice) {
                    setSelectedVoice(data.voices[0].voice_id);
                }
            } else {
                throw new Error(data.message || 'Failed to load voice library');
            }
        } catch (err) {
            console.error('Error loading voice library:', err);
            setError('Failed to load voice library');
        } finally {
            setIsLoadingLibrary(false);
        }
    };

    const pollJobStatus = async (jobId: string) => {
        const statusEndpoint = API_ENDPOINT.replace('/run', `/status/${jobId}`);
        abortControllerRef.current = new AbortController();
        
        try {
            while (true) {
                if (abortControllerRef.current.signal.aborted) {
                    throw new Error('Polling aborted');
                }

                const response = await fetch(statusEndpoint, {
                    headers: {
                        'Authorization': `Bearer ${RUNPOD_API_KEY}`
                    },
                    signal: abortControllerRef.current.signal
                });
                
                const data = await response.json();
                console.log('TTS Status check response:', data);
                
                if (data.status === 'COMPLETED') {
                    console.log('TTS Job completed, checking output:', data.output);
                    
                    if (data.output?.status === 'error') {
                        throw new Error(data.output.message);
                    }
                    if (data.output?.audio_base64) {
                        console.log('TTS Found audio_base64 in output, returning output object');
                        return data.output;
                    }
                    console.log('TTS No audio_base64 found, returning raw output');
                    return data.output;
                } else if (data.status === 'FAILED') {
                    throw new Error(data.error || 'Job failed');
                } else if (data.status === 'IN_QUEUE' || data.status === 'IN_PROGRESS') {
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    continue;
                } else {
                    throw new Error(`Unexpected status: ${data.status}`);
                }
            }
        } finally {
            abortControllerRef.current = null;
        }
    };

    const handleSubmit = async () => {
        try {
            setIsLoading(true);
            setError(null);
            setResult(null);

            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            abortControllerRef.current = new AbortController();

            if (!RUNPOD_API_KEY) {
                throw new Error('RunPod API key not configured');
            }

            if (!text.trim()) {
                throw new Error('Please enter text to synthesize');
            }

            if (!selectedVoice) {
                throw new Error('Please select a voice from the library');
            }

            console.log('🚀 Starting TTS generation...', { 
                text: text.substring(0, 50) + '...', 
                selectedVoice 
            });

            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`
                },
                body: JSON.stringify({
                    input: {
                        request_type: 'tts',
                        text: text,
                        voice_id: selectedVoice,
                        responseFormat: "base64",
                    },
                }),
                signal: abortControllerRef.current.signal,
            });

            const data = await response.json();
            console.log('📨 TTS RunPod API response received:', {
                hasId: !!data.id,
                hasOutput: !!data.output,
                hasError: !!data.error,
                status: data.status,
                keys: Object.keys(data)
            });

            if (!response.ok) {
                throw new Error(data.message || 'Failed to generate TTS');
            }

            if (data.id) {
                setCurrentJobId(data.id);
                console.log('⏳ TTS Job queued, polling for results...', { jobId: data.id });
                
                const result = await pollJobStatus(data.id);
                
                console.log('🏁 TTS Final result received:', {
                    hasResult: !!result,
                    resultType: typeof result,
                    hasAudioBase64: !!(result && result.audio_base64),
                    hasMetadata: !!(result && result.metadata),
                    status: result?.status
                });
                
                setResult(result);
                
            } else {
                throw new Error('No job ID in response');
            }
        } catch (err: unknown) {
            console.error('TTS API error:', err);
            if (err instanceof Error && err.name === 'AbortError') {
                setError('Operation cancelled');
            } else {
                setError(err instanceof Error ? err.message : 'An error occurred');
            }
        } finally {
            setIsLoading(false);
            setCurrentJobId(null);
        }
    };

    const stopJob = async () => {
        if (!currentJobId) return;

        try {
            const cancelEndpoint = API_ENDPOINT.replace('/run', `/cancel/${currentJobId}`);
            const response = await fetch(cancelEndpoint, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`
                }
            });

            const data = await response.json();
            console.log('TTS Cancel response:', data);

            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }

            setIsLoading(false);
            setCurrentJobId(null);
            setError('Job cancelled by user');
        } catch (err) {
            console.error('Error cancelling TTS job:', err);
            setError('Failed to cancel job');
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    return (
        <main className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
            <div className="max-w-2xl mx-auto space-y-6">
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-900">
                        Text-to-Speech Generator
                    </h1>
                    <p className="mt-2 text-sm text-gray-600">
                        Generate speech using your saved voice clones. Enter text and select a voice from your library.
                    </p>
                    <Link href="/" className="text-blue-600 hover:text-blue-800 text-sm">
                        ← Back to Voice Cloning
                    </Link>
                </div>

                <div className="card">
                    <div className="space-y-6">
                        <div>
                            <label htmlFor="text" className="form-label">
                                Text to Synthesize
                            </label>
                            <textarea
                                id="text"
                                value={text}
                                onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
                                className="form-input min-h-[120px]"
                                placeholder="Enter the text you want to convert to speech..."
                                rows={5}
                            />
                            <p className="mt-1 text-sm text-gray-500">
                                {text.length} characters
                            </p>
                        </div>

                        <div>
                            <label htmlFor="voice" className="form-label">
                                Select Voice
                            </label>
                            {isLoadingLibrary ? (
                                <div className="form-input bg-gray-100">Loading voices...</div>
                            ) : voiceLibrary.length === 0 ? (
                                <div className="form-input bg-gray-100 text-gray-500">
                                    No voices available. <Link href="/" className="text-blue-600 hover:text-blue-800">Create a voice first</Link>.
                                </div>
                            ) : (
                                <select
                                    id="voice"
                                    value={selectedVoice}
                                    onChange={(e) => setSelectedVoice(e.target.value)}
                                    className="form-input"
                                >
                                    {voiceLibrary.map((voice) => (
                                        <option key={voice.voice_id} value={voice.voice_id}>
                                            {voice.name} (created {new Date(voice.created_date * 1000).toLocaleDateString()})
                                        </option>
                                    ))}
                                </select>
                            )}
                        </div>

                        <div className="flex items-center space-x-2">
                            <button
                                type="button"
                                onClick={handleSubmit}
                                disabled={isLoading || !text.trim() || !selectedVoice || isLoadingLibrary}
                                className="btn-primary flex-1"
                            >
                                {isLoading ? 'Generating...' : 'Generate Speech'}
                            </button>
                            
                            {isLoading && currentJobId && (
                                <button
                                    type="button"
                                    onClick={stopJob}
                                    className="btn-secondary px-3 py-2 text-sm"
                                >
                                    Stop
                                </button>
                            )}
                        </div>

                        {error && (
                            <div className="rounded-md bg-red-50 p-4">
                                <div className="flex">
                                    <div className="ml-3">
                                        <h3 className="text-sm font-medium text-red-800">Error</h3>
                                        <div className="mt-2 text-sm text-red-700">
                                            <p>{error}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {result && (
                            <div className="space-y-4">
                                <div>
                                    <h3 className="form-label mb-2">Generated Speech</h3>
                                    <audio
                                        src={`data:audio/wav;base64,${result.audio_base64}`}
                                        controls
                                        className="w-full"
                                    />
                                </div>

                                {result.metadata && (
                                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                                        <h4 className="text-sm font-medium text-gray-900 mb-2">Generation Information</h4>
                                        <div className="space-y-2 text-sm text-gray-600">
                                            <p>Voice: {result.metadata.voice_name}</p>
                                            <p>Voice ID: {result.metadata.voice_id}</p>
                                            <p>Text: "{result.metadata.text_input}"</p>
                                            {result.metadata.generation_time && (
                                                <p>Generation Time: {result.metadata.generation_time.toFixed(2)}s</p>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
} 