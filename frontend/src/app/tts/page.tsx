'use client';

import { useEffect, useState, useRef } from 'react';
import { RUNPOD_API_KEY, TTS_API_ENDPOINT, VOICE_API, TTS_GENERATIONS_API } from '@/config/api';
import Link from 'next/link';

interface Voice {
    voice_id: string;
    name: string;
    sample_file: string;
    profile_file: string;
    created_date: number;
}

interface TTSResult {
    audio_base64: string;
    metadata: {
        voice_id: string;
        voice_name: string;
        text_input: string;
        generation_time: number;
        tts_file: string;
        timestamp: string;
        firebase_url?: string; // Firebase URL for file access
    };
}

interface TTSGeneration {
    file_id: string;
    voice_id: string;
    voice_name: string;
    file_path: string;
    created_date: number;
    timestamp: string;
    file_size: number;
}

export default function TTSPage() {
    const [text, setText] = useState('');
    const [selectedVoice, setSelectedVoice] = useState<string>('');
    const [voiceLibrary, setVoiceLibrary] = useState<Voice[]>([]);
    const [ttsGenerations, setTtsGenerations] = useState<TTSGeneration[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [isLoadingLibrary, setIsLoadingLibrary] = useState(false);
    const [isLoadingGenerations, setIsLoadingGenerations] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<TTSResult | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [playingGeneration, setPlayingGeneration] = useState<string | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Load voice library and TTS generations on component mount
    useEffect(() => {
        loadVoiceLibrary();
        loadTTSGenerations();
    }, []);

    const loadVoiceLibrary = async () => {
        setIsLoadingLibrary(true);
        try {
            const response = await fetch(VOICE_API, {
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

    const loadTTSGenerations = async () => {
        setIsLoadingGenerations(true);
        try {
            const response = await fetch(TTS_GENERATIONS_API, {
                method: 'GET',
            });

            const data = await response.json();
            console.log('TTS Generations API response:', data);

            if (data.status === 'success') {
                setTtsGenerations(data.generations || []);
            } else {
                throw new Error(data.message || 'Failed to load TTS generations');
            }
        } catch (err) {
            console.error('Error loading TTS generations:', err);
            setError('Failed to load TTS generations');
        } finally {
            setIsLoadingGenerations(false);
        }
    };

    const playTTSGeneration = async (fileId: string) => {
        setPlayingGeneration(fileId);
        try {
            const audioUrl = `${TTS_GENERATIONS_API}/${fileId}/audio`;
            
            const audio = new Audio(audioUrl);
            audio.onended = () => setPlayingGeneration(null);
            audio.onerror = () => {
                console.error('Error playing TTS generation');
                setPlayingGeneration(null);
            };
            
            await audio.play();
        } catch (err) {
            console.error('Error playing TTS generation:', err);
            setPlayingGeneration(null);
        }
    };

    const pollJobStatus = async (jobId: string) => {
        const statusEndpoint = TTS_API_ENDPOINT.replace('/run', `/status/${jobId}`);
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

            if (!TTS_API_ENDPOINT) {
                throw new Error('TTS endpoint not configured. Please set NEXT_PUBLIC_TTS_ENDPOINT_ID');
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

                    // Fetch the voice profile from FastAPI
        console.log('📥 Fetching voice profile from FastAPI...');
        const profileResponse = await fetch(`${VOICE_API}/${selectedVoice}/profile`);
        
        if (!profileResponse.ok) {
            const errorData = await profileResponse.json().catch(() => ({}));
            throw new Error(`Failed to fetch voice profile: ${errorData.message || profileResponse.statusText}`);
        }
        
        const profileData = await profileResponse.json();
        console.log('✅ Voice profile fetched:', {
            hasProfile: !!profileData.profile_base64,
            profileSize: profileData.profile_base64 ? profileData.profile_base64.length : 0
        });

            const response = await fetch(TTS_API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`
                },
                body: JSON.stringify({
                    input: {
                        text: text,
                        voice_id: selectedVoice,
                        profile_base64: profileData.profile_base64,
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
                
                // Refresh TTS generations after successful generation
                await loadTTSGenerations();
                
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
            const cancelEndpoint = TTS_API_ENDPOINT.replace('/run', `/cancel/${currentJobId}`);
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
                                    
                                    {result.audio_base64 ? (
                                        // Small file - play directly from base64
                                        <audio
                                            src={`data:audio/wav;base64,${result.audio_base64}`}
                                            controls
                                            className="w-full"
                                        />
                                    ) : (
                                        // Large file - saved to Firebase, show success message
                                        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                                            <div className="flex items-center">
                                                <div className="flex-shrink-0">
                                                    <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                                    </svg>
                                                </div>
                                                <div className="ml-3">
                                                    <h3 className="text-sm font-medium text-green-800">
                                                        TTS Generation Complete!
                                                    </h3>
                                                    <div className="mt-2 text-sm text-green-700">
                                                        <p>The audio file has been saved to Firebase storage.</p>
                                                        <p className="mt-1">You can find it in the TTS Generations Library below.</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )}
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
                                            {result.metadata.firebase_url && (
                                                <p>Firebase URL: <a href={result.metadata.firebase_url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">View File</a></p>
                                            )}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* TTS Generations Library Section */}
                <div className="card">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xl font-semibold text-gray-900">
                                TTS Generations Library
                            </h2>
                            <button
                                onClick={loadTTSGenerations}
                                disabled={isLoadingGenerations}
                                className="btn-secondary"
                            >
                                {isLoadingGenerations ? 'Loading...' : 'Refresh'}
                            </button>
                        </div>

                        {isLoadingGenerations ? (
                            <div className="text-center py-8">
                                <div className="text-gray-500">Loading TTS generations...</div>
                            </div>
                        ) : ttsGenerations.length === 0 ? (
                            <div className="text-center py-8">
                                <div className="text-gray-500 mb-4">No TTS generations yet.</div>
                                <div className="text-sm text-gray-400">Generate some speech above to see your creations here.</div>
                            </div>
                        ) : (
                            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                                {ttsGenerations.map((generation) => (
                                    <div key={generation.file_id} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
                                        <div className="flex items-center justify-between mb-3">
                                            <h3 className="font-medium text-gray-900">{generation.voice_name}</h3>
                                            <span className="text-xs text-gray-500">
                                                {new Date(generation.created_date * 1000).toLocaleDateString()}
                                            </span>
                                        </div>
                                        
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={() => playTTSGeneration(generation.file_id)}
                                                disabled={playingGeneration === generation.file_id}
                                                className="btn-primary flex-1 text-sm py-2"
                                            >
                                                {playingGeneration === generation.file_id ? (
                                                    <>
                                                        <span className="animate-pulse">Playing...</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        ▶ Play
                                                    </>
                                                )}
                                            </button>
                                        </div>
                                        
                                        <div className="mt-2 text-xs text-gray-400">
                                            <div>ID: {generation.voice_id}</div>
                                            <div>Size: {(generation.file_size / 1024).toFixed(1)} KB</div>
                                            <div>Time: {generation.timestamp}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
} 