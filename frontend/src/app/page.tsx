'use client';

import { useEffect, useState, useRef } from 'react';
import { AudioRecorder } from '@/components/AudioRecorder';
import { FileUploader } from '@/components/FileUploader';
import { API_ENDPOINT, RUNPOD_API_KEY } from '@/config/api';

interface FileMetadata {
    voice_file: string;
    output_file: string;
    sample_rate: number;
    audio_shape: number[];
}

export default function Home() {
    useEffect(() => {
        console.log('Environment variables check:', {
            RUNPOD_API_KEY: process.env.NEXT_PUBLIC_RUNPOD_API_KEY ? 'Set' : 'Not set',
            RUNPOD_ENDPOINT_ID: process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_ID ? 'Set' : 'Not set'
        });
    }, []);

    const [prompt, setPrompt] = useState('');
    const [audioData, setAudioData] = useState<string | null>(null);
    const [audioFormat, setAudioFormat] = useState<string>('wav');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [metadata, setMetadata] = useState<FileMetadata | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

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
            console.log('Cancel response:', data);

            // Abort the polling
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }

            // Cleanup
            setIsLoading(false);
            setCurrentJobId(null);
            setError('Job cancelled by user');
        } catch (err) {
            console.error('Error cancelling job:', err);
            setError('Failed to cancel job');
        }
    };

    const pollJobStatus = async (jobId: string) => {
        const statusEndpoint = API_ENDPOINT.replace('/run', `/status/${jobId}`);
        abortControllerRef.current = new AbortController();
        
        try {
            while (true) {
                // Check if polling should be stopped
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
                console.log('Status check response:', data);
                
                if (data.status === 'COMPLETED') {
                    if (data.output?.status === 'error') {
                        throw new Error(data.output.message);
                    }
                    if (data.output?.audio_base64) {
                        return data.output.audio_base64;
                    }
                    throw new Error('No audio data in response');
                } else if (data.status === 'FAILED') {
                    throw new Error(data.error || 'Job failed');
                } else if (data.status === 'IN_QUEUE' || data.status === 'IN_PROGRESS') {
                    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds before polling again
                    continue;
                } else {
                    throw new Error(`Unexpected status: ${data.status}`);
                }
            }
        } finally {
            // Cleanup
            abortControllerRef.current = null;
        }
    };

    const handleSubmit = async () => {
        console.log('Submitting with:', { prompt, audioFormat, hasAudioData: !!audioData });
        
        if (!prompt || !audioData) {
            setError('Please provide both a prompt and audio input');
            return;
        }

        if (!RUNPOD_API_KEY) {
            setError('RunPod API key is not configured');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);
        setCurrentJobId(null);
        setMetadata(null); // Clear previous metadata

        try {
            console.log('Making API request...');
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${RUNPOD_API_KEY}`
                },
                body: JSON.stringify({
                    input: {
                        prompt,
                        audio_data: audioData,
                        audio_format: audioFormat,
                    },
                }),
            });

            const data = await response.json();
            console.log('API response:', data);

            if (data.id) {
                setCurrentJobId(data.id);
                // Job was accepted, start polling for status
                const result = await pollJobStatus(data.id);
                setResult(result);
                // Save metadata if available
                if (data.metadata) {
                    setMetadata(data.metadata);
                }
            } else {
                throw new Error('No job ID in response');
            }
        } catch (err: unknown) {
            console.error('API error:', err);
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

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
        };
    }, []);

    const handleAudioReady = (base64Audio: string) => {
        console.log('Audio recording ready');
        setAudioData(base64Audio);
        setAudioFormat('wav');
    };

    const handleFileSelect = (base64Audio: string, format: string) => {
        console.log('File selected:', { format, audioLength: base64Audio?.length });
        setAudioData(base64Audio);
        setAudioFormat(format);
    };

    return (
        <main className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
            <div className="max-w-2xl mx-auto space-y-6">
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-900">
                        Voice Cloning Studio
                    </h1>
                    <p className="mt-2 text-sm text-gray-600">
                        Record or upload audio, then enter text to generate speech in your voice
                    </p>
                </div>

                <div className="card">
                    <div className="space-y-6">
                        <div>
                            <label htmlFor="prompt" className="form-label">
                                Text to Convert
                            </label>
                            <textarea
                                id="prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                className="form-input"
                                rows={3}
                                placeholder="Enter the text you want to convert to speech..."
                            />
                        </div>

                        <div>
                            <label className="form-label mb-2">
                                Voice Input
                            </label>
                            <div className="space-y-4">
                                <AudioRecorder onAudioReady={handleAudioReady} />
                                <div className="relative">
                                    <div className="absolute inset-0 flex items-center">
                                        <div className="w-full border-t border-gray-300" />
                                    </div>
                                    <div className="relative flex justify-center">
                                        <span className="bg-white px-2 text-sm text-gray-500">or</span>
                                    </div>
                                </div>
                                <FileUploader onFileSelect={handleFileSelect} />
                            </div>
                        </div>

                        <div className="flex gap-4">
                            <button
                                onClick={handleSubmit}
                                disabled={isLoading || !prompt || !audioData}
                                className="btn-primary flex-1"
                            >
                                {isLoading ? 'Processing...' : 'Generate Voice Clone'}
                            </button>

                            {isLoading && currentJobId && (
                                <button
                                    onClick={stopJob}
                                    className="btn-secondary"
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
                                    <h3 className="form-label mb-2">Generated Audio</h3>
                                    <audio
                                        src={`data:audio/wav;base64,${result}`}
                                        controls
                                        className="w-full"
                                    />
                                </div>

                                {metadata && (
                                    <div className="bg-white p-4 rounded-lg border border-gray-200">
                                        <h4 className="text-sm font-medium text-gray-900 mb-2">File Information</h4>
                                        <div className="space-y-2 text-sm text-gray-600">
                                            <p>Voice File: {metadata.voice_file}</p>
                                            <p>Output File: {metadata.output_file}</p>
                                            <p>Sample Rate: {metadata.sample_rate} Hz</p>
                                            <p>Audio Shape: [{metadata.audio_shape.join(', ')}]</p>
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
