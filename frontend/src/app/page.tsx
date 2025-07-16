'use client';

import { useState } from 'react';
import { AudioRecorder } from '@/components/AudioRecorder';
import { FileUploader } from '@/components/FileUploader';
import { API_ENDPOINT, RUNPOD_API_KEY } from '@/config/api';

export default function Home() {
    const [prompt, setPrompt] = useState('');
    const [audioData, setAudioData] = useState<string | null>(null);
    const [audioFormat, setAudioFormat] = useState<string>('wav');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);

    const handleSubmit = async () => {
        if (!prompt || !audioData) {
            setError('Please provide both audio input');
            return;
        }

        if (!RUNPOD_API_KEY) {
            setError('RunPod API key is not configured');
            return;
        }

        setIsLoading(true);
        setError(null);
        setResult(null);

        try {
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

            // Handle RunPod specific response format
            if (data.status === 'COMPLETED') {
                if (data.output?.status === 'error') {
                    throw new Error(data.output.message);
                }
                if (data.output?.audio_base64) {
                    setResult(data.output.audio_base64);
                }
            } else if (data.status === 'FAILED') {
                throw new Error(data.error);
            } else {
                throw new Error('Unexpected response from server');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred');
        } finally {
            setIsLoading(false);
        }
    };

    const handleAudioReady = (base64Audio: string) => {
        setAudioData(base64Audio);
        setAudioFormat('wav');
    };

    const handleFileSelect = (base64Audio: string, format: string) => {
        setAudioData(base64Audio);
        setAudioFormat(format);
    };

    return (
        <main className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto">
                <h1 className="text-4xl font-bold text-center text-gray-900 mb-8">
                    Chatterbox Voice Clone
                </h1>

                <div className="space-y-8">
                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <h2 className="text-xl font-semibold mb-4">
                            1. Enter Your Prompt
                        </h2>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full p-3 border border-gray-300 rounded-md"
                            rows={4}
                            placeholder="Enter the text you want to convert to speech..."
                        />
                    </div>

                    <div className="bg-white p-6 rounded-lg shadow-md">
                        <h2 className="text-xl font-semibold mb-4">
                            2. Record or Upload Audio
                        </h2>
                        <div className="space-y-6">
                            <div>
                                <h3 className="text-lg font-medium mb-2">
                                    Record Audio
                                </h3>
                                <AudioRecorder onAudioReady={handleAudioReady} />
                            </div>
                            <div className="border-t pt-6">
                                <h3 className="text-lg font-medium mb-2">
                                    Or Upload Audio File
                                </h3>
                                <FileUploader onFileSelect={handleFileSelect} />
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-center">
                        <button
                            onClick={handleSubmit}
                            disabled={isLoading || !prompt || !audioData}
                            className="px-6 py-3 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                        >
                            {isLoading ? 'Processing...' : 'Generate Voice Clone'}
                        </button>
                    </div>

                    {error && (
                        <div className="bg-red-50 p-4 rounded-md">
                            <p className="text-red-800">{error}</p>
                        </div>
                    )}

                    {result && (
                        <div className="bg-white p-6 rounded-lg shadow-md">
                            <h2 className="text-xl font-semibold mb-4">Result</h2>
                            <audio
                                src={`data:audio/wav;base64,${result}`}
                                controls
                                className="w-full"
                            />
                        </div>
                    )}
                </div>
            </div>
        </main>
    );
}
