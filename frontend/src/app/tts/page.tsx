'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import { RUNPOD_API_KEY, TTS_API_ENDPOINT, VOICE_API, TTS_API, TTS_GENERATIONS_API } from '@/config/api';

import Link from 'next/link';

interface Voice {
    voice_id: string;
    name: string;
    sample_file: string;
    profile_file: string;
    created_date: number;
}

interface TTSResult {
    status: string;
    audio_path?: string;  // Changed from audio_base64
    metadata?: {
        voice_id: string;
        voice_name: string;
        text_input: string;
        generation_time: number;
        sample_rate: number;
        audio_shape: number[];
        tts_file: string;
        timestamp: string;
        response_type: string;
        format: string;  // New: indicates MP3 format
        audio_path?: string;  // Add audio_path to metadata
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
    const [language, setLanguage] = useState<string>('en');
    const [isKidsVoice, setIsKidsVoice] = useState<boolean>(false);
    const [storyType, setStoryType] = useState<string>('user');
    const [modelType] = useState<'chatterbox'>('chatterbox');
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



    const loadVoiceLibrary = useCallback(async () => {
        setIsLoadingLibrary(true);
        try {
            // Use the voice library endpoint with language and kids voice parameters
            const response = await fetch(`${VOICE_API}/by-language/${language}?is_kids_voice=${isKidsVoice}`, {
                method: 'GET',
            });

            const data = await response.json();
            console.log('TTS Library API response:', data);

            if (data.status === 'success') {
                // Transform the API response to match our Voice interface
                const voices = data.voices.map((voice: any) => ({
                    voice_id: voice.voice_id,
                    name: voice.name || voice.voice_id,
                    sample_file: voice.sample_file || '',
                    profile_file: voice.embedding_file || '', // Map embedding_file to profile_file
                    created_date: voice.created_date || Date.now() / 1000
                }));
                
                setVoiceLibrary(voices);
                // Auto-select first voice if available
                if (voices.length > 0 && !selectedVoice) {
                    setSelectedVoice(voices[0].voice_id);
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
    }, []);

    const loadTTSGenerations = useCallback(async () => {
        setIsLoadingGenerations(true);
        try {
            // Use the new Firebase-based endpoint to list stories by language and type
            const response = await fetch(`${TTS_API}/stories/${language}?story_type=${storyType}`, {
                method: 'GET',
            });

            const data = await response.json();
            console.log('TTS Stories API response:', data);

            if (data.status === 'success') {
                // Transform Firebase data to match our TTSGeneration interface
                const generations = (data.stories || []).map((story: any) => ({
                    file_id: story.generation_id || story.file_id,
                    voice_id: story.voice_id || 'unknown',
                    voice_name: story.voice_name || 'Unknown Voice',
                    file_path: story.audio_file || '', // Use audio_file directly from Firebase
                    created_date: story.created_date ? story.created_date : Date.now() / 1000,
                    timestamp: story.timestamp || new Date().toISOString(),
                    file_size: story.file_size || 0
                }));
                setTtsGenerations(generations);
                console.log(`✅ Loaded ${generations.length} TTS generations`);
            } else {
                throw new Error(data.detail || data.message || 'Failed to load TTS generations');
            }
        } catch (err) {
            console.error('Error loading TTS generations:', err);
            // Don't show error if it's just that there are no generations yet
            if (err instanceof Error && err.message.includes('Not Found')) {
                console.log('ℹ️ No TTS generations found yet - this is normal for a new setup');
                setTtsGenerations([]);
            } else {
                setError('Failed to load TTS generations');
                setTtsGenerations([]);
            }
        } finally {
            setIsLoadingGenerations(false);
        }
    }, [language, storyType]);

    const playTTSGeneration = async (fileId: string) => {
        setPlayingGeneration(fileId);
        try {
            // Find the generation with the matching file_id
            const generation = ttsGenerations.find(g => g.file_id === fileId);
            
            if (!generation || !generation.file_path) {
                throw new Error('TTS generation not found or no audio file available');
            }
            
            console.log('Playing TTS generation from URL:', generation.file_path);
            
            // Create audio element and play directly from Firebase URL
            const audio = new Audio(generation.file_path);
            audio.onended = () => setPlayingGeneration(null);
            audio.onerror = () => {
                console.error('Error playing story audio from Firebase');
                setPlayingGeneration(null);
            };
            
            await audio.play();
        } catch (err) {
            console.error('Error playing TTS generation:', err);
            setPlayingGeneration(null);
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

            if (!text.trim()) {
                throw new Error('Please enter text to synthesize');
            }

            if (!selectedVoice) {
                throw new Error('Please select a voice from the library');
            }

            console.log('🚀 Starting TTS generation with organized workflow...', { 
                text: text.substring(0, 50) + '...', 
                selectedVoice 
            });

            // Get the voice profile using the proxy endpoint to avoid CORS issues
            const profileResponse = await fetch(`${VOICE_API}/${selectedVoice}/profile?language=${language}&is_kids_voice=${isKidsVoice}`);
            if (!profileResponse.ok) {
                throw new Error('Failed to fetch voice profile');
            }
            const profileData = await profileResponse.json();
            
            if (profileData.status !== 'success' || !profileData.profile_base64) {
                throw new Error('Voice profile not found or invalid');
            }
            
            const profileBase64 = profileData.profile_base64;

            // Create JSON request matching TTSGenerateRequest schema
            const requestData = {
                voice_id: selectedVoice,
                text: text,
                profile_base64: profileBase64, // Use the base64 string directly
                language: language,
                story_type: storyType,
                is_kids_voice: isKidsVoice,
                model_type: modelType  // New: include model type
            };

            console.log('📤 Sending organized TTS request to FastAPI...', {
                voice_id: selectedVoice,
                text_length: text.length,
                language: language,
                is_kids_voice: isKidsVoice,
                story_type: storyType
            });

            // Send to FastAPI using the new organized TTS endpoint
            const response = await fetch(`${TTS_API}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();
            console.log('📨 FastAPI TTS response received:', {
                status: data.status,
                hasGenerationId: !!data.generation_id,
                hasAudioPath: !!data.audio_path,
                hasMetadata: !!data.metadata,
                hasJobId: !!data.job_id,
                jobId: data.job_id,
                keys: Object.keys(data)
            });

            if (!response.ok) {
                throw new Error(data.detail || data.message || 'Failed to generate TTS');
            }

            if (data.status === 'success') {
                console.log('🏁 TTS generation completed successfully:', {
                    generation_id: data.generation_id,
                    hasAudioPath: !!data.audio_path,
                    firebase_urls: data.metadata?.firebase_urls
                });
                
                // Set the result for display
                setResult({
                    status: data.status,
                    audio_path: data.audio_path,
                    metadata: {
                        voice_id: data.voice_id,
                        voice_name: data.metadata?.voice_name || selectedVoice,
                        text_input: text,
                        generation_time: data.metadata?.generation_time || 0,
                        sample_rate: data.metadata?.sample_rate || 0,
                        audio_shape: data.metadata?.audio_shape || [],
                        tts_file: data.metadata?.tts_file || '',
                        timestamp: data.metadata?.timestamp || new Date().toISOString(),
                        response_type: data.metadata?.response_type || '',
                        format: data.metadata?.format || '',
                        audio_path: data.audio_path // Add audio_path to metadata
                    }
                });
                
                // Refresh TTS generations after successful generation
                await loadTTSGenerations();
                
            } else {
                throw new Error('TTS generation failed');
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

                        {/* Model Info */}
                        <div className="mb-6">
                            <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                <div className="text-sm text-blue-800">
                                    <strong>ChatterboxTTS:</strong> Fast, efficient TTS generation optimized for real-time applications.
                                </div>
                            </div>
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

                        {/* Metadata Configuration for Organized Storage */}
                        <div className="space-y-4">
                            <h3 className="text-sm font-medium text-gray-900">Storage Configuration</h3>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div>
                                    <label htmlFor="tts-language" className="block text-sm font-medium text-gray-700 mb-1">
                                        Language
                                    </label>
                                    <select
                                        id="tts-language"
                                        value={language}
                                        onChange={(e) => setLanguage(e.target.value)}
                                        className="form-input"
                                    >
                                        <option value="en">English</option>
                                        <option value="da">Danish</option>
                                        <option value="fr">French</option>
                                        <option value="de">German</option>
                                        <option value="es">Spanish</option>
                                        <option value="tr">Turkish</option>
                                        <option value="ar">Arabic</option>
                                        <option value="zh">Chinese</option>
                                        <option value="ja">Japanese</option>
                                        <option value="ko">Korean</option>
                                    </select>
                                </div>
                                <div>
                                    <label htmlFor="story-type" className="block text-sm font-medium text-gray-700 mb-1">
                                        Story Type
                                    </label>
                                    <select
                                        id="story-type"
                                        value={storyType}
                                        onChange={(e) => setStoryType(e.target.value)}
                                        className="form-input"
                                    >
                                        <option value="user">User Generated</option>
                                        <option value="app">App Generated</option>
                                    </select>
                                </div>
                                <div>
                                    <label className="flex items-center space-x-2">
                                        <input
                                            type="checkbox"
                                            checked={isKidsVoice}
                                            onChange={(e) => setIsKidsVoice(e.target.checked)}
                                            className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                        />
                                        <span className="text-sm font-medium text-gray-700">Kids Voice</span>
                                    </label>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Use kids voice settings
                                    </p>
                                </div>
                            </div>
                            <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                                <strong>Storage Path:</strong> audio/stories/{language}/{storyType}/
                            </div>
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
                                    
                                    {result.audio_path ? (
                                        // File saved to Firebase - show success message with path
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
                                                        <p className="mt-1">Path: {result.audio_path}</p>
                                                        <p className="mt-1">You can find it in the TTS Generations Library below.</p>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    ) : (
                                        // No audio path - show error
                                        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                                            <div className="flex items-center">
                                                <div className="flex-shrink-0">
                                                    <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                                    </svg>
                                                </div>
                                                <div className="ml-3">
                                                    <h3 className="text-sm font-medium text-red-800">
                                                        TTS Generation Failed
                                                    </h3>
                                                    <div className="mt-2 text-sm text-red-700">
                                                        <p>No audio file path received from the server.</p>
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
                                            {result.metadata.audio_path && (
                                                <p>Audio Path: <a href={result.metadata.audio_path} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">View File</a></p>
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