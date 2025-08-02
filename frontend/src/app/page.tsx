'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { AudioRecorder } from '@/components/AudioRecorder';
import { FileUploader } from '@/components/FileUploader';

import { API_ENDPOINT, VOICE_API } from '@/config/api';

interface FileMetadata {
    voice_id: string;
    voice_name: string;
    embedding_path: string;
    embedding_exists: boolean;
    has_embedding_support: boolean;
    generation_method: string;
    sample_file: string;
    template_message: string;
    sample_rate: number;
    audio_shape: number[];
}

interface Voice {
    voice_id: string;
    name: string;
    sample_file: string;
    embedding_file: string;
    created_date: number;
}

export default function Home() {
    useEffect(() => {
        console.log('Environment variables check:', {
            RUNPOD_API_KEY: process.env.NEXT_PUBLIC_RUNPOD_API_KEY ? 'Set' : 'Not set',
            VC_CB_ENDPOINT_ID: process.env.NEXT_PUBLIC_VC_CB_ENDPOINT_ID ? 'Set' : 'Not set'
        });
    }, []);

    const [name, setName] = useState('');
    const [audioData, setAudioData] = useState<string | null>(null);
    const [audioFormat, setAudioFormat] = useState<string>('wav');
    const [language, setLanguage] = useState<string>('en');
    const [isKidsVoice, setIsKidsVoice] = useState<boolean>(false);
    const [modelType] = useState<'chatterbox'>('chatterbox');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<string | null>(null);
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [metadata, setMetadata] = useState<FileMetadata | null>(null);
    const [voiceLibrary, setVoiceLibrary] = useState<Voice[]>([]);
    const [isLoadingLibrary, setIsLoadingLibrary] = useState(false);
    const [playingVoice, setPlayingVoice] = useState<string | null>(null);
    const abortControllerRef = useRef<AbortController | null>(null);

    // Load voice library on component mount
    useEffect(() => {
        loadVoiceLibrary();
    }, []);

    // Reload voice library when metadata settings change
    useEffect(() => {
        loadVoiceLibrary();
    }, [language, isKidsVoice]);

    const saveVoiceToAPI = async (result: any, voiceName: string) => {
        try {
            console.log('💾 Saving voice to API...', { result, voiceName });
            
            // Check if we have the required data
            if (!result || !result.sample_audio_path) {
                console.error('❌ No sample audio path in result - cannot save voice', {
                    hasResult: !!result,
                    resultType: typeof result,
                    hasSampleAudioPath: !!(result && result.sample_audio_path),
                    resultKeys: result ? Object.keys(result) : []
                });
                return;
            }
            
            // Voice is now saved directly by RunPod to Firebase
            // We just need to refresh the voice library
            console.log('✅ Voice saved to Firebase by RunPod, refreshing library...');
                await loadVoiceLibrary();
            
        } catch (error) {
            console.error('❌ Error saving voice to API:', error);
        }
    };

    const loadVoiceLibrary = async () => {
        setIsLoadingLibrary(true);
        try {
            // Use the new Firebase-based endpoint to list voices by language
            const response = await fetch(`${VOICE_API}/by-language/${language}?is_kids_voice=${isKidsVoice}`, {
                method: 'GET',
            });

            const data = await response.json();
            console.log('Voice Library API response:', data);

            if (data.status === 'success') {
                // Debug: Log the first voice structure
                if (data.voices && data.voices.length > 0) {
                    console.log('🔍 First voice structure from API:', data.voices[0]);
                }
                
                // Transform Firebase data to match our Voice interface
                const voices = data.voices.map((voice: any) => ({
                    voice_id: voice.voice_id,
                    name: voice.name || voice.voice_id,
                    sample_file: voice.sample_file || '',
                    embedding_file: voice.embedding_file || '',
                    created_date: voice.created_date || Date.now() / 1000
                }));
                setVoiceLibrary(voices);
            } else {
                throw new Error(data.detail || data.message || 'Failed to load voice library');
            }
        } catch (err) {
            console.error('Error loading voice library:', err);
            // Set empty array on error to avoid breaking the UI
            setVoiceLibrary([]);
        } finally {
            setIsLoadingLibrary(false);
        }
    };

    const playVoiceSample = async (voiceId: string) => {
        setPlayingVoice(voiceId);
        try {
            // Find the voice in the library and use its sample_file URL directly
            const voice = voiceLibrary.find(v => v.voice_id === voiceId);
            
            if (!voice || !voice.sample_file) {
                throw new Error('No sample URL available for this voice');
            }
            
            console.log('Playing voice sample from URL:', voice.sample_file);
            
                // Create audio element and play from Firebase URL
            const audio = new Audio(voice.sample_file);
                audio.onended = () => setPlayingVoice(null);
                audio.onerror = () => {
                    console.error('Error playing audio from Firebase');
                    setPlayingVoice(null);
                };
                
                await audio.play();
        } catch (err) {
            console.error('Error playing voice sample:', err);
            setPlayingVoice(null);
        }
    };

    const stopJob = async () => {
        if (!currentJobId) return;

        try {
            const cancelEndpoint = `${API_ENDPOINT}/cancel/${currentJobId}`;
            const response = await fetch(cancelEndpoint, {
                method: 'POST'
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
        const statusEndpoint = `${API_ENDPOINT}/status/${jobId}`;
        abortControllerRef.current = new AbortController();
        
        try {
            while (true) {
                // Check if polling should be stopped
                if (abortControllerRef.current.signal.aborted) {
                    throw new Error('Polling aborted');
                }

                const response = await fetch(statusEndpoint, {
                    signal: abortControllerRef.current.signal
                });
                
                const data = await response.json();
                console.log('Status check response:', data);
                
                if (data.status === 'COMPLETED') {
                    console.log('🔍 DEBUGGING: Job completed, checking output:', data.output);
                    console.log('🔍 DEBUGGING: Output type:', typeof data.output);
                    console.log('🔍 DEBUGGING: Output keys:', data.output ? Object.keys(data.output) : 'NO OUTPUT');
                    
                    if (data.output?.status === 'error') {
                        throw new Error(data.output.message);
                    }
                    if (data.output?.audio_base64) {
                        console.log('🔍 DEBUGGING: Found audio_base64 in output, returning output object');
                        return data.output;  // Return the entire output object, not just the string
                    }
                    console.log('🔍 DEBUGGING: No audio_base64 found, returning raw output');
                    return data.output;  // Return the raw output if no audio_base64 field
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
        try {
            setIsLoading(true);
            setError(null);
            setResult(null);
            setMetadata(null);

            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            abortControllerRef.current = new AbortController();

            if (!audioData) {
                throw new Error('Please upload or record audio first');
            }

            if (!name.trim()) {
                throw new Error('Please enter a voice clone name');
            }

            console.log('🚀 Starting voice generation...', { name, audioFormat, hasAudioData: !!audioData });

            // Debug audio data before sending
            console.log('🔍 Audio data details before API call:', {
                hasAudioData: !!audioData,
                audioDataLength: audioData?.length || 0,
                audioFormat: audioFormat,
                audioDataPreview: audioData?.substring(0, 200) + '...' || 'No data',
                audioDataEnd: audioData?.substring(-100) || 'No data'
            });

            // Validate audio data before sending
            if (!audioData || audioData.length < 1000) {
                console.error('❌ Invalid audio data before API call:', {
                    hasAudioData: !!audioData,
                    audioDataLength: audioData?.length || 0,
                    minimumExpected: 1000
                });
                throw new Error('Invalid audio data - please record or upload a proper audio file');
            }

            const requestBody = {
                name: name,
                audio_data: audioData,
                audio_format: audioFormat,
                language: language,
                is_kids_voice: isKidsVoice,
                model_type: modelType,  // New: include model type
            };

            console.log('📤 Sending request to API:', {
                endpoint: API_ENDPOINT,
                requestBodyKeys: Object.keys(requestBody),
                audioDataLength: requestBody.audio_data?.length || 0,
                modelType: requestBody.model_type
            });

            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody),
                signal: abortControllerRef.current.signal,
            });

            const data = await response.json();
            console.log('📨 FastAPI response received:', {
                hasId: !!data.id,
                hasOutput: !!data.output,
                hasError: !!data.error,
                status: data.status,
                keys: Object.keys(data)
            });

            if (!response.ok) {
                throw new Error(data.message || 'Failed to generate voice');
            }

            // Check if we got a direct result (new format) or a job ID (old format)
            if (data.status === 'success' && !data.id) {
                // New format: FastAPI returned the final result directly
                console.log('✅ Received final result directly from FastAPI');
                
                const result = data;
                console.log('🔍 DEBUGGING: Raw result from FastAPI:', result);
                console.log('🔍 DEBUGGING: Result type:', typeof result);
                console.log('🔍 DEBUGGING: Result keys:', result ? Object.keys(result) : 'N/A');
                
                console.log('🏁 Final result received:', {
                    hasResult: !!result,
                    resultType: typeof result,
                    hasMetadata: !!(result && result.metadata),
                    status: result?.status
                });
                
                setResult(result); // Store the result directly
                
                // Save metadata if available
                if (result && result.metadata) {
                    setMetadata(result.metadata);
                    console.log('📋 Metadata saved:', result.metadata);
                }
                
                // Refresh voice library after successful creation
                console.log('🔄 Refreshing voice library...');
                await loadVoiceLibrary();
                
            } else if (data.id) {
                // Old format: Job ID returned, need to poll for completion
                setCurrentJobId(data.id);
                console.log('⏳ Job queued, polling for results...', { jobId: data.id });
                
                // Job was accepted, start polling for status
                const result = await pollJobStatus(data.id);
                
                console.log('🔍 DEBUGGING: Raw result from FastAPI:', result);
                console.log('🔍 DEBUGGING: Result type:', typeof result);
                console.log('🔍 DEBUGGING: Result keys:', result ? Object.keys(result) : 'N/A');
                console.log('🔍 DEBUGGING: Has audio_base64:', !!(result && result.audio_base64));
                console.log('🔍 DEBUGGING: Has embedding_base64:', !!(result && result.embedding_base64));
                
                console.log('🏁 Final result received:', {
                    hasResult: !!result,
                    resultType: typeof result,
                    hasAudioBase64: !!(result && result.audio_base64),
                    hasEmbeddingBase64: !!(result && result.embedding_base64),
                    hasMetadata: !!(result && result.metadata),
                    status: result?.status
                });
                
                setResult(result); // Store the result directly
                
                // Save metadata if available
                if (result && result.metadata) {
                    setMetadata(result.metadata);
                    console.log('📋 Metadata saved:', result.metadata);
                }
                
                // Save voice to FastAPI/Firebase after successful generation
                if (result && result.sample_audio_path) {
                    console.log('💾 Voice audio path available:', result.sample_audio_path);
                    await saveVoiceToAPI(result, name);
                } else {
                    console.error('❌ No audio path in result - cannot save voice', {
                        hasResult: !!result,
                        resultType: typeof result,
                        hasAudioPath: !!(result && result.sample_audio_path),
                        resultKeys: result ? Object.keys(result) : []
                    });
                }
                
                // Refresh voice library after successful creation
                console.log('🔄 Refreshing voice library...');
                await loadVoiceLibrary();
            } else {
                throw new Error('Invalid response format - neither job ID nor final result received');
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
        console.log('🎤 Audio recording ready');
        console.log('🔍 Audio data details:', {
            hasData: !!base64Audio,
            dataType: typeof base64Audio,
            dataLength: base64Audio?.length || 0,
            dataPreview: base64Audio?.substring(0, 100) + '...' || 'No data'
        });
        
        // Validate the audio data
        if (!base64Audio || base64Audio.length < 100) {
            console.error('❌ Invalid audio data received:', {
                hasData: !!base64Audio,
                dataLength: base64Audio?.length || 0
            });
            return;
        }
        
        setAudioData(base64Audio);
        setAudioFormat('wav');
        console.log('✅ Audio data set successfully');
    };

    const handleFileSelect = (base64Audio: string, format: string) => {
        console.log('📁 File selected:', { 
            format, 
            audioLength: base64Audio?.length || 0,
            dataType: typeof base64Audio,
            dataPreview: base64Audio?.substring(0, 100) + '...' || 'No data'
        });
        
        // Validate the audio data
        if (!base64Audio || base64Audio.length < 100) {
            console.error('❌ Invalid audio file data received:', {
                hasData: !!base64Audio,
                dataLength: base64Audio?.length || 0,
                format: format
            });
            return;
        }
        
        setAudioData(base64Audio);
        setAudioFormat(format);
        console.log('✅ File audio data set successfully');
    };

    return (
        <main className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
            <div className="max-w-2xl mx-auto space-y-6">
                <div className="text-center">
                    <h1 className="text-3xl font-bold text-gray-900">
                        Voice Cloning Studio
                    </h1>
                    <p className="mt-2 text-sm text-gray-600">
                        Enter a name, record or upload audio, then create your personalized voice clone. Browse your voice library below.
                    </p>
                    <div className="mt-4">
                        <Link href="/tts" className="text-blue-600 hover:text-blue-800 text-sm">
                            🎤 Generate TTS with Saved Voices →
                        </Link>
                    </div>
                </div>

                <div className="card">
                    <div className="space-y-6">
                        <div>
                            <label htmlFor="name" className="form-label">
                                Voice Clone Name
                            </label>
                            <input
                                id="name"
                                type="text"
                                value={name}
                                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)}
                                className="form-input"
                                placeholder="Enter a name for this voice clone (e.g., John, Sarah, etc.)"
                            />
                        </div>

                        {/* Model Info */}
                        <div className="mb-6">
                            <div className="p-3 bg-blue-50 rounded-lg border border-blue-200">
                                <div className="text-sm text-blue-800">
                                    <strong>ChatterboxTTS:</strong> Fast, efficient voice cloning optimized for real-time applications.
                                </div>
                            </div>
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

                        {/* Metadata Configuration for Organized Storage */}
                        <div className="space-y-4">
                            <h3 className="text-sm font-medium text-gray-900">Storage Configuration</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                    <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                                        Language
                                    </label>
                                    <select
                                        id="language"
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
                                        Store in kids voice directory
                                    </p>
                                </div>
                            </div>
                            <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
                                <strong>Storage Path:</strong> audio/voices/{language}/{isKidsVoice ? 'kids/' : ''}recorded/
                            </div>
                        </div>

                        <div className="flex items-center space-x-2">
                            <button
                                type="button"
                                onClick={handleSubmit}
                                disabled={isLoading || !audioData || !name.trim()}
                                className="btn-primary flex-1"
                            >
                                {isLoading ? 'Creating...' : 'Clone Voice'}
                            </button>
                            
                            {/* Stop button when job is running */}
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
                                    <h3 className="form-label mb-2">Voice Clone Sample</h3>
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
                                            <p>Voice Name: {metadata.voice_name}</p>
                                            <p>Voice ID: {metadata.voice_id}</p>
                                            <p>Embedding: {metadata.embedding_exists ? '✅ Cached' : '🆕 Created'}</p>
                                            <p>Generation Method: {metadata.generation_method === 'embedding-based' ? '🚀 Advanced (Embedding)' : '📁 Standard (Audio File)'}</p>
                                            <p>Repository Support: {metadata.has_embedding_support ? '✅ Forked (Enhanced)' : '⚠️ Standard'}</p>
                                            <p>Sample File: {metadata.sample_file}</p>
                                            <p>Generated Message: "{metadata.template_message}"</p>
                                            <p>Sample Rate: {metadata.sample_rate} Hz</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

                {/* Voice Library Section */}
                <div className="card">
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <h2 className="text-xl font-semibold text-gray-900">
                                Voice Library
                            </h2>
                            <button
                                onClick={loadVoiceLibrary}
                                disabled={isLoadingLibrary}
                                className="btn-secondary"
                            >
                                {isLoadingLibrary ? 'Loading...' : 'Refresh'}
                            </button>
                        </div>

                        {isLoadingLibrary ? (
                            <div className="text-center py-8">
                                <div className="text-gray-500">Loading voice library...</div>
                            </div>
                        ) : voiceLibrary.length === 0 ? (
                            <div className="text-center py-8">
                                <div className="text-gray-500 mb-4">Your voice library is empty.</div>
                                <div className="text-sm text-gray-400">Create your first voice above or click "Refresh" to load existing voices.</div>
                            </div>
                        ) : (
                            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                                {voiceLibrary.map((voice) => (
                                    <div key={voice.voice_id} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
                                        <div className="flex items-center justify-between mb-3">
                                            <h3 className="font-medium text-gray-900">{voice.name}</h3>
                                            <span className="text-xs text-gray-500">
                                                {new Date(voice.created_date * 1000).toLocaleDateString()}
                                            </span>
                                        </div>
                                        
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={() => playVoiceSample(voice.voice_id)}
                                                disabled={playingVoice === voice.voice_id}
                                                className="btn-primary flex-1 text-sm py-2"
                                            >
                                                {playingVoice === voice.voice_id ? (
                                                    <>
                                                        <span className="animate-pulse">Playing...</span>
                                                    </>
                                                ) : (
                                                    <>
                                                        ▶ Play Sample
                                                    </>
                                                )}
                                            </button>
                                        </div>
                                        
                                        <div className="mt-2 text-xs text-gray-400">
                                            ID: {voice.voice_id}
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
