'use client';

import { useEffect, useRef, useState } from 'react';
import { MicrophoneIcon, StopIcon } from '@heroicons/react/24/solid';
import dynamic from 'next/dynamic';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';

// Dynamically import WaveSurfer with no SSR
const WaveSurfer = dynamic(() => 
    import('wavesurfer.js').then(mod => mod.default), { 
    ssr: false,
    loading: () => <div className="h-10 bg-gray-100 rounded animate-pulse" />
});

interface AudioRecorderProps {
    onAudioReady: (base64Audio: string) => void;
}

export const AudioRecorder = ({ onAudioReady }: AudioRecorderProps) => {
    const waveformRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<any>(null);
    const [waveformError, setWaveformError] = useState<string | null>(null);
    
    const {
        isRecording,
        status,
        mediaBlobUrl,
        startRecording,
        stopRecording,
        getBase64Audio
    } = useAudioRecorder();

    useEffect(() => {
        const initWaveform = async () => {
            if (typeof window === 'undefined' || !waveformRef.current || !mediaBlobUrl) {
                return;
            }

            try {
                // Cleanup previous instance
                if (wavesurferRef.current) {
                    wavesurferRef.current.destroy();
                    wavesurferRef.current = null;
                }

                // Create new instance
                const WaveSurferModule = await WaveSurfer;
                wavesurferRef.current = WaveSurferModule.create({
                    container: waveformRef.current,
                    waveColor: '#6366F1',
                    progressColor: '#4F46E5',
                    cursorColor: 'transparent',
                    barWidth: 2,
                    barGap: 2,
                    height: 40,
                    normalize: true,
                    responsive: true,
                });

                // Load audio
                await wavesurferRef.current.load(mediaBlobUrl);
                setWaveformError(null);
            } catch (error) {
                console.error('Error initializing waveform:', error);
                setWaveformError('Failed to display waveform');
            }
        };

        initWaveform();

        // Cleanup
        return () => {
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
                wavesurferRef.current = null;
            }
        };
    }, [mediaBlobUrl]);

    useEffect(() => {
        const getAudioData = async () => {
            if (mediaBlobUrl && !isRecording) {
                try {
                    const base64Audio = await getBase64Audio();
                    if (base64Audio) {
                        onAudioReady(base64Audio);
                    }
                } catch (error) {
                    console.error('Error getting audio data:', error);
                }
            }
        };

        getAudioData();
    }, [mediaBlobUrl, isRecording, getBase64Audio, onAudioReady]);

    return (
        <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center space-x-4 mb-3">
                <button
                    onClick={isRecording ? stopRecording : startRecording}
                    className={`flex items-center justify-center p-2 rounded-full transition-colors ${
                        isRecording
                            ? 'bg-red-500 hover:bg-red-600 text-white'
                            : status === 'failed'
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                    }`}
                    title={isRecording ? 'Stop Recording' : 'Start Recording'}
                    disabled={status === 'failed'}
                >
                    {isRecording ? (
                        <StopIcon className="h-5 w-5" />
                    ) : (
                        <MicrophoneIcon className="h-5 w-5" />
                    )}
                </button>
                <span className="text-sm text-gray-600 capitalize">
                    {status === 'recording' ? 'Recording...' : status}
                </span>
            </div>

            {status === 'failed' && (
                <div className="text-sm text-red-600 mb-3">
                    Failed to access microphone. Please check your browser permissions.
                </div>
            )}

            <div 
                ref={waveformRef}
                className={`bg-white rounded border border-gray-200 ${!mediaBlobUrl ? 'h-10' : ''}`}
            />

            {waveformError && (
                <div className="text-sm text-red-600 mt-2">
                    {waveformError}
                </div>
            )}

            {mediaBlobUrl && !isRecording && (
                <div className="mt-3">
                    <audio
                        src={mediaBlobUrl}
                        controls
                        className="w-full h-8"
                    />
                </div>
            )}
        </div>
    );
}; 