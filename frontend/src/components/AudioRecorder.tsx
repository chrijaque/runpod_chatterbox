'use client';

import { useEffect, useRef } from 'react';
import { MicrophoneIcon, StopIcon } from '@heroicons/react/24/solid';
import dynamic from 'next/dynamic';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';

// Dynamically import WaveSurfer with no SSR
const WaveSurfer = dynamic(() => import('wavesurfer.js'), { ssr: false });

interface AudioRecorderProps {
    onAudioReady: (base64Audio: string) => void;
}

export const AudioRecorder = ({ onAudioReady }: AudioRecorderProps) => {
    const waveformRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<any>(null);
    const {
        isRecording,
        status,
        mediaBlobUrl,
        startRecording,
        stopRecording,
        getBase64Audio
    } = useAudioRecorder();

    useEffect(() => {
        if (typeof window !== 'undefined' && waveformRef.current && mediaBlobUrl) {
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
            }

            wavesurferRef.current = WaveSurfer.create({
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

            wavesurferRef.current.load(mediaBlobUrl);
        }
    }, [mediaBlobUrl]);

    useEffect(() => {
        const getAudioData = async () => {
            if (mediaBlobUrl && !isRecording) {
                const base64Audio = await getBase64Audio();
                if (base64Audio) {
                    onAudioReady(base64Audio);
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
                            : 'bg-indigo-500 hover:bg-indigo-600 text-white'
                    }`}
                    title={isRecording ? 'Stop Recording' : 'Start Recording'}
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

            <div 
                ref={waveformRef}
                className={`bg-white rounded border border-gray-200 ${!mediaBlobUrl ? 'h-10' : ''}`}
            />

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