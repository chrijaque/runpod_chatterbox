import { useEffect, useRef } from 'react';
import { MicrophoneIcon, StopIcon } from '@heroicons/react/24/solid';
import WaveSurfer from 'wavesurfer.js';
import { useAudioRecorder } from '@/hooks/useAudioRecorder';

interface AudioRecorderProps {
    onAudioReady: (base64Audio: string) => void;
}

export const AudioRecorder = ({ onAudioReady }: AudioRecorderProps) => {
    const waveformRef = useRef<HTMLDivElement>(null);
    const wavesurferRef = useRef<WaveSurfer | null>(null);
    const {
        isRecording,
        status,
        mediaBlobUrl,
        startRecording,
        stopRecording,
        getBase64Audio
    } = useAudioRecorder();

    useEffect(() => {
        if (waveformRef.current && mediaBlobUrl) {
            if (wavesurferRef.current) {
                wavesurferRef.current.destroy();
            }

            wavesurferRef.current = WaveSurfer.create({
                container: waveformRef.current,
                waveColor: '#4F46E5',
                progressColor: '#818CF8',
                cursorColor: '#C7D2FE',
                barWidth: 2,
                barGap: 3,
                height: 60,
                normalize: true,
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
        <div className="w-full max-w-2xl mx-auto p-4 bg-white rounded-lg shadow-md">
            <div className="flex flex-col items-center gap-4">
                <div className="flex items-center gap-4">
                    <button
                        onClick={isRecording ? stopRecording : startRecording}
                        className={`p-4 rounded-full transition-colors ${
                            isRecording
                                ? 'bg-red-500 hover:bg-red-600'
                                : 'bg-indigo-500 hover:bg-indigo-600'
                        }`}
                    >
                        {isRecording ? (
                            <StopIcon className="h-6 w-6 text-white" />
                        ) : (
                            <MicrophoneIcon className="h-6 w-6 text-white" />
                        )}
                    </button>
                    <span className="text-sm text-gray-500 capitalize">
                        {status}
                    </span>
                </div>

                <div
                    ref={waveformRef}
                    className="w-full h-16 bg-gray-50 rounded-lg"
                />

                {mediaBlobUrl && !isRecording && (
                    <audio src={mediaBlobUrl} controls className="w-full" />
                )}
            </div>
        </div>
    );
}; 