'use client';

import { useState, useCallback, useEffect, useRef } from 'react';

export const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [status, setStatus] = useState<string>('idle');
    const chunks = useRef<BlobPart[]>([]);
    const streamRef = useRef<MediaStream | null>(null);

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const initMediaRecorder = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                streamRef.current = stream;
                const recorder = new MediaRecorder(stream);

                recorder.ondataavailable = (e) => {
                    chunks.current.push(e.data);
                };

                recorder.onstop = () => {
                    const blob = new Blob(chunks.current, { type: 'audio/wav' });
                    const url = URL.createObjectURL(blob);
                    setAudioUrl(url);
                    setStatus('stopped');
                    chunks.current = []; // Clear chunks after creating blob
                };

                setMediaRecorder(recorder);
                setStatus('ready');
            } catch (error) {
                console.error('Error initializing media recorder:', error);
                setStatus('failed');
            }
        };

        initMediaRecorder();

        // Cleanup function
        return () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            if (streamRef.current) {
                streamRef.current.getTracks().forEach(track => track.stop());
            }
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, []);

    const startRecording = useCallback(() => {
        if (!mediaRecorder || mediaRecorder.state !== 'inactive') return;
        try {
            chunks.current = []; // Clear any existing chunks
            mediaRecorder.start();
            setIsRecording(true);
            setStatus('recording');
        } catch (error) {
            console.error('Error starting recording:', error);
            setStatus('failed');
        }
    }, [mediaRecorder]);

    const stopRecording = useCallback(() => {
        if (!mediaRecorder || mediaRecorder.state !== 'recording') return;
        try {
            mediaRecorder.stop();
            setIsRecording(false);
        } catch (error) {
            console.error('Error stopping recording:', error);
            setStatus('failed');
        }
    }, [mediaRecorder]);

    const getBase64Audio = useCallback(async () => {
        if (!audioUrl) return null;

        try {
            const response = await fetch(audioUrl);
            const blob = await response.blob();
            return new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result as string;
                    resolve(base64data.split(',')[1]);
                };
                reader.onerror = () => reject(reader.error);
                reader.readAsDataURL(blob);
            });
        } catch (error) {
            console.error('Error converting audio to base64:', error);
            return null;
        }
    }, [audioUrl]);

    return {
        isRecording,
        status,
        mediaBlobUrl: audioUrl,
        startRecording,
        stopRecording,
        getBase64Audio
    };
}; 