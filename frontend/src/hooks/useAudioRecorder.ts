'use client';

import { useState, useCallback, useEffect } from 'react';

export const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);
    const [status, setStatus] = useState<string>('idle');

    useEffect(() => {
        if (typeof window === 'undefined') return;

        const initMediaRecorder = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const recorder = new MediaRecorder(stream);
                const chunks: BlobPart[] = [];

                recorder.ondataavailable = (e) => {
                    chunks.push(e.data);
                };

                recorder.onstop = () => {
                    const blob = new Blob(chunks, { type: 'audio/wav' });
                    const url = URL.createObjectURL(blob);
                    setAudioUrl(url);
                    setStatus('stopped');
                };

                setMediaRecorder(recorder);
            } catch (error) {
                console.error('Error initializing media recorder:', error);
                setStatus('failed');
            }
        };

        initMediaRecorder();
    }, []);

    const startRecording = useCallback(() => {
        if (!mediaRecorder) return;
        mediaRecorder.start();
        setIsRecording(true);
        setStatus('recording');
    }, [mediaRecorder]);

    const stopRecording = useCallback(() => {
        if (!mediaRecorder) return;
        mediaRecorder.stop();
        setIsRecording(false);
    }, [mediaRecorder]);

    const getBase64Audio = useCallback(async () => {
        if (!audioUrl) return null;

        try {
            const response = await fetch(audioUrl);
            const blob = await response.blob();
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result as string;
                    resolve(base64data.split(',')[1]);
                };
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