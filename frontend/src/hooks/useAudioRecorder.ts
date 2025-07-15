import { useState, useCallback } from 'react';
import { useReactMediaRecorder } from 'react-media-recorder';

export const useAudioRecorder = () => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

    const { status, startRecording, stopRecording, mediaBlobUrl } = useReactMediaRecorder({
        audio: true,
        video: false,
        blobPropertyBag: { type: "audio/wav" }
    });

    const handleStartRecording = useCallback(() => {
        setIsRecording(true);
        setAudioBlob(null);
        startRecording();
    }, [startRecording]);

    const handleStopRecording = useCallback(() => {
        setIsRecording(false);
        stopRecording();
    }, [stopRecording]);

    const getBase64Audio = useCallback(async (): Promise<string | null> => {
        if (!mediaBlobUrl) return null;

        try {
            const response = await fetch(mediaBlobUrl);
            const blob = await response.blob();
            setAudioBlob(blob);

            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result as string;
                    resolve(base64data.split(',')[1]); // Remove data URL prefix
                };
                reader.onerror = reject;
                reader.readAsDataURL(blob);
            });
        } catch (error) {
            console.error('Error converting audio to base64:', error);
            return null;
        }
    }, [mediaBlobUrl]);

    return {
        isRecording,
        status,
        audioBlob,
        mediaBlobUrl,
        startRecording: handleStartRecording,
        stopRecording: handleStopRecording,
        getBase64Audio
    };
}; 