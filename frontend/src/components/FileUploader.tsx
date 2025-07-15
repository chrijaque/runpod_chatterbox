import { ChangeEvent, useCallback } from 'react';
import { ArrowUpTrayIcon } from '@heroicons/react/24/solid';

interface FileUploaderProps {
    onFileSelect: (base64Audio: string, format: string) => void;
}

export const FileUploader = ({ onFileSelect }: FileUploaderProps) => {
    const handleFileChange = useCallback(
        async (event: ChangeEvent<HTMLInputElement>) => {
            const file = event.target.files?.[0];
            if (!file) return;

            try {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64data = reader.result as string;
                    const base64Audio = base64data.split(',')[1];
                    const format = file.name.split('.').pop() || 'wav';
                    onFileSelect(base64Audio, format);
                };
                reader.readAsDataURL(file);
            } catch (error) {
                console.error('Error reading file:', error);
            }
        },
        [onFileSelect]
    );

    return (
        <div className="w-full max-w-2xl mx-auto">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <ArrowUpTrayIcon className="w-8 h-8 mb-4 text-gray-500" />
                    <p className="mb-2 text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span> or
                        drag and drop
                    </p>
                    <p className="text-xs text-gray-500">
                        WAV, MP3, or M4A (MAX. 10MB)
                    </p>
                </div>
                <input
                    type="file"
                    className="hidden"
                    accept="audio/*"
                    onChange={handleFileChange}
                />
            </label>
        </div>
    );
}; 