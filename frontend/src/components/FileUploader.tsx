'use client';

import { ChangeEvent, useCallback } from 'react';
import { ArrowUpTrayIcon } from '@heroicons/react/24/solid';

interface FileUploaderProps {
    onFileSelect: (base64Audio: string, format: string) => void;
}

export const FileUploader = ({ onFileSelect }: FileUploaderProps) => {
    const handleFileChange = useCallback(
        async (event: ChangeEvent<HTMLInputElement>) => {
            console.log('File input change detected');
            const file = event.target.files?.[0];
            
            if (!file) {
                console.log('No file selected');
                return;
            }

            console.log('File selected:', {
                name: file.name,
                type: file.type,
                size: file.size
            });

            // Validate file type
            if (!file.type.startsWith('audio/')) {
                console.error('Invalid file type. Please upload an audio file.');
                return;
            }

            // Validate file size (10MB max)
            if (file.size > 10 * 1024 * 1024) {
                console.error('File is too large. Maximum size is 10MB.');
                return;
            }

            try {
                const reader = new FileReader();
                
                reader.onerror = () => {
                    console.error('Error reading file:', reader.error);
                };

                reader.onloadstart = () => {
                    console.log('Started reading file');
                };

                reader.onloadend = () => {
                    console.log('Finished reading file');
                    const base64data = reader.result as string;
                    const base64Audio = base64data.split(',')[1];
                    const format = file.name.split('.').pop()?.toLowerCase() || 'wav';
                    console.log('Calling onFileSelect with format:', format);
                    onFileSelect(base64Audio, format);
                };

                reader.readAsDataURL(file);
            } catch (error) {
                console.error('Error processing file:', error);
            }
        },
        [onFileSelect]
    );

    return (
        <div className="bg-gray-50 rounded-lg p-4">
            <label className="flex flex-col items-center justify-center w-full h-20 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-white hover:bg-gray-50 transition-colors">
                <div className="flex items-center space-x-2">
                    <ArrowUpTrayIcon className="h-5 w-5 text-gray-400" />
                    <span className="text-sm text-gray-600">
                        <span className="text-indigo-600 font-medium">Click to upload</span> or drag and drop
                    </span>
                </div>
                <p className="mt-1 text-xs text-gray-500">WAV, MP3, or M4A (MAX. 10MB)</p>
                <input
                    type="file"
                    className="hidden"
                    accept="audio/wav,audio/mp3,audio/mp4,audio/x-m4a,audio/*"
                    onChange={handleFileChange}
                    onClick={(e) => {
                        // Reset the input value to allow selecting the same file again
                        (e.target as HTMLInputElement).value = '';
                    }}
                />
            </label>
        </div>
    );
}; 