'use client';

import { useState } from 'react';

export type ModelType = 'chatterbox' | 'higgs';

interface ModelToggleProps {
    modelType: ModelType;
    onModelChange: (model: ModelType) => void;
    disabled?: boolean;
}

export function ModelToggle({ modelType, onModelChange, disabled = false }: ModelToggleProps) {
    const [isHovered, setIsHovered] = useState(false);

    const models = [
        {
            id: 'chatterbox' as ModelType,
            name: 'ChatterboxTTS',
            description: 'Fast, efficient voice cloning',
            color: 'bg-blue-500',
            hoverColor: 'bg-blue-600'
        },
        {
            id: 'higgs' as ModelType,
            name: 'Higgs Audio',
            description: 'Advanced expressive TTS',
            color: 'bg-purple-500',
            hoverColor: 'bg-purple-600'
        }
    ];

    return (
        <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
                AI Model Selection
            </label>
            <div className="flex space-x-2">
                {models.map((model) => (
                    <button
                        key={model.id}
                        onClick={() => onModelChange(model.id)}
                        disabled={disabled}
                        onMouseEnter={() => setIsHovered(true)}
                        onMouseLeave={() => setIsHovered(false)}
                        className={`
                            flex-1 px-4 py-3 rounded-lg border-2 transition-all duration-200
                            ${modelType === model.id 
                                ? `${model.color} text-white border-transparent shadow-lg` 
                                : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'
                            }
                            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                            ${!disabled && modelType !== model.id ? 'hover:shadow-md' : ''}
                        `}
                    >
                        <div className="text-center">
                            <div className={`font-semibold text-sm ${modelType === model.id ? 'text-white' : 'text-gray-900'}`}>
                                {model.name}
                            </div>
                            <div className={`text-xs mt-1 ${modelType === model.id ? 'text-blue-100' : 'text-gray-500'}`}>
                                {model.description}
                            </div>
                        </div>
                    </button>
                ))}
            </div>
            
            {/* Model info tooltip */}
            {isHovered && (
                <div className="mt-2 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="text-xs text-gray-600">
                        <strong>ChatterboxTTS:</strong> Optimized for speed and efficiency, great for real-time applications.
                        <br />
                        <strong>Higgs Audio:</strong> Advanced model with better expressiveness and long-form capabilities.
                    </div>
                </div>
            )}
        </div>
    );
} 