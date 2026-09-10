'use client';

import {
    MODEL_OPTIONS,
    languagesForModel,
    type ChatterboxModelType,
} from '@/config/models';

interface ModelPickerProps {
    modelType: ChatterboxModelType
    language: string
    onModelTypeChange: (modelType: ChatterboxModelType) => void
    onLanguageChange: (language: string) => void
}

export function ModelPicker({
    modelType,
    language,
    onModelTypeChange,
    onLanguageChange,
}: ModelPickerProps) {
    const languages = languagesForModel(modelType)
    const selected = MODEL_OPTIONS.find((option) => option.value === modelType)

    return (
        <div className="space-y-4">
            <div>
                <label htmlFor="model-type" className="form-label">
                    Model
                </label>
                <select
                    id="model-type"
                    value={modelType}
                    onChange={(event) => onModelTypeChange(event.target.value as ChatterboxModelType)}
                    className="form-input"
                >
                    {MODEL_OPTIONS.map((option) => (
                        <option key={option.value} value={option.value}>
                            {option.label}
                        </option>
                    ))}
                </select>
                {selected && (
                    <p className="mt-2 text-sm text-blue-800 bg-blue-50 border border-blue-200 rounded-lg p-3">
                        {selected.description} Profiles are not interchangeable across models.
                    </p>
                )}
            </div>
            <div>
                <label htmlFor="model-language" className="form-label">
                    Language
                </label>
                <select
                    id="model-language"
                    value={language}
                    onChange={(event) => onLanguageChange(event.target.value)}
                    className="form-input"
                >
                    {languages.map((item) => (
                        <option key={item.code} value={item.code}>
                            {item.name}
                        </option>
                    ))}
                </select>
            </div>
        </div>
    )
}
