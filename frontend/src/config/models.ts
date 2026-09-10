export const CHATTERBOX_MODEL_TYPES = ['chatterbox-turbo', 'chatterbox-mtl', 'chatterbox'] as const

export type ChatterboxModelType = (typeof CHATTERBOX_MODEL_TYPES)[number]

export const MODEL_STORAGE_KEY = 'chatterbox-test-model-type'

export const MODEL_OPTIONS: Array<{
    value: ChatterboxModelType
    label: string
    description: string
}> = [
    {
        value: 'chatterbox-turbo',
        label: 'Turbo',
        description: 'New English model. Clone and narrate in English only.',
    },
    {
        value: 'chatterbox-mtl',
        label: 'Multilingual V3',
        description: '23-language model. Clone and narrate in the selected language.',
    },
    {
        value: 'chatterbox',
        label: 'Original',
        description: 'Legacy English backup. Old .npy profiles only work here.',
    },
]

export const CHATTERBOX_LANGUAGES: Array<{ code: string; name: string }> = [
    { code: 'ar', name: 'Arabic' },
    { code: 'zh', name: 'Chinese' },
    { code: 'da', name: 'Danish' },
    { code: 'nl', name: 'Dutch' },
    { code: 'en', name: 'English' },
    { code: 'fi', name: 'Finnish' },
    { code: 'fr', name: 'French' },
    { code: 'de', name: 'German' },
    { code: 'el', name: 'Greek' },
    { code: 'he', name: 'Hebrew' },
    { code: 'hi', name: 'Hindi' },
    { code: 'it', name: 'Italian' },
    { code: 'ja', name: 'Japanese' },
    { code: 'ko', name: 'Korean' },
    { code: 'ms', name: 'Malay' },
    { code: 'no', name: 'Norwegian' },
    { code: 'pl', name: 'Polish' },
    { code: 'pt', name: 'Portuguese' },
    { code: 'ru', name: 'Russian' },
    { code: 'es', name: 'Spanish' },
    { code: 'sw', name: 'Swahili' },
    { code: 'sv', name: 'Swedish' },
    { code: 'tr', name: 'Turkish' },
]

export function parseModelType(raw: unknown): ChatterboxModelType | undefined {
    if (typeof raw !== 'string') return undefined
    const value = raw.trim().toLowerCase()
    if (value === 'turbo' || value === 'chatterbox-turbo') return 'chatterbox-turbo'
    if (value === 'mtl' || value === 'chatterbox-mtl') return 'chatterbox-mtl'
    if (value === 'chatterbox' || value === 'original' || value === 'legacy') return 'chatterbox'
    return undefined
}

export function readStoredModelType(): ChatterboxModelType {
    if (typeof window === 'undefined') return 'chatterbox-turbo'
    return parseModelType(window.localStorage.getItem(MODEL_STORAGE_KEY)) ?? 'chatterbox-turbo'
}

export function storeModelType(modelType: ChatterboxModelType): void {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(MODEL_STORAGE_KEY, modelType)
}

export function isEnglishOnlyModel(modelType: ChatterboxModelType): boolean {
    return modelType === 'chatterbox-turbo' || modelType === 'chatterbox'
}

export function languagesForModel(modelType: ChatterboxModelType): Array<{ code: string; name: string }> {
    if (isEnglishOnlyModel(modelType)) {
        return CHATTERBOX_LANGUAGES.filter((language) => language.code === 'en')
    }
    return CHATTERBOX_LANGUAGES
}

export function modelTypeForLanguage(language: string): ChatterboxModelType {
    return language.trim().toLowerCase() === 'en' ? 'chatterbox-turbo' : 'chatterbox-mtl'
}

export function modelLabel(modelType: ChatterboxModelType): string {
    return MODEL_OPTIONS.find((option) => option.value === modelType)?.label ?? modelType
}
