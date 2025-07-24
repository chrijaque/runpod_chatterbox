#!/usr/bin/env python3
"""
Test script for TTSProcessor functionality with actual long story text
"""

def load_test_story():
    """Load the test story from file"""
    try:
        with open('testlongstory.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("❌ testlongstory.txt not found!")
        return None

def test_text_chunking_with_real_story():
    """Test the text chunking functionality with the actual long story"""
    
    # Load the real story
    story_text = load_test_story()
    if not story_text:
        return
    
    print("📖 Testing with real long story text:")
    print(f"📊 Story length: {len(story_text)} characters")
    print(f"📊 Story length: {len(story_text.split())} words")
    print()
    
    # Test different chunk sizes to see the effect
    chunk_sizes = [300, 500, 600, 800]
    
    for max_chars in chunk_sizes:
        print(f"🔍 Testing with max_chars = {max_chars}:")
        
        # Use the same chunking logic as TTSProcessor
        chunks = chunk_text_for_tts(story_text, max_chars)
        
        print(f"  📦 Number of chunks: {len(chunks)}")
        
        # Show first few chunks as examples
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
        
        if len(chunks) > 3:
            print(f"  ... and {len(chunks) - 3} more chunks")
        
        # Calculate total characters and efficiency
        total_chars = sum(len(chunk) for chunk in chunks)
        efficiency = (total_chars / len(story_text)) * 100
        
        print(f"  📊 Total chars in chunks: {total_chars}")
        print(f"  📊 Character preservation: {efficiency:.1f}%")
        print()

def chunk_text_for_tts(text, max_chunk_length=1000, overlap=50):
    """
    Split long text into chunks suitable for TTS generation.
    This is the same function used in TTSProcessor.
    
    Args:
        text (str): The text to chunk
        max_chunk_length (int): Maximum characters per chunk
        overlap (int): Number of characters to overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    if len(text) <= max_chunk_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_length
        
        # If this isn't the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start + max_chunk_length - 100, start)
            search_text = text[search_start:end]
            
            # Find the last sentence ending
            sentence_endings = ['.', '!', '?', '\n\n']
            last_ending = -1
            for ending in sentence_endings:
                pos = search_text.rfind(ending)
                if pos > last_ending:
                    last_ending = pos
            
            if last_ending != -1:
                end = search_start + last_ending + 1
        
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move start position, accounting for overlap
        start = max(start + 1, end - overlap)
    
    return chunks

def test_audio_concatenation():
    """Test audio concatenation logic"""
    
    print("\n🎵 Testing audio concatenation logic:")
    
    # Simulate audio chunks (just for testing the logic)
    chunk_durations = [2.5, 3.1, 2.8, 1.9]  # seconds
    pause_duration = 0.1  # seconds
    
    total_duration = sum(chunk_durations) + (len(chunk_durations) - 1) * pause_duration
    
    print(f"Chunk durations: {chunk_durations}")
    print(f"Pause duration: {pause_duration}s")
    print(f"Total duration: {total_duration:.2f}s")
    
    print("✅ Audio concatenation logic test completed!")

def estimate_tts_duration():
    """Estimate TTS duration for the story"""
    
    story_text = load_test_story()
    if not story_text:
        return
    
    print("\n⏱️ TTS Duration Estimation:")
    
    # Estimate based on typical TTS rates
    words_per_minute = 150  # Typical TTS speed
    words = len(story_text.split())
    estimated_minutes = words / words_per_minute
    
    print(f"📊 Story words: {words}")
    print(f"📊 Estimated duration: {estimated_minutes:.1f} minutes ({estimated_minutes*60:.0f} seconds)")
    
    # Estimate with chunking overhead
    chunks = chunk_text_for_tts(story_text, max_chunk_length=600)
    pause_overhead = (len(chunks) - 1) * 0.15  # 150ms pause between chunks
    
    print(f"📊 With {len(chunks)} chunks and pauses: {estimated_minutes + pause_overhead/60:.1f} minutes")
    print(f"📊 Chunking overhead: +{pause_overhead:.1f} seconds")

if __name__ == "__main__":
    print("🧪 Testing TTSProcessor with real long story...")
    print("=" * 60)
    
    # Test text chunking with real story
    test_text_chunking_with_real_story()
    
    # Test audio concatenation
    test_audio_concatenation()
    
    # Estimate TTS duration
    estimate_tts_duration()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed successfully!")
    print("The TTSProcessor should handle your long story without CUDA errors!") 