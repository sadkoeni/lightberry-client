import requests
import sys
import os
import io
from pydub import AudioSegment

def text_to_wav(api_key, text, voice_id=None, output_file='output.wav'):
    # Get a list of available voices
    voices_url = "https://api.elevenlabs.io/v1/voices"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    if not voice_id:
        # If no voice_id is provided, fetch available voices
        voices_response = requests.get(voices_url, headers=headers)
        
        if voices_response.status_code != 200:
            print(f"Failed to get voices: {voices_response.status_code} - {voices_response.text}")
            return
            
        voices = voices_response.json().get("voices", [])
        if not voices:
            print("No voices found in your account")
            return
            
        # Use the first voice available if no voice_id is provided
        voice_id = voices[0]["voice_id"]
    
    # Text to speech API endpoint with voice_id
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    # Prepare the data payload
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    # Specify a high-quality MP3 format (ElevenLabs doesn't directly support WAV)
    params = {
        "output_format": "mp3_44100_128"  # High quality MP3 format
    }
    
    print(f"Requesting audio from ElevenLabs API for text: '{text}'")
    response = requests.post(url, headers=headers, json=data, params=params)
    
    if response.status_code == 200:
        content_type = response.headers.get('Content-Type', '')
        print(f"Received response with Content-Type: {content_type}")
        
        # Use a temporary MP3 file in a different location
        temp_dir = os.path.join(os.path.dirname(output_file), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_mp3_file = os.path.join(temp_dir, f"{os.path.basename(output_file).replace('.wav', '.mp3')}")
        
        with open(temp_mp3_file, 'wb') as f:
            f.write(response.content)
        print(f"Saved temporary MP3 audio to {temp_mp3_file}")
        
        # Convert MP3 to WAV if the output file has .wav extension
        if output_file.lower().endswith('.wav'):
            try:
                print(f"Converting MP3 to WAV: {temp_mp3_file} -> {output_file}")
                # Load MP3 file
                audio = AudioSegment.from_mp3(temp_mp3_file)
                # Export as WAV
                audio.export(output_file, format="wav")
                print(f"Successfully converted and saved to {output_file}")
                
                # Clean up the temporary MP3 file
                os.remove(temp_mp3_file)
                print(f"Removed temporary MP3 file: {temp_mp3_file}")
                
                # Return the WAV file path
                return output_file
            except Exception as e:
                print(f"Conversion failed: {e}")
                print(f"Using MP3 file instead")
                
                # Move the MP3 file to the final location
                final_mp3_file = output_file.replace('.wav', '.mp3')
                import shutil
                shutil.move(temp_mp3_file, final_mp3_file)
                print(f"Moved MP3 file to {final_mp3_file}")
                
                # Return the MP3 file path
                return final_mp3_file
        else:
            # Just move the MP3 file to the final location
            import shutil
            shutil.move(temp_mp3_file, output_file)
            print(f"Moved MP3 file to {output_file}")
            return output_file
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python create_audio_instructions.py <API_KEY> <TEXT> [VOICE_ID]")
        sys.exit(1)

    api_key = sys.argv[1]
    text = sys.argv[2]
    voice_id = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Default output file in function_calls directory with timestamp
    from datetime import datetime
    output_dir = "function_calls"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"instruction_{timestamp}.wav")
    
    # Make sure FFmpeg is installed
    try:
        import subprocess
        ffmpeg_check = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if ffmpeg_check.returncode != 0:
            print("Warning: FFmpeg is not installed or not in PATH. Audio conversion may fail.")
            print("Try: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
    except Exception:
        print("Warning: Could not check for FFmpeg. If audio conversion fails, install FFmpeg.")
    
    result_file = text_to_wav(api_key, text, voice_id, output_file)
    if result_file:
        print(f"\nFinal audio file saved to: {result_file}")
        print("The audio client will automatically detect and process this file.")