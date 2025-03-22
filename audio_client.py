import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Conditional imports based on platform
if sys.platform == "darwin":
    import requests
elif sys.platform == "linux":
    # Removing GPIO import since we won't be using the button functionality
    pass

import pyaudio
import wave
import websockets
import asyncio
import json
import base64
from pathlib import Path
from datetime import datetime
import sounddevice as sd
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from io import BytesIO
import threading
import tempfile
import queue
import time
from queue import Queue
import configparser
import argparse
from scipy import signal
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import uuid
import glob
import pvporcupine
from pvrecorder import PvRecorder
import platform
import struct
import socket  # Add socket module for IPC

# Constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
SILENCE_THRESHOLD = 0.3  # Updated from 0.01 to 0.3 as requested
SILENCE_DURATION = 1.2
SILENCE_CHUNKS = int(SILENCE_DURATION * RATE / CHUNK)
GOODBYE_PHRASES = ["goodbye", "bye", "exit", "quit", "stop"]
CHUNK_SIZE = 1024
VOLUME_SCALE = 0.9
MAX_VOLUME = 0.95
MIN_RMS_THRESHOLD = 0.001
VOLUME_SMOOTHING = 0.7
HEARTBEAT_INTERVAL = 5
PICOVOICE_ACCESS_KEY = os.environ.get('PICOVOICE_API_KEY', '')  # Add this to your .env file

# Server details
#SERVER_IP = os.environ.get('AUDIO_SERVER_IP', 'localhost' if sys.platform == "darwin" else 'MacBook-Pro.local')
SERVER_IP = "34.116.196.182"
SERVER_PORT = int(os.environ.get('AUDIO_SERVER_PORT', '8765'))
SERVER_URL = f"ws://{SERVER_IP}:{SERVER_PORT}"

# Platform-specific constants
if sys.platform == "linux":
    os.environ['AUDIODRIVER'] = 'pulse'

# Common functions and classes

def get_audio_rms(audio_data):
    """Calculate the Root Mean Square (RMS) of audio data"""
    return np.sqrt(np.mean(np.square(audio_data)))

def normalize_audio(audio_data, target_rms):
    """Normalize audio data to match target RMS"""
    current_rms = get_audio_rms(audio_data)
    if current_rms > 0:  # Avoid division by zero
        adjustment_factor = target_rms / current_rms
        return audio_data * adjustment_factor
    return audio_data

async def handle_status_updates(status_msg):
    status = status_msg.get('status')
    if status == 'processing':
        print("Server is processing the audio...")
    elif status == 'transcribing':
        print("Transcribing audio...")
    elif status == 'getting_ai_response':
        print("Getting AI response...")
    elif status == 'generating_speech':
        print("Generating speech...")
    elif status == 'error':
        print(f"Error: {status_msg.get('error')}")

def check_for_silence(audio_data, threshold=SILENCE_THRESHOLD):
    return np.max(np.abs(audio_data)) < threshold

def contains_goodbye(transcript):
    return any(phrase in transcript.lower() for phrase in GOODBYE_PHRASES)

def list_audio_devices():
    """List all available audio devices to help identify correct device indices"""
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    for i, device in enumerate(devices):
        input_channels = device['max_input_channels']
        output_channels = device['max_output_channels']
        name = device['name']
        if input_channels > 0:
            print(f"Input Device {i}: {name} (Inputs: {input_channels})")
        if output_channels > 0:
            print(f"Output Device {i}: {name} (Outputs: {output_channels})")
    print()

async def wait_for_enter():
    """Wait for the user to press Enter"""
    print("\n>>> Press ENTER to start recording...")
    
    # Use asyncio to read from stdin without blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input)
    return True

class FunctionCallHandler:
    def __init__(self, queue_reference, timeout=10):
        self.function_queue = queue.Queue()
        self.audio_queue = queue_reference
        self.is_waiting = False
        self.timeout = timeout  # Timeout in seconds
        self.last_activity_time = None
        self.watch_directory = "function_calls"  # Directory to watch for audio files
        
        # Create the watch directory if it doesn't exist
        os.makedirs(self.watch_directory, exist_ok=True)
        print(f"Watching for audio files (WAV/MP3) in '{self.watch_directory}' directory")
        
        # Store processed files to avoid reprocessing
        self.processed_files = set()
        # Scan for existing files
        for file in os.listdir(self.watch_directory):
            if file.lower().endswith('.wav') or file.lower().endswith('.mp3'):
                self.processed_files.add(file)
                print(f"Registered existing file: {file}")
                
        print(f"Found {len(self.processed_files)} existing audio files to ignore")
        
    def process_wav_file(self, wav_file_path):
        """Process a WAV file and add its chunks to the audio queue"""
        try:
            print(f"Processing audio file: {wav_file_path}")
            if not os.path.exists(wav_file_path):
                print(f"Error: Audio file not found: {wav_file_path}")
                return False
            
            # Check the actual file format regardless of extension
            file_format = None
            try:
                import magic  # Try to use python-magic if available
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(wav_file_path)
                if "audio/mpeg" in file_type:
                    file_format = "mp3"
                elif "audio/wav" in file_type or "audio/x-wav" in file_type:
                    file_format = "wav"
                print(f"Detected file format: {file_format} (MIME: {file_type})")
            except ImportError:
                # Fallback: Try to infer from file header
                with open(wav_file_path, 'rb') as f:
                    header = f.read(4)
                    if header.startswith(b'RIFF'):
                        file_format = "wav"
                    elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                        file_format = "mp3"
                print(f"Inferred file format from header: {file_format}")
                
            # Load the audio file based on detected format
            if file_format == "mp3" or (file_format is None and wav_file_path.lower().endswith('.mp3')):
                print("Loading MP3 file...")
                # Use pydub to load MP3
                audio_segment = AudioSegment.from_mp3(wav_file_path)
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                if audio_segment.channels == 2:  # Convert stereo to mono
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                # Normalize to [-1.0, 1.0]
                samples = samples / (2**15)
                sample_rate = audio_segment.frame_rate
                print(f"Loaded MP3: {len(samples)} samples, {sample_rate}Hz, {audio_segment.channels} channels")
            else:
                # Default to WAV loading
                print("Loading WAV file...")
                audio_data, sample_rate = sf.read(wav_file_path)
                
                # Ensure audio data is in float32 format and mono
                if audio_data.ndim > 1:  # Check if stereo
                    audio_data = audio_data[:, 0]  # Take first channel
                
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                samples = audio_data
                print(f"Loaded WAV: {len(samples)} samples, {sample_rate}Hz")
            
            # Normalize audio (regardless of source format)
            max_amplitude = np.max(np.abs(samples))
            if max_amplitude > 0:
                samples = samples / max_amplitude * 0.9  # Normalize with headroom
                print(f"Normalized audio with max amplitude {max_amplitude}")
            
            # Ensure sample rate is what the server expects
            if sample_rate != RATE:
                print(f"Resampling from {sample_rate}Hz to {RATE}Hz")
                # Resample to expected rate
                resampled_count = int(len(samples) * RATE / sample_rate)
                samples = signal.resample(samples, resampled_count)
                print(f"Resampled to {len(samples)} samples at {RATE}Hz")
            
            # Split into chunks and add to the queue
            chunk_size = int(RATE * CHUNK / 1000)  # Convert to samples
            total_chunks = 0
            
            for i in range(0, len(samples), chunk_size):
                chunk = samples[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # Pad the last chunk if needed
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
                
                # Add to the audio queue
                self.audio_queue.put(chunk)
                total_chunks += 1
            
            # Add a final termination signal
            # This is an empty chunk with a special marker to indicate end of recording
            print(">>> Adding termination signal (None) to audio queue to signal end of audio file")
            self.audio_queue.put(None)  # Special marker for final chunk
            
            print(f"Added {total_chunks} chunks from audio file to audio queue (plus termination signal)")
            self.last_activity_time = time.time()
            return True
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_for_new_wav_files(self):
        """Check the watch directory for new audio files (WAV or MP3)"""
        try:
            new_files = []
            base_names = set()  # Track base filenames without extension
            
            # First pass: collect base names of already processed files
            for processed_file in self.processed_files:
                base_name = os.path.splitext(processed_file)[0]
                base_names.add(base_name)
            
            # Second pass: find new files while avoiding duplicates
            for file in os.listdir(self.watch_directory):
                base_name = os.path.splitext(file)[0]
                extension = os.path.splitext(file)[1].lower()
                
                # Check if it's an audio file we support
                if extension not in ['.wav', '.mp3']:
                    continue
                    
                # Skip if we've already processed this file
                if file in self.processed_files:
                    continue
                    
                # Skip if we've processed a file with the same base name
                if base_name in base_names:
                    print(f"Skipping {file} as a file with the same base name was already processed")
                    self.processed_files.add(file)  # Mark it as processed
                    continue
                
                # Found a new audio file
                full_path = os.path.join(self.watch_directory, file)
                new_files.append(full_path)
                self.processed_files.add(file)
                base_names.add(base_name)  # Add to processed base names
            
            if new_files:
                print(f"Found {len(new_files)} new audio file(s): {new_files}")
                # Process the first new file found
                if self.process_wav_file(new_files[0]):
                    # Optionally move or delete the processed file
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking for new audio files: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_function_call(self, wav_file_path):
        """Add a function call to the queue"""
        self.function_queue.put(wav_file_path)
        print(f"Added function call with WAV file: {wav_file_path}")
        
    async def wait_for_function_call(self):
        """Wait for a function call or timeout"""
        self.is_waiting = True
        print("\nWaiting for function call with WAV file...")
        print(f"You can place WAV files in the '{self.watch_directory}' directory to trigger processing")
        
        start_time = time.time()
        while self.is_waiting:
            try:
                # First check if there's a direct function call (non-blocking)
                try:
                    wav_file_path = self.function_queue.get_nowait()
                    print(f"Received direct function call with WAV file: {wav_file_path}")
                    success = self.process_wav_file(wav_file_path)
                    if success:
                        self.is_waiting = False
                        return True
                except queue.Empty:
                    # No direct function call, check the watch directory
                    if self.check_for_new_wav_files():
                        self.is_waiting = False
                        return True
                
                # Brief sleep to prevent CPU hogging
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Error waiting for function call: {e}")
                import traceback
                traceback.print_exc()
                
        return False
        
    def check_timeout(self):
        """Check if we've timed out since the last activity"""
        if self.last_activity_time is None:
            return False
            
        elapsed = time.time() - self.last_activity_time
        return elapsed > self.timeout
    
    def stop_waiting(self):
        """Stop waiting for function calls"""
        self.is_waiting = False

# Platform-specific implementations
if sys.platform == "darwin":
    class AudioStreamPlayer:
        def __init__(self):
            self.buffer = BytesIO()
            self.is_playing = False
            self.stream = None
            self.py_audio = pyaudio.PyAudio()
            self.target_rms = None
            self.playback_thread = None

        async def process_chunk(self, chunk_data, target_rms=None):
            try:
                print("Processing new audio chunk...")
                if target_rms is not None:
                    if self.target_rms is None:
                        self.target_rms = target_rms
                    else:
                        self.target_rms = (VOLUME_SMOOTHING * self.target_rms + 
                                         (1 - VOLUME_SMOOTHING) * target_rms)
                
                chunk = base64.b64decode(chunk_data)
                self.buffer.write(chunk)
                print(f"Added {len(chunk)} bytes to buffer")
                    
            except Exception as e:
                print(f"Error processing chunk: {e}")

        def play_audio(self):
            try:
                print("Starting audio playback...")
                self.buffer.seek(0)
                
                audio = AudioSegment.from_mp3(self.buffer)
                print(f"Converted MP3 to AudioSegment: {audio.duration_seconds:.2f} seconds")
                
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2)).mean(axis=1)
                
                samples = samples / (2**15)
                
                if self.target_rms and self.target_rms > MIN_RMS_THRESHOLD:
                    current_rms = np.sqrt(np.mean(np.square(samples)))
                    if current_rms > MIN_RMS_THRESHOLD:
                        adjustment = self.target_rms / current_rms
                        adjustment *= VOLUME_SCALE
                        adjustment = min(adjustment, MAX_VOLUME / current_rms)
                        samples = samples * adjustment
                        print(f"Applied volume adjustment: {adjustment:.2f}")
                
                self.is_playing = True
                
                def callback(outdata, frames, time, status):
                    if status:
                        print(f"Playback status: {status}")
                
                print("Playing audio...")
                self.playback_thread = threading.Thread(
                    target=self._play_audio_thread,
                    args=(samples, audio.frame_rate)
                )
                self.playback_thread.daemon = True
                self.playback_thread.start()
                
            except Exception as e:
                print(f"Error starting audio playback: {e}")
                import traceback
                traceback.print_exc()
                self.stop()
        
        def _play_audio_thread(self, samples, sample_rate):
            try:
                device_info = sd.query_devices(kind='output')
                latency = 0.3
                
                sd.play(
                    samples, 
                    sample_rate,
                    blocksize=4096,
                    latency=latency
                )
                
                duration = len(samples) / sample_rate
                print(f"Expected playback duration: {duration:.2f} seconds")
                
                start_time = time.time()
                max_wait_time = duration + 2.0
                
                while sd.get_stream().active and (time.time() - start_time) < max_wait_time:
                    time.sleep(0.1)
                
                if sd.get_stream() and sd.get_stream().active:
                    print("Playback timed out, forcing stop")
                    sd.stop()
                else:
                    print("Finished playing audio")
                
            except Exception as e:
                print(f"Error in audio playback thread: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.is_playing = False
                self.stop()

        def stop(self):
            try:
                sd.stop()
            except Exception as e:
                print(f"Error stopping sound device: {e}")
                
            self.buffer = BytesIO()
            self.is_playing = False
            print("Stopped audio player")

        def __del__(self):
            self.stop()
            self.py_audio.terminate()

elif sys.platform == "linux":
    # Removing ButtonMonitor class as requested - we'll use wake word detection instead
    pass  # We'll use the same wake word functionality for Linux as for Mac



def register_remote_device(device_auth, server_url, admin_username, admin_password):
    """Register device with remote server using API endpoint"""
    try:
        # Extract server base URL from WebSocket URL
        if server_url.startswith('ws://'):
            base_url = server_url.replace('ws://', 'http://')
        elif server_url.startswith('wss://'):
            base_url = server_url.replace('wss://', 'https://')
        else:
            base_url = f"http://{server_url.split(':')[0]}"
        
        if ':' in base_url:
            # Remove WebSocket port if present
            base_parts = base_url.split(':')
            if len(base_parts) > 2:
                # Has protocol and port
                base_url = f"{base_parts[0]}:{base_parts[1]}"
            else:
                # Just has port
                base_url = base_parts[0]
        
        # First try the REST API endpoint if available
        try:
            # Use the same port as the WebSocket server
            register_url = f"{base_url}/api/register_device"
            
            # Read public key
            public_key_path = device_auth.keys_dir / 'public_key.pem'
            with open(public_key_path, 'r') as f:
                public_key = f.read()
            
            # Prepare registration data
            data = {
                'device_id': device_auth.device_id,
                'public_key': public_key
            }
            
            print(f"Attempting to register device with remote server at {register_url}")
            
            # Add timeout to the request and include basic auth
            response = requests.post(
                register_url, 
                json=data, 
                timeout=5,  # Use a shorter timeout
                auth=(admin_username, admin_password)
            )
            
            if response.status_code == 200:
                print("Device registered successfully with remote server!")
                return True
            elif response.status_code == 401:
                print("Authentication failed. Please check your admin credentials.")
                return False
            else:
                print(f"Failed to register device: {response.text}")
                # Fall through to websocket registration
        except requests.exceptions.Timeout:
            print("REST API registration request timed out.")
            # Fall through to websocket registration
        except requests.exceptions.ConnectionError:
            print("Could not connect to REST API. Trying websocket registration...")
            # Fall through to websocket registration
        
        # If REST API failed, try WebSocket registration as fallback
        print("Attempting registration via WebSocket...")
        return register_via_websocket(device_auth, server_url, admin_username, admin_password)
            
    except Exception as e:
        print(f"Error in registration process: {e}")
        import traceback
        traceback.print_exc()
        return False

async def async_register_via_websocket(device_auth, server_url, admin_username, admin_password):
    """Register device with remote server using WebSocket"""
    try:
        print(f"Connecting to WebSocket server at {server_url} for registration...")
        
        # Read public key
        public_key_path = device_auth.keys_dir / 'public_key.pem'
        with open(public_key_path, 'r') as f:
            public_key = f.read()
        
        # Connect to WebSocket
        async with websockets.connect(server_url) as websocket:
            # Send registration message
            registration_data = {
                "type": "register_device",
                "device_id": device_auth.device_id,
                "public_key": public_key,
                "admin_username": admin_username,
                "admin_password": admin_password
            }
            
            await websocket.send(json.dumps(registration_data))
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if data.get("status") == "success":
                    print("Device registered successfully via WebSocket!")
                    return True
                else:
                    print(f"WebSocket registration failed: {data.get('message', 'Unknown error')}")
                    return False
                    
            except asyncio.TimeoutError:
                print("WebSocket registration timed out")
                return False
                
    except Exception as e:
        print(f"WebSocket registration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def register_via_websocket(device_auth, server_url, admin_username, admin_password):
    """Synchronous wrapper for async WebSocket registration"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            async_register_via_websocket(device_auth, server_url, admin_username, admin_password)
        )
        return result
    except Exception as e:
        print(f"Error in WebSocket registration wrapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def register_local_device(device_auth):
    """Register the local device in the database to bypass authentication"""
    try:
        # Add the server directory to the path to import the database module
        server_dir = Path(__file__).parent / "server"
        if server_dir.exists():
            sys.path.insert(0, str(server_dir))
        else:
            # Try parent directory
            server_dir = Path(__file__).parent.parent / "server"
            if server_dir.exists():
                sys.path.insert(0, str(server_dir))
            else:
                print("Warning: Could not find server directory")
                return False
        
        # Import database module - try to import from device_database first
        try:
            from device_database import init_db, Device
            from datetime import datetime
            print("Successfully imported from device_database")
        except ImportError:
            try:
                # Fallback to compatibility layer
                from database import init_db, Device
                from datetime import datetime
                print("Successfully imported from compatibility layer (database)")
            except ImportError:
                print("Warning: Could not import database module")
                return False
        
        # Initialize database
        db = init_db()
        
        # Read public key
        public_key_path = device_auth.keys_dir / 'public_key.pem'
        with open(public_key_path, 'r') as f:
            public_key = f.read()
        
        # Check if device already exists
        device = db.query(Device).filter(Device.device_id == device_auth.device_id).first()
        if device:
            print(f"Device {device_auth.device_id} already exists, updating public key...")
            device.public_key = public_key
            device.last_seen = datetime.utcnow()
        else:
            print(f"Registering new device {device_auth.device_id}...")
            # Create with proper JSON settings object for the updated schema
            device = Device(
                device_id=device_auth.device_id,
                public_key=public_key,
                settings={
                    "llm_provider": "openai", 
                    "model": "gpt-4",
                    "voice_provider": "elevenlabs",
                    "voice_id": "21m00Tcm4TlvDq8ikWAM"
                },
                last_seen=datetime.utcnow()
            )
            db.add(device)
        
        db.commit()
        print("Device registered successfully!")
        return True
    except Exception as e:
        print(f"Error registering device: {e}")
        import traceback
        traceback.print_exc()
        return False


def play_audio(audio_data, sample_rate=RATE):
    """Play audio data with error handling"""
    try:
        # Ensure audio data is in the correct format
        if not isinstance(audio_data, np.ndarray):
            print("Warning: Audio data is not a numpy array")
            return
        
        # Ensure audio data is float32 and in the range [-1.0, 1.0]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Clip to prevent distortion
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Play the audio
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")
        import traceback
        traceback.print_exc()

class WakeWordDetector:
    def __init__(self, access_key, model_path=None):
        self.access_key = access_key
        self.model_path = model_path
        self.porcupine = None
        self.recorder = None
        self.is_running = False
        
        # Determine platform architecture
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        print(f"Platform: {self.platform}, Architecture: {self.architecture}")
        
        # Check if model is compatible
        if self.model_path:
            if "raspberry" in self.model_path.lower() and not (self.platform == "linux" and "arm" in self.architecture):
                print("Warning: Using Raspberry Pi model on non-ARM platform may not work")
            
    def initialize(self):
        try:
            # Initialize Porcupine with custom keyword
            if self.model_path:
                print(f"Initializing with custom wake word model: {self.model_path}")
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.model_path]
                )
            else:
                # Fallback to default wake word
                print("No custom model provided, using default 'computer' wake word")
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=['computer']
                )
            
            # Initialize audio recorder
            self.recorder = PvRecorder(
                device_index=-1,  # default audio device
                frame_length=self.porcupine.frame_length
            )
            
            # List available audio devices using sounddevice
            devices = sd.query_devices()
            print("\nAvailable audio devices:")
            for i, device in enumerate(devices):
                input_channels = device['max_input_channels']
                output_channels = device['max_output_channels']
                name = device['name']
                if input_channels > 0:
                    print(f"Input Device {i}: {name} (Inputs: {input_channels})")
                if output_channels > 0:
                    print(f"Output Device {i}: {name} (Outputs: {output_channels})")
            
            print("\nWake word detector initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing wake word detector: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def listen_for_wake_word(self):
        """Listen for wake word and return True when detected"""
        if not self.porcupine or not self.recorder:
            if not self.initialize():
                return False
        
        try:
            print("\nListening for wake word...")
            self.recorder.start()
            self.is_running = True
            
            while self.is_running:
                pcm = self.recorder.read()
                result = self.porcupine.process(pcm)
                
                if result >= 0:
                    print("Wake word detected!")
                    return True
                
                # Small sleep to prevent CPU overuse
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if self.recorder.is_recording:
                self.recorder.stop()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.recorder and self.recorder.is_recording:
            self.recorder.stop()
        if self.recorder:
            self.recorder.delete()
        if self.porcupine:
            self.porcupine.delete()

class AudioRecorder:
    def __init__(self, debug_mode=False, test_audio_dir='test_audio', local_mode=False, wake_word_path=None, function_mode=False):
        self.recording = False
        self.audio_buffer = []
        self.silence_counter = 0
        self.websocket = None
        self.send_interval = 0.1  # Send every 100ms
        self.queue = Queue()
        self.loop = None
        self.complete_audio = []
        self.local_mode = local_mode
        self.silence_detected = False
        self.audio_player = AudioBuffer()
        self.manual_stop = False
        self.wake_word_path = wake_word_path  # Add wake word path
        self.devices_shown = False  # Add flag to track if devices have been shown
        self.enable_manual_stop = False  # Disable manual stop by default since silence detection is primary
        self.current_stop_task = None  # Track current stop task
        self.function_mode = function_mode  # Add function mode flag
        self.function_handler = None  # Will be initialized in record_and_process
        self.last_activity_time = time.time()  # Track last activity time for timeout
        self.wake_timeout = 10  # Timeout in seconds for wake word detection
        
        # Debug mode flag and settings
        self.debug_mode = debug_mode
        self.test_audio_dir = test_audio_dir
        self.available_test_files = []
        
        if self.debug_mode:
            self._load_test_audio_files()
            print(f"Debug mode enabled. Using test audio files from {test_audio_dir}")
        else:
            # Update audio parameters for better quality
            # Find built-in microphone device index
            devices = sd.query_devices()
            self.input_device = None
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    if 'Built-in' in device['name'] or 'MacBook' in device['name']:
                        self.input_device = i
                        print(f"Selected built-in microphone: {device['name']}")
                        break
            
            self.channels = 1  # Mono
            self.sample_rate = 16000  # Match server's expected rate
            self.dtype = np.float32
            
            # Adjust silence chunks
            global SILENCE_CHUNKS
            SILENCE_CHUNKS = int(SILENCE_DURATION * self.sample_rate / CHUNK)
        
        # Add directory for local recordings
        self.recordings_dir = Path("local_recordings")
        self.recordings_dir.mkdir(exist_ok=True)
        print(f"Saving local recordings to {self.recordings_dir}")
        
        # Track recording start time for filename
        self.recording_start_time = None
        
        # Track minimum noise level
        self.min_noise_level = float('inf')

    def _load_test_audio_files(self):
        """Load available test audio files"""
        os.makedirs(self.test_audio_dir, exist_ok=True)
        wav_files = glob.glob(os.path.join(self.test_audio_dir, "*.wav"))
        mp3_files = glob.glob(os.path.join(self.test_audio_dir, "*.mp3"))
        self.available_test_files = sorted(wav_files + mp3_files)
        
        if not self.available_test_files:
            print(f"WARNING: No test audio files found in {self.test_audio_dir}")
        else:
            print(f"Found {len(self.available_test_files)} test audio files:")
            for i, file in enumerate(self.available_test_files):
                print(f"  {i+1}. {os.path.basename(file)}")

    def record_audio_blocks(self):
        """Non-callback blocking recording approach with improved audio quality"""
        try:
            # Set recording start time for filename
            self.recording_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print("Starting audio recording - please speak into the microphone...")
            print("Recording will stop automatically after silence or when you press Enter again.")
            
            # Only show device list on first recording
            if not self.devices_shown:
                print("Available audio devices:")
                list_audio_devices()  # Show available devices
                self.devices_shown = True
            
            with sd.InputStream(
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=CHUNK_SIZE,
                dtype=self.dtype,
                latency='low',
                device=self.input_device
            ) as stream:
                silence_count = 0
                raw_audio = []  # Store raw audio before normalization
                
                print(f"Recording with device: {self.input_device or 'Default'}")
                print(f"Current SILENCE_THRESHOLD: {SILENCE_THRESHOLD}")
                
                # Add a counter to limit debug output
                block_counter = 0
                
                while self.recording:
                    # Check if we should manually stop (controlled by another thread)
                    if self.manual_stop:
                        print("Manual stop detected - stopping recording")
                        break
                        
                    # Get audio block in blocking mode
                    block, overflowed = stream.read(CHUNK_SIZE)
                    if overflowed:
                        print("Audio overflow detected")
                    
                    # Store raw audio before normalization
                    raw_audio.append(block.copy())
                    
                    # Normalize audio block to prevent distortion
                    block = block.flatten()  # Ensure 1D array
                    current_level = np.max(np.abs(block))
                    
                    # Print audio level every 10 blocks to avoid flooding the console
                    if block_counter % 10 == 0:
                        print(f"Audio level: {current_level:.6f} (Threshold: {SILENCE_THRESHOLD})")
                    block_counter += 1
                    
                    # Update minimum noise level
                    if current_level > 0:  # Ignore complete silence
                        self.min_noise_level = min(self.min_noise_level, current_level)
                    
                    if current_level > 0:
                        block = block / current_level * 0.9  # Normalize with headroom
                    
                    # Check for silence with normalized values
                    if current_level < SILENCE_THRESHOLD:
                        silence_count += 1
                        if silence_count >= SILENCE_CHUNKS:  # Change here to allow for longer silence
                            print(f"Silence detected for {silence_count} chunks, stopping recording")
                            break  # Stop recording after a defined number of silent chunks
                    else:
                        silence_count = 0  # Reset silence count if audio is detected
                    
                    # Store normalized audio
                    self.complete_audio.append(block)
                    self.queue.put(block)
                
                # Save both raw and normalized audio locally
                if raw_audio:
                    # Save raw audio
                    raw_path = self.recordings_dir / f"raw_{self.recording_start_time}.wav"
                    raw_combined = np.concatenate(raw_audio)
                    sf.write(raw_path, raw_combined, self.sample_rate)
                    print(f"\nSaved raw audio to {raw_path}")
                    
                    # Save normalized audio
                    norm_path = self.recordings_dir / f"normalized_{self.recording_start_time}.wav"
                    norm_combined = np.concatenate(self.complete_audio)
                    sf.write(norm_path, norm_combined, self.sample_rate)
                    print(f"Saved normalized audio to {norm_path}")
                
                if self.manual_stop:
                    print("\nRecording stopped manually")
                else:
                    print(f"\nRecording thread stopped: {silence_count} silent chunks")
                
                print(f"Minimum noise level detected: {self.min_noise_level:.6f}")
                self.silence_detected = True
                
        except Exception as e:
            print(f"Recording thread error: {e}")
            import traceback
            traceback.print_exc()

    async def stop_recording_task(self):
        """Task to wait for Enter key to stop recording"""
        print("\n>>> Press ENTER again to stop recording...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, input)
        print("Enter key pressed - stopping recording")
        self.manual_stop = True
        return True

    async def process_queue(self):
        """Process audio chunks from the queue and send to server"""
        buffer_chunks = []
        last_send_time = time.time()
        self.silence_detected = False
        sent_chunks = []  # Store chunks being sent to server
        total_chunks_sent = 0

        print("Starting to process audio queue...")

        while self.recording:
            try:
                # Check queue for new chunks
                chunks_in_queue = 0
                while not self.queue.empty():
                    chunk = self.queue.get_nowait()
                    
                    # Check if this is a termination signal (None marker)
                    if chunk is None:
                        print("Found termination signal in queue")
                        # Process any remaining chunks before sending termination
                        if buffer_chunks:
                            await self._send_chunk(buffer_chunks.pop(0), sent_chunks)
                            total_chunks_sent += 1
                        
                        self.silence_detected = True
                        break
                    
                    # Process each chunk individually instead of combining them
                    buffer_chunks.append(chunk)
                    chunks_in_queue += 1
                
                if chunks_in_queue > 0 and chunks_in_queue % 10 == 0:
                    print(f"Got {chunks_in_queue} new chunks from queue, buffer now has {len(buffer_chunks)} chunks")
                
                # If we found a termination signal, break the loop
                if self.silence_detected:
                    break
                
                # Send chunks one at a time instead of combining
                current_time = time.time()
                if current_time - last_send_time >= self.send_interval and buffer_chunks:
                    # Send one chunk at a time
                    await self._send_chunk(buffer_chunks.pop(0), sent_chunks)
                    last_send_time = current_time
                    total_chunks_sent += 1
                
                await asyncio.sleep(0.01)
                
                if self.silence_detected:
                    print("Silence detected by recording thread, finalizing stream...")
                    break
                    
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
                break

        print(f"Finished processing queue. Total chunks sent: {total_chunks_sent}")

        # Save what was sent to server
        if sent_chunks and self.recording_start_time:
            server_path = self.recordings_dir / f"sent_to_server_{self.recording_start_time}.wav"
            server_combined = np.concatenate(sent_chunks)
            sf.write(server_path, server_combined, self.sample_rate)
            print(f"Saved server-bound audio to {server_path}")

        # Send any remaining chunks
        try:
            # Process all remaining chunks
            while buffer_chunks:
                await self._send_chunk(buffer_chunks.pop(0), sent_chunks)
                print("Sent remaining buffer chunk")
                total_chunks_sent += 1

            # Send termination signal only after all chunks are sent
            print("Sending final termination signal...")
            await self.websocket.send(json.dumps({
                "type": "stream_chunk",
                "final": True
            }))
            print(f"Sent termination signal after {total_chunks_sent} total chunks. Waiting for server response...")
            
            # DO NOT close the connection - wait for server response
            # The response will be handled in the record_and_process method
            
        except Exception as e:
            print(f"Error finalizing stream: {e}")
            import traceback
            traceback.print_exc()
            
        # Return True to indicate we've sent all chunks and are waiting for response
        return True

    async def _send_chunk(self, chunk, sent_chunks=None):
        """Helper method to send a single audio chunk"""
        try:
            # Store copy of sent data if tracking
            if sent_chunks is not None:
                sent_chunks.append(chunk.copy())
            
            # Check if the audio contains actual sound or is just silence
            audio_level = np.max(np.abs(chunk))
            if audio_level > 0.01:  # Only print for non-silent chunks
                print(f"Preparing to send chunk with audio level: {audio_level:.6f}")
            
            # Ensure audio is properly scaled for WAV format
            chunk = np.clip(chunk, -1.0, 1.0)
            
            # Convert to WAV with consistent parameters
            with BytesIO() as wav_buffer:
                # Use consistent parameters for all chunks:
                # - PCM_16 format (more reliable than FLOAT)
                # - Mono channel
                # - Fixed sample rate
                sf.write(
                    wav_buffer,
                    chunk,
                    self.sample_rate,
                    format='WAV',
                    subtype='PCM_16'  # Use consistent PCM_16 format that's widely supported
                )
                wav_data = wav_buffer.getvalue()
                
                # Base64 encode the WAV data
                audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                
                # Send the chunk in the format server expects
                await self.websocket.send(json.dumps({
                    "type": "stream_chunk",
                    "audio": audio_base64,
                    "final": False
                }))
                
                print(f"Sent WAV chunk of size: {len(wav_data)} bytes")
            
            return True
            
        except Exception as e:
            print(f"Error sending chunk: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _send_combined_chunk(self, chunks, sent_chunks=None):
        """Helper method to combine and send audio chunks (kept for compatibility)"""
        try:
            # Combine chunks
            combined_chunk = np.concatenate(chunks)
            if sent_chunks is not None:
                sent_chunks.append(combined_chunk.copy())  # Store copy of sent data
            
            # Check if the audio contains actual sound or is just silence
            audio_level = np.max(np.abs(combined_chunk))
            if audio_level > 0.01:  # Only print for non-silent chunks
                print(f"Preparing to send chunk with audio level: {audio_level:.6f}")
            
            # Ensure audio is properly scaled for WAV format
            combined_chunk = np.clip(combined_chunk, -1.0, 1.0)
            
            # Convert to WAV with proper parameters
            with BytesIO() as wav_buffer:
                sf.write(
                    wav_buffer,
                    combined_chunk,
                    self.sample_rate,
                    format='WAV',
                    subtype='FLOAT'
                )
                wav_data = wav_buffer.getvalue()
                
                # Split into smaller chunks to avoid message size limits (1MB)
                max_chunk_size = 500000  # Stay well under the 1MB limit
                
                if len(wav_data) > max_chunk_size:
                    print(f"WAV data size ({len(wav_data)} bytes) exceeds limit, splitting into smaller chunks")
                    chunks_count = (len(wav_data) + max_chunk_size - 1) // max_chunk_size
                    
                    for i in range(chunks_count):
                        start_idx = i * max_chunk_size
                        end_idx = min((i + 1) * max_chunk_size, len(wav_data))
                        chunk_data = wav_data[start_idx:end_idx]
                        
                        # Send each sub-chunk
                        audio_base64 = base64.b64encode(chunk_data).decode('utf-8')
                        await self.websocket.send(json.dumps({
                            "type": "stream_chunk",
                            "audio": audio_base64,
                            "final": False,
                            "part": i + 1,
                            "of": chunks_count
                        }))
                        print(f"Sent chunk part {i+1}/{chunks_count}, size: {len(chunk_data)} bytes")
                        
                        # Small delay between chunks
                        await asyncio.sleep(0.05)
                else:
                    # Send as a single chunk if small enough
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    await self.websocket.send(json.dumps({
                        "type": "stream_chunk",
                        "audio": audio_base64,
                        "final": False
                    }))
                    print(f"Sent chunk of size: {len(wav_data)} bytes")
            
            return True
            
        except Exception as e:
            print(f"Error sending combined chunk: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def record_and_process(self):
        """Main recording and processing loop"""
        try:
            # Initialize wake word detector
            detector = WakeWordDetector(
                access_key=PICOVOICE_ACCESS_KEY,
                model_path=self.wake_word_path
            )
            
            # Initialize function call handler if in function mode
            if self.function_mode:
                self.function_handler = FunctionCallHandler(self.queue, timeout=self.wake_timeout)
            
            async with websockets.connect(SERVER_URL) as websocket:
                self.websocket = websocket
                print(f"Connected to server at {SERVER_URL}")
                
                # Authenticate with server
                device_auth = DeviceAuth()
                if not await authenticate_with_server(websocket, device_auth):
                    print("Authentication failed. Closing connection.")
                    return
                
                # Start heartbeat task
                heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
                
                print("\n" + "="*50)
                if self.function_mode:
                    print("FUNCTION MODE ENABLED")
                    print("Waiting for function call with WAV file or wake word")
                else:
                    print("AUDIO RECORDING CONTROL")
                    print("Say wake word to start recording")
                if self.enable_manual_stop:
                    print("Press ENTER again to stop recording manually")
                print("Recording will stop automatically after silence")
                print("="*50 + "\n")
                
                try:
                    session_count = 0
                    while True:
                        try:
                            # Check if websocket is still open before proceeding
                            print(f"Session count: {session_count}")
                            if session_count > 0:
                                # Ensure connection is still alive
                                try:
                                    pong = await asyncio.wait_for(websocket.ping(), timeout=2.0)
                                    await pong  # Confirms connection is working
                                    print("Connection confirmed active")
                                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed) as e:
                                    print(f"Connection error: {e}")
                                    print("Reconnecting to server...")
                                    # Try to reconnect
                                    try:
                                        websocket = await websockets.connect(SERVER_URL)
                                        self.websocket = websocket
                                        
                                        # Authenticate again
                                        if not await authenticate_with_server(websocket, device_auth):
                                            print("Re-authentication failed. Exiting.")
                                            return
                                            
                                        # Restart heartbeat task
                                        heartbeat_task.cancel()
                                        try:
                                            await heartbeat_task
                                        except asyncio.CancelledError:
                                            pass
                                        heartbeat_task = asyncio.create_task(send_heartbeat(websocket))
                                        print("Reconnected successfully!")
                                    except Exception as e:
                                        print(f"Failed to reconnect: {e}")
                                        return
                            
                            # Ensure any previous stop_task is properly cancelled
                            if self.current_stop_task is not None:
                                if not self.current_stop_task.done():
                                    print("Cancelling previous stop task")
                                    self.current_stop_task.cancel()
                                    try:
                                        await self.current_stop_task
                                    except asyncio.CancelledError:
                                        pass
                                self.current_stop_task = None
                            
                            session_count += 1
                            
                            # Function mode with dual waiting - wait for either wake word or function call
                            if self.function_mode:
                                print("Waiting for function call or wake word...")
                                # Create tasks for both wake word and function call
                                wake_word_task = asyncio.create_task(detector.listen_for_wake_word())
                                function_call_task = asyncio.create_task(self.function_handler.wait_for_function_call())
                                
                                # Wait for either task to complete
                                done, pending = await asyncio.wait(
                                    [wake_word_task, function_call_task],
                                    return_when=asyncio.FIRST_COMPLETED
                                )
                                
                                # Cancel the pending task
                                for task in pending:
                                    task.cancel()
                                    try:
                                        await task
                                    except asyncio.CancelledError:
                                        pass
                                
                                # Determine which task completed
                                wake_word_detected = False
                                function_call_detected = False
                                
                                for task in done:
                                    if task == wake_word_task:
                                        wake_word_detected = task.result()
                                        if wake_word_detected:
                                            print("Wake word detected!")
                                    elif task == function_call_task:
                                        function_call_detected = task.result()
                                        if function_call_detected:
                                            print("Function call processed!")
                                
                                if not (wake_word_detected or function_call_detected):
                                    print("Neither wake word nor function call detected, falling back to Enter key")
                                    await wait_for_enter()
                            else:
                                # Always use wake word detection regardless of platform
                                wake_word_detected = await detector.listen_for_wake_word()
                                if not wake_word_detected:
                                    print("Wake word detection failed, falling back to Enter key")
                                    await wait_for_enter()
                            
                            print("Starting recording!")
                            
                            # Reset flags for new recording
                            self.manual_stop = False
                            
                            # Start recording
                            self.recording = True
                            self.complete_audio = []
                            if not self.function_mode or not function_call_detected:
                                self.queue = Queue()  # Only reset queue if not from function call
                            self.silence_detected = False
                            self.last_activity_time = time.time()  # Update activity time
                            
                            if not self.function_mode or not function_call_detected:
                                print("Starting recording thread")
                                # Start recording thread
                                recording_thread = threading.Thread(target=self.record_audio_blocks)
                                recording_thread.start()
                            
                            # Start a task to wait for Enter key to stop recording (only if enabled)
                            if self.enable_manual_stop:
                                self.current_stop_task = asyncio.create_task(self.stop_recording_task())
                            
                            # Process audio chunks
                            await self.process_queue()
                            
                            # Cancel stop task if it's still running
                            if self.current_stop_task is not None and not self.current_stop_task.done():
                                self.current_stop_task.cancel()
                                try:
                                    await self.current_stop_task
                                except asyncio.CancelledError:
                                    pass
                            
                            # Wait for server to process and respond
                            print("\nWaiting for server to process audio and respond...")
                            
                            # Wait for response
                            transcript = ""
                            goodbye_detected = False
                            response_received = False
                            
                            # Set a timeout for waiting for response
                            start_wait_time = time.time()
                            max_wait_time = 60  # 60 seconds max waiting time
                            
                            while not response_received and time.time() - start_wait_time < max_wait_time:
                                try:
                                    # Use a timeout for receiving messages
                                    response = await asyncio.wait_for(
                                        websocket.recv(), 
                                        timeout=5.0  # 5 second timeout per attempt
                                    )
                                    data = json.loads(response)
                                    
                                    # Reset the start wait time since we got a response
                                    start_wait_time = time.time()
                                    
                                    if data.get("status") == "text_response":
                                        transcript = data.get('transcript', "")
                                        print(f"\nTranscription: {transcript}")
                                        print(f"Response: {data.get('response')}")
                                        
                                        # Check if this is a goodbye message
                                        if contains_goodbye(transcript):
                                            goodbye_detected = True
                                            print("Goodbye phrase detected, will return to wait mode after playback")
                                    
                                    elif data.get("status") == "stream_start":
                                        print("\nReceiving audio response...")
                                        # Clear the audio buffer for new response
                                        self.audio_player.clear()
                                    
                                    elif data.get("status") == "stream_chunk":
                                        # Process audio chunk
                                        try:
                                            chunk_data = base64.b64decode(data["chunk"])
                                            content_type = data.get("content_type", "audio/wav")
                                            
                                            # Add chunk to the buffer
                                            self.audio_player.add_chunk(chunk_data, content_type)
                                            
                                        except Exception as e:
                                            print(f"Error processing audio chunk: {e}")
                                            import traceback
                                            traceback.print_exc()
                                    
                                    elif data.get("status") == "stream_end":
                                        print("Finished receiving audio response, playing...")
                                        # Play the combined audio
                                        self.audio_player.combine_and_play()
                                        
                                        # Update last activity time
                                        self.last_activity_time = time.time()
                                        
                                        # Mark as received full response
                                        response_received = True
                                        
                                        # Check for goodbye or function mode
                                        if self.function_mode and (goodbye_detected or self.function_handler.check_timeout()):
                                            print("\n>>> Returning to function call/wake word wait mode...")
                                        else:
                                            print("\n>>> Recording session complete. Say wake word to record again...")
                                        break
                                    
                                    elif data.get("status") == "error":
                                        print(f"\nError from server: {data.get('message', 'Unknown error')}")
                                        response_received = True
                                        break
                                    
                                    elif data.get("status") == "processing":
                                        print(f"Server status: {data.get('message', 'Processing audio...')}")
                                    
                                except asyncio.TimeoutError:
                                    print("Waiting for server response... (timeout occurred, still waiting)")
                                    continue
                                    
                                except websockets.exceptions.ConnectionClosed:
                                    print("Connection closed by server during response")
                                    break
                            
                            if not response_received:
                                print("\nDidn't receive complete response from server. Continuing...")
                                
                        except Exception as e:
                            print(f"Error in recording session: {e}")
                            import traceback
                            traceback.print_exc()
                            await asyncio.sleep(1)
                
                finally:
                    # Cancel heartbeat task
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    
                    # Clean up audio player resources
                    self.audio_player.cleanup()
                    detector.cleanup()
        
        except Exception as e:
            print(f"Error in record_and_process: {e}")
            import traceback
            traceback.print_exc()
                
            # Clean up audio player resources
            self.audio_player.cleanup()

class AudioBuffer:
    """Class to handle buffering and playing audio chunks"""
    def __init__(self):
        self.chunks = []
        self.temp_dir = tempfile.mkdtemp()
        self.combined_file = None
        self.is_playing = False
        self.content_type = None
        print(f"Created temporary directory for audio: {self.temp_dir}")
    
    def add_chunk(self, chunk_data, content_type=None):
        """Add a chunk to the buffer"""
        if content_type:
            self.content_type = content_type
        self.chunks.append(chunk_data)
    
    def combine_and_play(self):
        """Combine all chunks and play the audio"""
        if not self.chunks:
            print("No audio chunks to play")
            return
        
        try:
            # Determine file extension based on content type
            extension = '.mp3' if self.content_type == 'audio/mpeg' else '.wav'
            
            # Create a combined file
            self.combined_file = os.path.join(self.temp_dir, f"combined{extension}")
            
            # Write all chunks to the file
            with open(self.combined_file, 'wb') as f:
                for chunk in self.chunks:
                    f.write(chunk)
            
            print(f"Combined {len(self.chunks)} chunks into {self.combined_file}")
            
            # Play the combined file
            self.play_file(self.combined_file)
            
        except Exception as e:
            print(f"Error combining and playing audio: {e}")
            import traceback
            traceback.print_exc()
    
    def play_file(self, file_path):
        """Play an audio file using the system's audio player"""
        try:
            self.is_playing = True
            
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay {file_path}")
                print("Played audio using afplay")
            elif sys.platform == "win32":  # Windows
                os.system(f'start {file_path}')
                print("Played audio using Windows default player")
            elif sys.platform == "linux":  # Linux
                os.system(f"aplay {file_path}")
                print("Played audio using aplay")
            else:
                print(f"Unsupported platform: {sys.platform}")
            
            self.is_playing = False
            
        except Exception as e:
            print(f"Error playing audio with system player: {e}")
            self.is_playing = False
    
    def clear(self):
        """Clear the buffer"""
        self.chunks = []
        
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if self.combined_file and os.path.exists(self.combined_file):
                os.unlink(self.combined_file)
            
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                
            print("Cleaned up temporary audio files")
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

class DeviceAuth:
    def __init__(self, keys_dir='device_keys'):
        self.keys_dir = Path(keys_dir)
        self.keys_dir.mkdir(exist_ok=True)
        self.device_id = self._load_or_generate_device_id()
        self.private_key = self._load_or_generate_keys()

    def _load_or_generate_device_id(self):
        id_file = self.keys_dir / 'device_id'
        if id_file.exists():
            return id_file.read_text().strip()
        
        # Generate new device ID
        device_id = str(uuid.uuid4())
        id_file.write_text(device_id)
        return device_id

    def _load_or_generate_keys(self):
        key_file = self.keys_dir / 'private_key.pem'
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
        
        # Generate new key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Save private key
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        key_file.write_bytes(pem)
        
        # Save public key for registration
        public_key = private_key.public_key()
        pub_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        (self.keys_dir / 'public_key.pem').write_bytes(pub_pem)
        
        return private_key

    def sign_challenge(self, challenge):
        signature = self.private_key.sign(
            challenge.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

async def authenticate_with_server(websocket, device_auth):
    """Handle authentication with the server using challenge-response"""
    try:
        # Wait for auth challenge from server
        print("Waiting for authentication challenge...")
        response = await websocket.recv()
        auth_data = json.loads(response)
        
        if auth_data.get('type') != 'auth_challenge':
            print(f"Unexpected response during authentication: {auth_data}")
            return False
        
        # Get challenge
        challenge = auth_data.get('challenge')
        print(f"Received authentication challenge")
        
        # Sign challenge
        signature = device_auth.sign_challenge(challenge)
        
        # Send response
        await websocket.send(json.dumps({
            'device_id': device_auth.device_id,
            'signature': signature
        }))
        print(f"Sent authentication response for device {device_auth.device_id}")
        
        return True
    except Exception as e:
        print(f"Authentication error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def send_heartbeat(websocket):
    """Send periodic heartbeat to keep the connection alive"""
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(HEARTBEAT_INTERVAL)
        except websockets.exceptions.ConnectionClosed:
            break

# Global variable to reference the function call recorder instance
function_call_recorder = None
# Global socket server instance
socket_server_instance = None

class SocketServer:
    """Simple socket server to listen for function call commands"""
    def __init__(self, function_handler, host='127.0.0.1', port=9876):
        self.function_handler = function_handler
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self.server_thread = None
    
    def start(self):
        """Start the socket server in a separate thread"""
        try:
            print(f"Creating socket server on {self.host}:{self.port}...")
            
            # Check if the server is already running
            if self.is_running and self.server_thread and self.server_thread.is_alive():
                print("Socket server is already running")
                return
                
            # Initialize socket before starting thread
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Allow reuse of address to avoid "Address already in use" errors
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Try to find an available port if the default is in use
            max_retries = 5
            original_port = self.port
            for retry in range(max_retries):
                try:
                    print(f"Attempting to bind to {self.host}:{self.port}...")
                    self.server_socket.bind((self.host, self.port))
                    print(f"Successfully bound to {self.host}:{self.port}")
                    break
                except OSError as e:
                    print(f"Failed to bind to port {self.port}: {e}")
                    if "Address already in use" in str(e) and retry < max_retries - 1:
                        # Try the next port
                        self.port = original_port + retry + 1
                        print(f"Trying alternate port {self.port}...")
                    elif retry < max_retries - 1:
                        print(f"Binding error, retrying in 2 seconds... ({e})")
                        time.sleep(2)
                    else:
                        print(f"Failed to bind after {max_retries} attempts.")
                        raise
            
            # Set up listening
            print(f"Setting up socket to listen for connections...")
            self.server_socket.listen(5)
            self.is_running = True
            
            # Start the thread after successful socket setup
            print("Starting socket server thread...")
            self.server_thread = threading.Thread(target=self._run_server)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            # Verify thread started
            if self.server_thread.is_alive():
                print(f"Socket server successfully started on {self.host}:{self.port}")
            else:
                print("Socket server thread failed to start")
                
        except Exception as e:
            print(f"Failed to start socket server: {e}")
            import traceback
            traceback.print_exc()
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
            self.is_running = False
        
    def _run_server(self):
        """Run the socket server"""
        try:
            print(f"Socket server listening on {self.host}:{self.port}")
            
            while self.is_running:
                try:
                    # Set a timeout so we can check is_running periodically
                    self.server_socket.settimeout(1.0)
                    try:
                        client_socket, addr = self.server_socket.accept()
                        print(f"Accepted connection from {addr}")
                        self._handle_client(client_socket)
                    except socket.timeout:
                        # This is expected - just loop to check is_running
                        continue
                        
                except Exception as e:
                    if self.is_running:  # Only print error if still supposed to be running
                        print(f"Error accepting connection: {e}")
                        if not isinstance(e, socket.timeout):
                            import traceback
                            traceback.print_exc()
                            # Sleep briefly to avoid rapid error loops
                            time.sleep(0.5)
                    
        except Exception as e:
            print(f"Error in socket server thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _handle_client(self, client_socket):
        """Handle client connection"""
        try:
            # Receive data with timeout
            client_socket.settimeout(5.0)  # 5 second timeout for receiving data
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                print("Empty data received from client")
                return
                
            print(f"Received data: {data}")
            
            # Parse the command
            if data.startswith("PROCESS_WAV:"):
                wav_file_path = data[len("PROCESS_WAV:"):].strip()
                print(f"Processing WAV file: {wav_file_path}")
                
                # Verify file exists
                if not os.path.exists(wav_file_path):
                    error_msg = f"ERROR: WAV file not found: {wav_file_path}"
                    print(error_msg)
                    client_socket.send(error_msg.encode('utf-8'))
                    return
                    
                # Verify function handler is available
                if not self.function_handler:
                    error_msg = "ERROR: Function handler not available"
                    print(error_msg)
                    client_socket.send(error_msg.encode('utf-8'))
                    return
                
                # Process the WAV file
                self.function_handler.add_function_call(wav_file_path)
                success_msg = "SUCCESS: WAV file added to processing queue"
                print(success_msg)
                client_socket.send(success_msg.encode('utf-8'))
            else:
                error_msg = f"ERROR: Unknown command: {data}"
                print(error_msg)
                client_socket.send(error_msg.encode('utf-8'))
                
        except Exception as e:
            print(f"Error handling client: {e}")
            import traceback
            traceback.print_exc()
            try:
                client_socket.send(f"ERROR: {str(e)}".encode('utf-8'))
            except:
                pass
        finally:
            try:
                client_socket.close()
                print("Client connection closed")
            except:
                pass
    
    def stop(self):
        """Stop the socket server"""
        if not self.is_running:
            print("Socket server is not running")
            return
            
        print("Stopping socket server...")
        self.is_running = False
        
        if self.server_socket:
            try:
                # Force socket to stop accepting by closing it
                self.server_socket.close()
                print("Socket closed")
            except Exception as e:
                print(f"Error closing socket: {e}")
        
        if self.server_thread and self.server_thread.is_alive():
            try:
                # Wait for thread to finish
                self.server_thread.join(timeout=2.0)
                if self.server_thread.is_alive():
                    print("Warning: Socket server thread did not exit cleanly")
                else:
                    print("Socket server thread stopped")
            except Exception as e:
                print(f"Error stopping thread: {e}")
                
        self.server_socket = None
        self.server_thread = None
        print("Socket server stopped completely")

async def record_and_process(debug_mode=False, test_audio_dir='test_audio', local_mode=False, wake_word_path=None, function_mode=False):
    """Global function to create and run the AudioRecorder"""
    global function_call_recorder, socket_server_instance
    
    recorder = AudioRecorder(debug_mode=debug_mode, test_audio_dir=test_audio_dir, local_mode=local_mode, 
                           wake_word_path=wake_word_path, function_mode=function_mode)
    
    # If in function mode, store the recorder globally for external access
    if function_mode:
        function_call_recorder = recorder
        
        # Wait briefly for initialization
        await asyncio.sleep(1)
        
        # Start the socket server if in function mode
        if recorder.function_handler:
            try:
                print("Initializing socket server...")
                # Create a new server if it doesn't exist yet
                if socket_server_instance is None:
                    socket_server_instance = SocketServer(recorder.function_handler)
                
                # Start the server explicitly
                print("Starting socket server...")
                socket_server_instance.start()
                
                # Verify the server is running
                if socket_server_instance.is_running and socket_server_instance.server_thread:
                    print(f"Socket server started and ready to receive commands on {socket_server_instance.host}:{socket_server_instance.port}")
                else:
                    print("Warning: Socket server did not start properly!")
            except Exception as e:
                print(f"Error starting socket server: {e}")
                import traceback
                traceback.print_exc()
    
    await recorder.record_and_process()
    
    # Clean up socket server if it was created
    if function_mode and socket_server_instance:
        socket_server_instance.stop()
        socket_server_instance = None
    
    # Clear the global reference when done
    if function_mode:
        function_call_recorder = None

# Function to allow external code to process a WAV file
def process_wav_file(wav_file_path):
    """External function to process a WAV file"""
    global function_call_recorder
    if function_call_recorder and function_call_recorder.function_handler:
        function_call_recorder.function_handler.add_function_call(wav_file_path)
        return True
    else:
        print("Error: Function call recorder not initialized or not in function mode")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Audio client for remote server')
    parser.add_argument('--server', '-s', default='localhost', 
                        help='Server IP address or hostname')
    parser.add_argument('--port', '-p', type=int, default=8765, 
                        help='Server port')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode using test audio files instead of microphone')
    parser.add_argument('--test-dir', default='test_audio',
                        help='Directory containing test audio files (WAV or MP3)')
    parser.add_argument('--local', action='store_true',
                        help='Run in local mode using localhost server')
    parser.add_argument('--register', action='store_true',
                        help='Register the device before connecting')
    parser.add_argument('--wake-word', type=str,
                       help='Path to custom wake word model file')
    parser.add_argument('--admin-user', type=str, default='admin',
                       help='Admin username for device registration')
    parser.add_argument('--admin-password', type=str,
                       help='Admin password for device registration')
    parser.add_argument('--function-start', action='store_true',
                        help='Wait for external function call with WAV file instead of wake word')
    return parser.parse_args()

# Main function
if __name__ == "__main__":
    # Parse arguments and run the main loop
    args = parse_args()
    
    # Print current timestamp to verify we're using the latest code
    print(f"\nStarting audio client at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using SpaceBar control on macOS: {'Yes' if sys.platform == 'darwin' else 'No'}")
    
    # Check if pynput is installed for keyboard input on macOS
    if sys.platform == "darwin":
        try:
            import pynput.keyboard
        except ImportError:
            print("pynput module not found. Installing...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
            print("pynput installed. Restarting application...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    
    try:
        # Initialize device authentication
        device_auth = DeviceAuth()
        print(f"\nDevice ID: {device_auth.device_id}")
        
        # Register device if needed
        if args.register or args.local:
            print("\nRegistering device...")
            if args.local:
                success = register_local_device(device_auth)
            else:
                if not args.admin_password:
                    print("Error: --admin-password is required for remote registration")
                    sys.exit(1)
                    
                server_url = f"wss://{args.server}"
                print(f"Attempting to register with server: {server_url}")
                success = register_remote_device(
                    device_auth, 
                    server_url,
                    args.admin_user,
                    args.admin_password
                )
            
            if not success:
                print("\nWarning: Device registration failed.")
                if not args.local:
                    print("Please check that:")
                    print("1. The server is running")
                    print("2. The server URL is correct")
                    print("3. The admin credentials are correct")
                print("\nContinuing anyway...")
            else:
                print("\nDevice registration successful!")
        
        # Run the main audio processing loop
        print("\nStarting audio processing...")
        
        # Run in function mode or regular mode based on command line argument
        if args.function_start:
            print("\nFunction start mode enabled - can accept external WAV files")
            asyncio.run(record_and_process(
                debug_mode=args.debug,
                test_audio_dir=args.test_dir,
                local_mode=args.local,
                wake_word_path=args.wake_word,
                function_mode=True
            ))
        else:
            # Regular mode
            asyncio.run(record_and_process(
                debug_mode=args.debug,
                test_audio_dir=args.test_dir,
                local_mode=args.local,
                wake_word_path=args.wake_word,
                function_mode=False
            ))
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
