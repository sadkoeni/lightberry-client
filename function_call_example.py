import os
import shutil
import time
import socket

# Path to the function_calls directory that the audio client watches
FUNCTION_CALLS_DIR = "function_calls"

def process_wav_file_via_directory(wav_file_path):
    """Process a WAV file by copying it to the function_calls directory"""
    if not os.path.exists(wav_file_path):
        print(f"Error: Source WAV file not found: {wav_file_path}")
        return False
        
    if not os.path.exists(FUNCTION_CALLS_DIR):
        os.makedirs(FUNCTION_CALLS_DIR, exist_ok=True)
    
    # Generate a unique filename based on timestamp to avoid conflicts
    timestamp = int(time.time())
    filename = f"instruction_{timestamp}.wav"
    target_path = os.path.join(FUNCTION_CALLS_DIR, filename)
    
    try:
        # Copy the WAV file to the function_calls directory
        shutil.copy2(wav_file_path, target_path)
        print(f"Successfully copied {wav_file_path} to {target_path}")
        return True
    except Exception as e:
        print(f"Error copying WAV file: {e}")
        return False

def process_wav_file_via_socket(wav_file_path, host='127.0.0.1', port=9876, max_retries=3, retry_delay=0.5):
    """Process a WAV file by sending a command to the socket server"""
    if not os.path.exists(wav_file_path):
        print(f"Error: Source WAV file not found: {wav_file_path}")
        return False
    
    # Try multiple times to connect to the socket server
    for retry in range(max_retries):
        try:
            print(f"Connecting to socket server at {host}:{port} (attempt {retry+1}/{max_retries})...")
            
            # Connect to the socket server with a timeout
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)  # 5 second timeout
            client_socket.connect((host, port))
            print("Connection established!")
            
            # Send the process command with the full path to the WAV file
            abs_path = os.path.abspath(wav_file_path)
            command = f"PROCESS_WAV:{abs_path}"
            print(f"Sending command: {command}")
            client_socket.send(command.encode('utf-8'))
            
            # Wait for response with timeout
            client_socket.settimeout(10)  # 10 second timeout for response
            response = client_socket.recv(1024).decode('utf-8')
            print(f"Server response: {response}")
            
            # Close the connection
            client_socket.close()
            
            return response.startswith("SUCCESS")
            
        except ConnectionRefusedError:
            print(f"Connection refused on attempt {retry+1}")
            if retry < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                print("Maximum retry attempts reached.")
                print("Please make sure the audio client is running with --function-start")
                return False
        except socket.timeout:
            print(f"Connection timed out on attempt {retry+1}")
            if retry < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                print("Maximum retry attempts reached.")
                return False
        except Exception as e:
            print(f"Error communicating with socket server: {e}")
            import traceback
            traceback.print_exc()
            if retry < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
            else:
                return False
    
    return False

# Process a WAV file
if __name__ == "__main__":
    wav_file = 'instructions.wav'
    
    # Try socket approach first (requires client to be running)
    print("\n=== AUDIO CLIENT FUNCTION CALL EXAMPLE ===")
    print("This script demonstrates how to trigger audio processing via function calls.")
    print("Make sure the audio client is running with: python audio_client.py --function-start\n")
    
    print("Attempting to use socket connection...")
    try:
        if process_wav_file_via_socket(wav_file):
            print("\nSUCCESS: Successfully sent WAV file to audio client via socket.")
        else:
            print("\nFalling back to directory method...")
            if process_wav_file_via_directory(wav_file):
                print("\nSUCCESS: WAV file copied to function_calls directory.")
                print("If the audio client is running, it will process this file.")
            else:
                print("\nERROR: Failed to process WAV file by any method.")
    except Exception as e:
        print(f"\nSocket connection failed: {e}")
        print("Falling back to directory method...")
        if process_wav_file_via_directory(wav_file):
            print("\nSUCCESS: WAV file copied to function_calls directory.")
            print("If the audio client is running, it will process this file.")
        else:
            print("\nERROR: Failed to process WAV file by any method.")
