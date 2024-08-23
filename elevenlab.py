import os
from typing import IO
from io import BytesIO
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Access the API key from the environment variables
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise ValueError("API Key is not set. Please set the ELEVENLABS_API_KEY environment variable.")

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def text_to_speech_stream(text: str) -> IO[bytes]:
    try:
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",  # Adam pre-made voice
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # Create a BytesIO object to hold the audio data in memory
        audio_stream = BytesIO()

        # Write each chunk of audio data to the stream
        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        # Reset stream position to the beginning
        audio_stream.seek(0)

        # Return the BytesIO object for further use
        return audio_stream

    except Exception as e:
        print(f"An error occurred: {e}")

def save_audio_to_file(text: str, file_path: str):
    audio_stream = text_to_speech_stream(text)
    if audio_stream:
        with open(file_path, 'wb') as file:
            file.write(audio_stream.getvalue())
        print(f"Audio saved to {file_path}")
    else:
        print("Failed to create audio stream.")

# Test the function
save_audio_to_file("This is James", "output_audio.mp3")
