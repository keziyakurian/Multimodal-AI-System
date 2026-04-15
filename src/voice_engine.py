import os
import base64
import requests
from deepgram import DeepgramClient, PrerecordedOptions
from cartesia import Cartesia
from dotenv import load_dotenv

load_dotenv()

class VoiceEngine:
    def __init__(self):
        self.dg_api_key = os.getenv("DEEPGRAM_API_KEY")
        self.cartesia_api_key = os.getenv("CARTESIA_API_KEY")
        
        if self.dg_api_key:
            self.dg_client = DeepgramClient(self.dg_api_key)
        else:
            self.dg_client = None
            
        if self.cartesia_api_key:
            self.cartesia_client = Cartesia(api_key=self.cartesia_api_key)
        else:
            self.cartesia_client = None

    def stt(self, audio_bytes):
        """Converts audio bytes to text using Deepgram."""
        if not self.dg_client:
            return "Error: Deepgram API Key not set."
        
        try:
            payload = {'buffer': audio_bytes}
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
            )
            response = self.dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)
            return response.results.channels[0].alternatives[0].transcript
        except Exception as e:
            return f"STT Error: {str(e)}"

    def tts(self, text):
        """Converts text to speech bytes using Cartesia."""
        if not self.cartesia_client:
            return None
        
        try:
            # Using Cartesia's fast TTS
            # Note: This is a simplified version. Cartesia has better streaming, 
            # but for Streamlit play, we can get the full buffer.
            voice_id = "a0e99840-4383-4a40-8670-13f5d05a933d" # Default pleasant voice
            model_id = "sonic-english"
            
            output_format = {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": 44100
            }
            
            # Use the cartesia client to generate audio
            # Since Cartesia is evolving, let's use a safe HTTP implementation if SDK is tricky
            # But the SDK usually works well.
            audio_data = self.cartesia_client.tts.bytes(
                model_id=model_id,
                transcript=text,
                voice_id=voice_id,
                output_format=output_format
            )
            return audio_data
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None

def get_audio_html(audio_bytes):
    """Helper to create an autoplaying audio element in Streamlit."""
    b64 = base64.b64encode(audio_bytes).decode()
    return f'<audio autoplay="true" src="data:audio/wav;base64,{b64}">'
