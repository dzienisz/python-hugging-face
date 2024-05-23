from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

client = OpenAI()

audio_file= open("Mateusz.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)