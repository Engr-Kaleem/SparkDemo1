import moviepy.editor as mp
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

# Path to the video file
video_path = "foodreview.mp4"
audio_path = "audio.wav"
text_path = "result.txt"

video = mp.VideoFileClip(video_path)

# Extract the audio from the video
audio_file = video.audio
audio_file.write_audiofile(audio_path)

# Initialize recognizer
recognizer = sr.Recognizer()

# Function to split audio into chunks and process each chunk
def process_audio_chunks(audio_path):
    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=1000,
        silence_thresh=-40,
        keep_silence=500,
    )

    full_text = []

    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk{i}.wav"
        chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                full_text.append(text)
            except sr.UnknownValueError:
                print(f"Google Speech Recognition could not understand audio chunk {i}")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service for chunk {i}; {e}")

        # Delete the chunk file after processing
        os.remove(chunk_filename)

    return " ".join(full_text)

# Process the audio chunks and convert to text
text = process_audio_chunks(audio_path)

# Save the text to a .txt file
with open(text_path, "w") as text_file:
    text_file.write(text)

print("\nThe resultant text from the video is: \n")
print(text)
print(f"\nThe text has been successfully saved to {text_path}.")

