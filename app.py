from flask import Flask, render_template, request
import os, tempfile, subprocess
from openai import OpenAI
import yt_dlp
from faster_whisper import WhisperModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

# -------------------
# تحميل الفيديو
# -------------------
def download_youtube_video(url):
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "video.mp4")

    ydl_opts = {
        "format": "mp4",
        "outtmpl": temp_file,
        "quiet": True,
        "noplaylist": True,
        "ignoreerrors": True,
        "merge_output_format": "mp4"
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return temp_file

# -------------------
# قص الفيديو
# -------------------
def trim_video(video_path, duration=30):
    trimmed_file = video_path.replace(".mp4", "_short.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-t", str(duration),
        "-c", "copy", trimmed_file
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return trimmed_file

# -------------------
# استخراج النص باستخدام faster-whisper
# -------------------
def transcribe_video(video_path):
    model = WhisperModel("base")  # يمكنك استخدام "small" أو "medium"
    segments, _ = model.transcribe(video_path, beam_size=5)
    transcript = " ".join([segment.text for segment in segments])
    return transcript

# -------------------
# استخراج أهم الكلمات باستخدام TF-IDF
# -------------------
def extract_keywords(text, top_n=15):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text])
    scores = tfidf_matrix.toarray()[0]
    indices = np.argsort(scores)[::-1][:top_n]
    keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
    return keywords

# -------------------
# توليد Veo3 prompt بناءً على محتوى الفيديو
# -------------------
def generate_prompt_from_video(text, style="cinematic"):
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    keywords = extract_keywords(text)
    keywords_str = ", ".join(keywords)

    instruction = f"Create a single, detailed Veo 3 {style} prompt based ONLY on these keywords: {keywords_str}. Do not reference the video link itself."
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an AI that generates high-quality Veo 3 prompts."},
            {"role": "user", "content": instruction},
        ],
    )

    return response.choices[0].message.content.strip()

# -------------------
# Routes
# -------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prompts = None
    error = None
    if request.method == "POST":
        video_url = request.form.get("url")
        style = request.form.get("style", "cinematic")

        if not video_url:
            error = "Please enter a video URL."
        else:
            try:
                video_path = download_youtube_video(video_url)
                short_path = trim_video(video_path, duration=30)
                transcript = transcribe_video(short_path)
                prompts = generate_prompt_from_video(transcript, style)
            except Exception as e:
                error = str(e)
            finally:
                if "video_path" in locals() and os.path.exists(video_path):
                    os.remove(video_path)
                if "short_path" in locals() and os.path.exists(short_path):
                    os.remove(short_path)

    return render_template("index.html", prompts=prompts, error=error)

# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
