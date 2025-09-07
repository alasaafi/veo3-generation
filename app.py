from flask import Flask, render_template, request
import os, subprocess, tempfile
from openai import OpenAI
import yt_dlp
from faster_whisper import WhisperModel

app = Flask(__name__)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openai/gpt-4o-mini"

# -------------------
# تحميل الفيديو وقصه باستخدام TemporaryDirectory
# -------------------
def download_and_trim_youtube(video_url, duration=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        short_path = os.path.join(tmpdir, "video_short.mp4")

        # تحميل الفيديو
        ydl_opts = {
            "format": "mp4",
            "outtmpl": video_path,
            "cookies": "cookies.txt",
            "quiet": True,
            "noplaylist": True

        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # قص الفيديو
        cmd = ["ffmpeg", "-y", "-i", video_path, "-t", str(duration), "-c", "copy", short_path]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if not os.path.exists(short_path):
            raise FileNotFoundError("Failed to create the short video.")

        return short_path

# -------------------
# تفريغ الصوت باستخدام faster-whisper
# -------------------
def transcribe_video(video_path):
    model = WhisperModel("base")  # يمكن تغيير الحجم "small", "medium", "large-v2"
    segments, _ = model.transcribe(video_path, beam_size=5)
    transcript = " ".join([seg.text for seg in segments])
    return transcript

# -------------------
# استخراج أهم الكلمات بدون scikit-learn
# -------------------
def extract_keywords(text, top_n=15):
    words = text.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    keywords = sorted(freq, key=freq.get, reverse=True)[:top_n]
    return keywords

# -------------------
# توليد Veo 3 prompt
# -------------------
def generate_prompt_from_text(text, style="cinematic"):
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
                short_path = download_and_trim_youtube(video_url, duration=30)
                transcript = transcribe_video(short_path)
                prompts = generate_prompt_from_text(transcript, style)
            except Exception as e:
                error = str(e)

    return render_template("index.html", prompts=prompts, error=error)

# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
