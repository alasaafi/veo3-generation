from flask import Flask, render_template, request
import os, tempfile, subprocess
from openai import OpenAI
import yt_dlp

app = Flask(__name__)

# مفتاح OpenRouter من البيئة
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
# قص الفيديو (اختياري)
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
# توليد برومبت Veo3 طويل
# -------------------
def generate_prompt(style="cinematic"):
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    
    if style == "cinematic":
        instruction = "Write a single, detailed Veo 3 cinematic prompt with vivid visuals, dynamic lighting, and immersive composition. Do not include any reference to the source video."
    elif style == "artistic":
        instruction = "Write a single, detailed Veo 3 artistic prompt with abstract, creative, and colorful style. Do not reference any video."
    elif style == "asmr":
        instruction = "Write a single, detailed Veo 3 ASMR-style prompt with calm, sensory-focused description. No mention of the video."
    else:
        instruction = "Write one long, detailed Veo 3 prompt in cinematic, artistic, and ASMR styles combined. Do not mention any video."

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
                prompts = generate_prompt(style)
            except Exception as e:
                error = str(e)
            finally:
                # حذف الملفات المؤقتة
                if "video_path" in locals() and os.path.exists(video_path):
                    os.remove(video_path)
                if "short_path" in locals() and os.path.exists(short_path):
                    os.remove(short_path)

    return render_template("index.html", prompts=prompts, error=error)

# -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
