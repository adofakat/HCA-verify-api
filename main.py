import os
import sys
import uuid
import asyncio
import subprocess
import tempfile
import json
import re
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── CONFIG ────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
TEMP_DIR = Path(tempfile.gettempdir()) / "hca_videos"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="HCA Verify API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODELS ────────────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    url: str

class LayerResult(BaseModel):
    status: str          # "pass" | "fail" | "warn"
    label: str           # e.g. "✓ No AI patterns"
    detail: str          # explanation

class VerifyResponse(BaseModel):
    success: bool
    score: int           # 0-100
    verdict: str         # "human" | "ai"
    title: str           # "Human-Made Content" | "AI-Generated Signals Detected"
    platform: str
    scan_id: str
    video_frames: LayerResult
    audio_signal: LayerResult
    face_analysis: LayerResult
    metadata: LayerResult
    script_analysis: LayerResult
    overall: LayerResult
    findings: list[str]  # 3 key findings
    error: Optional[str] = None

# ── FORENSIC PROMPT ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a forensic AI content analyst for the Human Content Alliance (HCA).
Your job is to analyze videos and determine whether they were made by a human or generated/heavily manipulated by AI.

Analyze the provided video across these five forensic layers:

1. VIDEO FRAMES: Look for generative video artifacts — unnatural motion interpolation, temporal flickering, 
   pixel-level noise patterns consistent with diffusion models (Sora, Runway, Pika, Kling), 
   morphing faces, impossible physics, repeating background patterns, or any visual inconsistencies 
   that suggest AI generation rather than camera capture.

2. AUDIO SIGNAL: Analyze the audio for signs of AI voice synthesis — unnatural prosody, 
   missing breath sounds, spectral flatness in voice frequencies, formant transitions inconsistent 
   with natural speech, robotic cadence, or audio that sounds "too clean" with no room tone.
   Also flag AI-generated music (too perfect, no natural variation).

3. FACE ANALYSIS: Look for deepfake or AI-generated face indicators — hairline boundary artifacts, 
   inconsistent ear rendering, unnatural blinking frequency, eye specular reflections that don't 
   match lighting, skin texture that's too smooth/uniform, teeth that look rendered, 
   or any face-swap indicators.

4. METADATA & PROVENANCE: Assess what you can observe about the video's origin — 
   does it look like real camera footage with natural grain, compression artifacts from a real device?
   Or does it look like a render output — too crisp, no camera shake, perfect stabilization?
   Note any signs of being screen-recorded from an AI generation interface.

5. SCRIPT & NARRATION: If there is spoken content, evaluate whether the language patterns 
   feel human — natural hesitations, personality, genuine emotion, conversational imperfections.
   Or does it feel like GPT-generated copy being read out — overly structured, perfect grammar,
   lacks authentic voice, sounds like a listicle.

Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON.

{
  "score": <integer 0-100, where 100 = definitely human, 0 = definitely AI>,
  "verdict": "<'human' if score >= 55, else 'ai'>",
  "video_frames": {
    "status": "<'pass' | 'fail' | 'warn'>",
    "label": "<short result e.g. 'No AI patterns detected' or 'Generative artifacts found'>",
    "detail": "<1-2 sentence specific finding>"
  },
  "audio_signal": {
    "status": "<'pass' | 'fail' | 'warn'>",
    "label": "<short result>",
    "detail": "<1-2 sentence specific finding>"
  },
  "face_analysis": {
    "status": "<'pass' | 'fail' | 'warn'>",
    "label": "<short result>",
    "detail": "<1-2 sentence specific finding>"
  },
  "metadata": {
    "status": "<'pass' | 'fail' | 'warn'>",
    "label": "<short result>",
    "detail": "<1-2 sentence specific finding>"
  },
  "script_analysis": {
    "status": "<'pass' | 'fail' | 'warn'>",
    "label": "<short result>",
    "detail": "<1-2 sentence specific finding>"
  },
  "findings": [
    "<Key finding 1 — most important signal detected>",
    "<Key finding 2>",
    "<Key finding 3>"
  ]
}

Be direct, specific, and accurate. Don't hedge excessively — give your best forensic assessment.
If you genuinely cannot assess a layer (e.g. no face visible), mark it 'warn' and explain.
"""

# ── HELPERS ───────────────────────────────────────────────────────
def detect_platform(url: str) -> str:
    url = url.lower()
    if "tiktok.com" in url:        return "TikTok"
    if "instagram.com" in url:     return "Instagram"
    if "youtube.com" in url or "youtu.be" in url: return "YouTube"
    if "twitter.com" in url or "x.com" in url:    return "X / Twitter"
    if "facebook.com" in url or "fb.watch" in url: return "Facebook"
    return "Social Media"

def make_scan_id() -> str:
    return "HCA-" + uuid.uuid4().hex[:6].upper()

# Cookies file path - set COOKIES_FILE env var to path of a cookies.txt file
# Or set COOKIES_B64 env var to base64-encoded cookies.txt content
COOKIES_PATH = Path(os.environ.get("COOKIES_FILE", "/app/cookies.txt"))

def setup_cookies() -> Optional[Path]:
    """Return path to cookies file if available."""
    # Option 1: direct file path
    if COOKIES_PATH.exists():
        print(f"Using cookies from {COOKIES_PATH}")
        return COOKIES_PATH
    # Option 2: base64 encoded cookies in env var
    cookies_b64 = os.environ.get("COOKIES_B64", "")
    if cookies_b64:
        import base64
        cookies_data = base64.b64decode(cookies_b64).decode("utf-8")
        tmp_cookies = TEMP_DIR / "cookies.txt"
        tmp_cookies.write_text(cookies_data)
        print("Using cookies from COOKIES_B64 env var")
        return tmp_cookies
    print("No cookies available - downloads may fail for TikTok/Instagram")
    return None

async def run_ytdlp(cmd: list, timeout: int = 120) -> tuple:
    """Run a yt-dlp command and return (returncode, stderr_text)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        stderr_text = stderr.decode(errors="replace") if stderr else ""
        return proc.returncode, stderr_text
    except asyncio.TimeoutError:
        return -1, "timeout"
    except Exception as e:
        return -1, str(e)

async def download_video(url: str, output_path: Path) -> bool:
    """Download first 60 seconds of video using yt-dlp."""
    cookies_path = setup_cookies()
    cookies_args = ["--cookies", str(cookies_path)] if cookies_path else []

    base_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--download-sections", "*0:00-1:00",
        "--force-keyframes-at-cuts",
        "--no-check-certificates",
        "--socket-timeout", "30",
        "-o", str(output_path),
    ] + cookies_args

    # Headers to look like a real browser
    browser_headers = [
        "--add-header", "User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    ]

    # Attempt 1: best quality with cookies + browser headers
    cmd1 = base_cmd + browser_headers + [
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        url
    ]
    code, stderr = await run_ytdlp(cmd1)
    print(f"Attempt 1 - exit: {code}, file: {output_path.exists()}")
    if stderr: print(f"stderr: {stderr[:300]}")
    if code == 0 and output_path.exists():
        return True

    # Clean up partial file
    if output_path.exists(): output_path.unlink()

    # Attempt 2: simpler format, no sections
    simple_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "best[height<=480]/best",
        "--no-check-certificates",
        "--socket-timeout", "30",
        "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", str(output_path),
    ] + cookies_args + [url]
    code2, stderr2 = await run_ytdlp(simple_cmd, timeout=90)
    print(f"Attempt 2 - exit: {code2}, file: {output_path.exists()}")
    if stderr2: print(f"stderr2: {stderr2[:300]}")
    if code2 == 0 and output_path.exists():
        return True

    print("Both download attempts failed")
    return False

def upload_to_gemini(video_path: Path) -> Optional[object]:
    """Upload video file to Gemini File API."""
    try:
        file_size = video_path.stat().st_size
        print(f"Uploading to Gemini: {video_path.name}, size: {file_size//1024}KB")
        with open(video_path, "rb") as f:
            file = gemini_client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type="video/mp4",
                    display_name=video_path.name
                )
            )
        print(f"Upload complete, state: {file.state}")
        # Wait for processing
        max_wait = 120
        waited = 0
        while str(file.state) in ("FileState.PROCESSING", "PROCESSING") and waited < max_wait:
            time.sleep(3)
            waited += 3
            file = gemini_client.files.get(name=file.name)
            print(f"File state after {waited}s: {file.state}")
        if str(file.state) in ("FileState.FAILED", "FAILED"):
            print("Gemini file processing FAILED")
            return None
        print(f"File ready: {file.state}")
        return file
    except Exception as e:
        print(f"Gemini upload error: {type(e).__name__}: {e}")
        return None

def analyze_with_gemini(gemini_file) -> Optional[dict]:
    """Send video to Gemini for forensic analysis."""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=gemini_file.uri,
                            mime_type="video/mp4"
                        ),
                        types.Part.from_text(
                            "Analyze this video forensically and return your assessment as JSON."
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1500,
                response_mime_type="application/json"
            )
        )
        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        return json.loads(raw)
    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return None

def cleanup_gemini_file(gemini_file):
    """Delete file from Gemini after analysis."""
    try:
        gemini_client.files.delete(name=gemini_file.name)
    except Exception:
        pass

def cleanup_local_file(path: Path):
    """Delete local video file."""
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

def build_layer(raw: dict, key: str) -> LayerResult:
    """Build a LayerResult from Gemini's response."""
    layer = raw.get(key, {})
    status = layer.get("status", "warn")
    label_raw = layer.get("label", "Unable to assess")
    detail = layer.get("detail", "Insufficient data for this layer.")

    # Add symbol prefix based on status
    if status == "pass":
        label = f"✓ {label_raw}"
    elif status == "fail":
        label = f"✕ {label_raw}"
    else:
        label = f"⚠ {label_raw}"

    return LayerResult(status=status, label=label, detail=detail)

# ── ROUTES ────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "HCA Verify API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify", response_model=VerifyResponse)
async def verify_video(req: VerifyRequest):
    if not GEMINI_API_KEY or not gemini_client:
        raise HTTPException(500, "GEMINI_API_KEY not configured")

    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL")

    scan_id = make_scan_id()
    platform = detect_platform(url)
    video_path = TEMP_DIR / f"{scan_id}.mp4"

    try:
        # Step 1: Download
        downloaded = await download_video(url, video_path)
        if not downloaded:
            raise HTTPException(422, "Could not download video. The link may be private, expired, or unsupported.")

        # Step 2: Upload to Gemini
        gemini_file = upload_to_gemini(video_path)
        if not gemini_file:
            raise HTTPException(500, "Failed to process video for analysis.")

        # Step 3: Delete local file immediately
        cleanup_local_file(video_path)

        # Step 4: Analyze
        result = analyze_with_gemini(gemini_file)

        # Step 5: Delete from Gemini
        cleanup_gemini_file(gemini_file)

        if not result:
            raise HTTPException(500, "Analysis failed. Please try again.")

        # Step 6: Build response
        score   = max(0, min(100, int(result.get("score", 50))))
        verdict = result.get("verdict", "human" if score >= 55 else "ai")

        title = "Human-Made Content" if verdict == "human" else "AI-Generated Signals Detected"

        overall_status  = "pass" if verdict == "human" else "fail"
        overall_label   = "HMC Eligible" if verdict == "human" else "Not HMC Eligible"
        overall_detail  = (
            "This content qualifies for HMC certification. Apply to receive your certificate and watermark."
            if verdict == "human" else
            "This content does not qualify for any HMC tier based on the signals detected."
        )

        findings = result.get("findings", [
            "Analysis complete.",
            "See layer breakdown above for details.",
            "Contact HCA if you believe this result is incorrect."
        ])
        # Ensure exactly 3
        while len(findings) < 3:
            findings.append("No additional findings.")
        findings = findings[:3]

        return VerifyResponse(
            success=True,
            score=score,
            verdict=verdict,
            title=title,
            platform=platform,
            scan_id=scan_id,
            video_frames=build_layer(result, "video_frames"),
            audio_signal=build_layer(result, "audio_signal"),
            face_analysis=build_layer(result, "face_analysis"),
            metadata=build_layer(result, "metadata"),
            script_analysis=build_layer(result, "script_analysis"),
            overall=LayerResult(
                status=overall_status,
                label=overall_label,
                detail=overall_detail
            ),
            findings=findings
        )

    except HTTPException:
        raise
    except Exception as e:
        cleanup_local_file(video_path)
        raise HTTPException(500, f"Unexpected error: {str(e)}")
