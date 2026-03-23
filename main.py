import os
import sys
import uuid
import asyncio
import tempfile
import json
import re
import time
import base64
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

class VerifyRequest(BaseModel):
    url: str

class LayerResult(BaseModel):
    status: str
    label: str
    detail: str

class VerifyResponse(BaseModel):
    success: bool
    score: int
    verdict: str
    title: str
    platform: str
    scan_id: str
    video_frames: LayerResult
    audio_signal: LayerResult
    face_analysis: LayerResult
    metadata: LayerResult
    script_analysis: LayerResult
    overall: LayerResult
    findings: list[str]
    error: Optional[str] = None

SYSTEM_PROMPT = (
    "You are a forensic AI content analyst for the Human Content Alliance (HCA). "
    "Your job is to determine whether a video was made by a human or generated/manipulated by AI.\n\n"
    "STEP 1 - SYNTHEID WATERMARK CHECK (highest priority):\n"
    "Before anything else, check for SynthID watermarks. SynthID is Google DeepMind's imperceptible "
    "watermarking technology embedded in AI-generated content at the pixel and audio level. "
    "If you detect any SynthID watermark signals or patterns consistent with SynthID embedding, "
    "this is strong evidence of AI generation and should heavily weight the score toward 0.\n\n"
    "STEP 2 - FIVE FORENSIC LAYERS:\n"
    "1. VIDEO FRAMES: Check for generative artifacts - unnatural motion interpolation, temporal "
    "flickering, diffusion model noise patterns (Sora, Runway, Pika, Kling), morphing faces, "
    "impossible physics, repeating backgrounds, or any visual signs of AI generation.\n"
    "2. AUDIO SIGNAL: Check for AI voice synthesis - unnatural prosody, missing breath sounds, "
    "spectral flatness, formant transitions inconsistent with natural speech, robotic cadence, "
    "or AI-generated music (too perfect, no natural variation).\n"
    "3. FACE ANALYSIS: Check for deepfake indicators - hairline boundary artifacts, inconsistent "
    "ear rendering, unnatural blinking, eye specular reflections mismatching lighting, skin "
    "texture too smooth, rendered-looking teeth, or face-swap indicators.\n"
    "4. METADATA AND PROVENANCE: Does this look like real camera footage with natural grain? "
    "Or a render output - too crisp, no camera shake, perfect stabilization? "
    "Any signs of being screen-recorded from an AI generation interface?\n"
    "5. SCRIPT AND NARRATION: If spoken content exists, does it feel human - natural hesitations, "
    "genuine emotion, conversational imperfections? Or GPT-generated - overly structured, "
    "perfect grammar, sounds like a listicle?\n\n"
    "RESPOND ONLY with a valid JSON object. No markdown, no text outside the JSON.\n\n"
    "JSON FORMAT:\n"
    "{\n"
    '  "score": <integer 0-100, where 100 = definitely human, 0 = definitely AI>,\n'
    '  "verdict": "<human if score >= 55, else ai>",\n'
    '  "video_frames": {"status": "<pass|fail|warn>", "label": "<6 words max>", "detail": "<1-2 sentences>"},\n'
    '  "audio_signal": {"status": "<pass|fail|warn>", "label": "<6 words max>", "detail": "<1-2 sentences>"},\n'
    '  "face_analysis": {"status": "<pass|fail|warn>", "label": "<6 words max>", "detail": "<1-2 sentences>"},\n'
    '  "metadata": {"status": "<pass|fail|warn>", "label": "<6 words max>", "detail": "<1-2 sentences>"},\n'
    '  "script_analysis": {"status": "<pass|fail|warn>", "label": "<6 words max>", "detail": "<1-2 sentences>"},\n'
    '  "findings": ["<finding 1>", "<finding 2>", "<finding 3>"]\n'
    "}\n\n"
    "RULES: labels max 6 words, details max 2 sentences, JSON must be complete and valid, "
    "never truncate a string. If a layer cannot be assessed, use warn status."
)


def detect_platform(url: str) -> str:
    u = url.lower()
    if "tiktok.com" in u: return "TikTok"
    if "instagram.com" in u: return "Instagram"
    if "youtube.com" in u or "youtu.be" in u: return "YouTube"
    if "twitter.com" in u or "x.com" in u: return "X / Twitter"
    if "facebook.com" in u or "fb.watch" in u: return "Facebook"
    return "Social Media"

def make_scan_id() -> str:
    return "HCA-" + uuid.uuid4().hex[:6].upper()

def get_cookies_path() -> Optional[Path]:
    direct = Path("/app/cookies.txt")
    if direct.exists():
        print("Using cookies from /app/cookies.txt")
        return direct
    cookies_b64 = os.environ.get("COOKIES_B64", "")
    if cookies_b64:
        decoded = base64.b64decode(cookies_b64).decode("utf-8")
        tmp = TEMP_DIR / "cookies.txt"
        tmp.write_text(decoded)
        print("Using cookies from COOKIES_B64 env var")
        return tmp
    print("No cookies available")
    return None

async def run_ytdlp(cmd: list, timeout: int = 120) -> tuple:
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
    cookies_path = get_cookies_path()
    cookies_args = ["--cookies", str(cookies_path)] if cookies_path else []

    base_cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--download-sections", "*0:00-0:30",
        "--force-keyframes-at-cuts",
        "--no-check-certificates",
        "--socket-timeout", "30",
        "-o", str(output_path),
    ] + cookies_args

    browser_headers = [
        "--add-header", "User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
    ]

    cmd1 = base_cmd + browser_headers + [
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        url
    ]
    code, stderr = await run_ytdlp(cmd1)
    print(f"Attempt 1 - exit: {code}, file: {output_path.exists()}")
    if stderr: print(f"stderr: {stderr[:300]}")
    if code == 0 and output_path.exists():
        return True

    if output_path.exists(): output_path.unlink()

    cmd2 = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "best[height<=480]/best",
        "--no-check-certificates",
        "--socket-timeout", "30",
        "--add-header", "User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-o", str(output_path),
    ] + cookies_args + [url]

    code2, stderr2 = await run_ytdlp(cmd2, timeout=90)
    print(f"Attempt 2 - exit: {code2}, file: {output_path.exists()}")
    if stderr2: print(f"stderr2: {stderr2[:300]}")
    if code2 == 0 and output_path.exists():
        return True

    print("Both download attempts failed")
    return False

def upload_to_gemini(video_path: Path) -> Optional[object]:
    try:
        file_size = video_path.stat().st_size
        print(f"Uploading to Gemini: {video_path.name}, size: {file_size//1024}KB")

        if file_size > 50 * 1024 * 1024:
            print(f"File too large ({file_size//1024//1024}MB), rejecting")
            return None

        with open(video_path, "rb") as f:
            file = gemini_client.files.upload(
                file=f,
                config=types.UploadFileConfig(
                    mime_type="video/mp4",
                    display_name=video_path.name
                )
            )

        print(f"Upload complete, state: {file.state}")
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
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=gemini_file.uri,
                            mime_type="video/mp4"
                        ),
                        types.Part.from_text(
                            text="Analyze this video forensically and return your assessment as JSON."
                        )
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=4096,
                response_mime_type="application/json"
            )
        )

        raw = response.text.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            print(f"JSON parse failed, raw: {raw[:300]}")
            return None

    except Exception as e:
        print(f"Gemini analysis error: {e}")
        return None

def cleanup_gemini_file(gemini_file):
    try:
        gemini_client.files.delete(name=gemini_file.name)
    except Exception:
        pass

def cleanup_local_file(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass

def build_layer(raw: dict, key: str) -> LayerResult:
    layer = raw.get(key, {})
    status = layer.get("status", "warn")
    label_raw = layer.get("label", "Unable to assess")
    detail = layer.get("detail", "Insufficient data for this layer.")
    prefix = {"pass": "✓", "fail": "✕", "warn": "⚠"}.get(status, "⚠")
    return LayerResult(status=status, label=f"{prefix} {label_raw}", detail=detail)


@app.get("/")
def root():
    return {"status": "ok", "service": "HCA Verify API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/verify", response_model=VerifyResponse)
async def verify_video(req: VerifyRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY not configured")

    url = req.url.strip()
    if not url.startswith("http"):
        raise HTTPException(400, "Invalid URL")

    scan_id = make_scan_id()
    platform = detect_platform(url)
    video_path = TEMP_DIR / f"{scan_id}.mp4"

    try:
        downloaded = await download_video(url, video_path)
        if not downloaded:
            raise HTTPException(422, "Could not download video. The link may be private, expired, or unsupported.")

        gemini_file = upload_to_gemini(video_path)
        cleanup_local_file(video_path)

        if not gemini_file:
            raise HTTPException(500, "Failed to process video for analysis.")

        result = analyze_with_gemini(gemini_file)
        cleanup_gemini_file(gemini_file)

        if not result:
            raise HTTPException(500, "Analysis failed. Please try again.")

        score = max(0, min(100, int(result.get("score", 50))))
        verdict = result.get("verdict", "human" if score >= 55 else "ai")
        title = "Human-Made Content" if verdict == "human" else "AI-Generated Signals Detected"

        findings = result.get("findings", ["Analysis complete.", "See layer breakdown above.", "Contact HCA if result seems incorrect."])
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
                status="pass" if verdict == "human" else "fail",
                label="HMC Eligible" if verdict == "human" else "Not HMC Eligible",
                detail=(
                    "This content qualifies for HMC certification. Apply to receive your certificate and watermark."
                    if verdict == "human" else
                    "This content does not qualify for any HMC tier based on the signals detected."
                )
            ),
            findings=findings
        )

    except HTTPException:
        raise
    except Exception as e:
        cleanup_local_file(video_path)
        raise HTTPException(500, f"Unexpected error: {str(e)}")
