# HCA Verify API

FastAPI backend for the HCA Verify tool. Downloads videos via yt-dlp, analyzes with Gemini 1.5 Flash.

## Deploy to Railway (5 minutes)

1. Go to railway.app → New Project → Deploy from GitHub repo
2. Push this folder to a GitHub repo first, OR use Railway CLI:
   ```
   npm install -g @railway/cli
   railway login
   railway init
   railway up
   ```

3. Set environment variables in Railway dashboard:
   - `GEMINI_API_KEY` = your Google AI Studio key
   - `ALLOWED_ORIGINS` = https://joinhca.org,https://www.joinhca.org

4. Railway gives you a URL like `https://hca-verify-api.up.railway.app`
   → Put that URL in your frontend as the API_BASE_URL

## API

**POST /verify**
```json
{ "url": "https://www.tiktok.com/@user/video/123" }
```

Returns full forensic analysis JSON.

**GET /health**
Returns `{"status": "ok"}`

## Local dev
```
pip install -r requirements.txt
GEMINI_API_KEY=your_key uvicorn main:app --reload
```
