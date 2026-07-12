# Setup — backend + Lovable frontend

## 1. Backend (your Mac, once)

```bash
cd /Users/siddharth/Desktop/GitHub/mscthesis_abm
pip3 install fastapi "uvicorn[standard]"
uvicorn api.server:app --port 8000
```

Verify in a second terminal:

```bash
curl http://localhost:8000/meta
curl -N -X POST http://localhost:8000/simulate_stream \
  -H "Content-Type: application/json" \
  -d '{"regime":"stressed","fv_source":"synthetic","horizon_days":5,"seed":171}'
```

The second command should print JSON lines one by one as days complete. Keep uvicorn
running; it reloads nothing, so restart it if `api/server.py` changes.

## 2. Tunnel (so the hosted Lovable app can reach your Mac)

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000
```

Copy the printed `https://<random>.trycloudflare.com` URL. It changes on every restart —
that is why the app has an editable API-URL setting. If you want a stable URL instead:
`brew install ngrok`, sign up (free static domain), then
`ngrok http 8000 --domain <your-static-domain>.ngrok-free.app`.

Sanity check: open `https://<tunnel-url>/meta` in a browser.

## 3. Lovable

1. lovable.dev → Create new project.
2. Paste everything below the `---` line of `api/LOVABLE_PROMPT.md` as the first message.
3. Attach `api/mockups/dashboard.png` and `api/mockups/network.png` to that same message
   (image button in the chat input) — the prompt references them as style targets.
4. Let it build, then in the preview: settings icon → paste the tunnel URL → run the
   "Calm baseline" preset to verify end-to-end.
5. Iterate in the Lovable chat ("make the network nodes larger", "add x"). One thing at a
   time works best. If streaming looks broken, tell it: "consume /simulate_stream as
   NDJSON via fetch ReadableStream, do not wait for the full response, do not re-mount
   charts per message."

No Lovable paid plan is required for building/preview; the free tier's daily message limit
is the only constraint, so paste the full prompt in one message.

## 4. GitHub + fully local (most reliable for demo day)

The Lovable project is connected to GitHub as
[`siddharthyashkukreja-cloud/ccp-risk-explorer`](https://github.com/siddharthyashkukreja-cloud/ccp-risk-explorer)
(frontend only — the thesis repo is untouched). To run it locally:

```bash
git clone https://github.com/siddharthyashkukreja-cloud/ccp-risk-explorer.git
cd ccp-risk-explorer
npm install
npm run dev        # http://localhost:5173
```

Set the app's API URL to `http://localhost:8000` — frontend and backend both local, no
tunnel, no internet dependency in the meeting. Edits pushed to the repo sync back into
Lovable and vice versa.

## 5. Demo-day checklist

```bash
# terminal 1
cd /Users/siddharth/Desktop/GitHub/mscthesis_abm && uvicorn api.server:app --port 8000
# terminal 2 (skip if running the frontend locally)
cloudflared tunnel --url http://localhost:8000
# terminal 3 (local frontend instead of the tunnel)
cd /Users/siddharth/Desktop/GitHub/mscthesis_abm/ccp-risk-explorer && npm run dev   # http://localhost:5173, API URL http://localhost:8000
```

- Update the API URL in the app if the tunnel URL changed.
- Warm-up run: "Calm baseline" preset (also pre-loads the FV CSVs).
- Amphetamine/caffeinate the Mac so it doesn't sleep mid-stream: `caffeinate -dis` in a
  spare terminal.
- Keep `synthetic_ensemble.ipynb` open in a tab for calibration questions.
- Showcase pair: "Synthetic stress path" (seed 171, recovery 0.80) vs "Disorderly
  close-out" (recovery 0.50) — 5 member defaults / no waterfall vs 14 defaults / L5.
