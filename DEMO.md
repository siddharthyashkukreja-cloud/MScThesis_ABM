# Demo kit — presenting the thesis to Krishnen Vytelingum

Everything built for the demo, how to run it, and what to say. Slot: 15–20 min.
Focus: the **joint FV + agent calibration via XGBoost surrogate** (thesis §4.5) — the novel
contribution and the part closest to Krishnen's market-simulation interests. Flow: Lovable
frontend first (live simulation), then the Jupyter notebook (method + results).

## The three artifacts

| Artifact | Where | What it does |
|---|---|---|
| **Live simulator API** | `api/server.py` (this repo) | FastAPI wrapping the model; streams runs day-by-day as NDJSON (~0.3 s/session) |
| **Frontend** | `~/Desktop/GitHub/mscthesis_abm/ccp-risk-explorer` → GitHub `siddharthyashkukreja-cloud/ccp-risk-explorer` (two-way Lovable sync) | Dark two-page dashboard: charts page + clearing-network map, parameter panel with clearing levers |
| **Notebook** | `demo_walkthrough.ipynb` (repo root, executed, outputs saved) | Method walkthrough: model → calibration machinery → joint calibration (centrepiece) → two-horizon vol → scenario engine |

Supporting: `api/LOVABLE_PROMPT.md` (the prompt that built the frontend),
`api/mockups/{dashboard,network}.png` (design refs), `api/SETUP.md` (original setup notes —
superseded by this file).

## Run it (demo-day terminals)

```bash
# Terminal 1 — backend
cd ~/Desktop/GitHub/mscthesis_abm
uvicorn api.server:app --port 8000

# Terminal 2 — frontend
cd ~/Desktop/GitHub/mscthesis_abm/ccp-risk-explorer
npm run dev            # open the URL it prints (usually http://localhost:3000)

# Terminal 3 — keep the Mac awake
caffeinate -dis
```

- App settings → API URL = `http://127.0.0.1:8000` (numeric, not `localhost` — Safari IPv6 quirk).
- Hard-reload the app (Cmd+Shift+R) after pulling frontend changes.
- Warm up with the "Calm baseline" preset before the call.
- Notebook: `jupyter lab demo_walkthrough.ipynb` — already executed; runs top-to-bottom in <1 min if re-run.
- Frontend sync: commit+push in `ccp-risk-explorer` → Lovable rebuilds automatically.
- The hosted Lovable preview cannot reach a local backend from Safari (https→http block);
  use the local dev URL, or Chrome, or a `cloudflared tunnel --url http://localhost:8000` URL.

## Demo runbook (~18 min)

1. **Frame (2 min).** Two-tier ABM: calibrated LOB market simulator (extends the Simudyne
   CCP ODD with a client tier, data-derived FV, calibrated placement); clearing layer all
   rulebook constants — the market layer carries the empirical burden.
2. **Lovable live (8 min).**
   - "Calm baseline" → clean; "COVID replay" → IM ratchet ≈ CME's 4→11%, benign outcome
     (matches history: March 2020 was a margin event, not a default event).
   - "Synthetic stress" (seed 171, 42 d) on the **Network page** — watch defaults propagate.
   - "Disorderly close-out" (same seed, recovery 0.50) — L5 reached, ~$21B CCP cash.
     Showcase contrast: recovery 0.95 → 5 member defaults, waterfall untouched; 0.50 → 14
     defaults, L5.
   - Clearing levers if time: Flat 12% + fixed DF (risk relocation), Direct (H1 star
     topology), vol slider (θ/FV coupling talking point — agents stay at joint optimum).
3. **Notebook (6 min).** Jump to §3: the whose-parameters-are-they argument → free/measured/
   fixed table → convergence + identification panels (1,698 committed evals) → loss ledger →
   §4 two-horizon vol (half-lives 1 vs 3.1 sessions) → ACF overlay → §5 severity-vs-vol scatter.
4. **Close (2 min).** §6 research directions: joint calibration with clearing active
   (measured non-invariance motivates it), n-factor vol via the same identification slices,
   endogenising V_t.

## Numbers cheat sheet

- Joint 9-d calibration: **D 6.73 vs 20.86 baseline** (1,698 evals, 10.2 h, surrogate held-out R² 0.94).
- Fresh-path validation: **9/10 moments in empirical 95% CI**, mean gap 0.64 vs 2.64 SD; ensemble D 18.3 vs 24.5. Lone miss: lag-1 bid-ask bounce (thin-LOB artifact).
- Two-horizon vol: α_fast measured (BNS 2004), half-life ≈ 1 session; α_slow calibrated, ≈ 3.1 sessions; slow share w ≈ 0.51; σ_d ≈ 40.5% annualised.
- Baseline stressed lock: D 6.01; in-sample argmin 5.16 **rejected** by fresh-seed re-rank (winner's curse). Calm 43.8 full-year / ≈26.5 matched-window (window artifact).
- MCR stressed: 9/10 moments, mean coverage 87%.
- Ensemble (80 paths): median 3 client defaults / 0 member / fund untouched; 15/80 mutualise; severity tracks path **volatility** more than drawdown (margin ratchet channel).
- Margin ladder (150 seeds, fixed ≈$12–14B fund): flat 4/8/12% → client defaults 14.8/11.1/6.9, member defaults 0.01/0.35/0.55, reach-fund 1/30/42%; reactive: 10.1 clients, 0.33 members, 25%, calm cost $18.1B vs flat-12's $42.3B. **Risk relocated, not removed** (freeze trap; single-connection clients — Gadgil 2026).
- Engine speed: 0.26 s per cleared session; full COVID window ~20 s.

## API quick reference

`GET /meta` — panel defaults/ranges. `POST /simulate_stream` — NDJSON: `init` (roster) →
`day` × N (series chunk, margin cycles, events, all-agent states) → `done` (summary) or
`error` (detail). `POST /simulate` — same in one response (frontend fallback).

Request fields (camelCase also accepted): `regime` calm|stressed, `fv_source`
empirical|synthetic (synthetic ⇒ stressed, auto-coerced), `ann_vol_pct` 10–90 (synthetic
only; calibrated 40.5), `recovery` 0.30–1.00, `horizon_days` 3–75, `seed` 0–10000,
`im_mode` reactive|flat, `flat_im_pct` 2–20, `im_floor_pct` 1–10, `fixed_df` bool,
`structure` tiered|direct, `netting` gross|net.

Implementation notes: synthetic runs set `IM_DAILY=False` (thesis stage-C convention);
ensemble path *i* = generation seed *100+i* (path 71 → seed 171); globals snapshot/restored
per request under a lock (409 if busy); all payloads NaN-sanitised.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "Load failed" on every run | Backend down (`curl http://127.0.0.1:8000/meta`), or API URL is `localhost` in Safari → use `http://127.0.0.1:8000`, or hosted-https app + local backend → use local dev URL |
| 404 | Wrong path — endpoints are `/meta`, `/simulate`, `/simulate_stream`, no `/api` prefix |
| 422 | Bad body — uvicorn logs the failing field and raw body |
| 409 | Previous run still holds the lock — wait a few seconds |
| Error toast "Simulation error: …" | Real model exception — the detail names it; check uvicorn log |
| UI crash panel | Click "Reset view" (error boundary) — no refresh needed |
| Tunnel URL dead | Quick tunnels rotate on every cloudflared restart — repaste, or stay local |

## Repo map (demo-related)

```
demo_walkthrough.ipynb      the notebook (executed)
DEMO.md                     this file
api/server.py               FastAPI backend
api/LOVABLE_PROMPT.md       frontend build prompt (v2)
api/SETUP.md                original setup instructions
api/mockups/*.png           dark-theme design mockups
ccp-risk-explorer/          frontend clone (own git repo — push separately; thesis repo stays unpushed)
output/overnight_joint_logvol/  joint-calibration artifacts (calibration.json = 1,698 evals, validation, ACF)
output/thesis_final/        MCR, relock ledger, 150-seed margin experiments
output/synth_results/       per-path ensemble severity data
```
