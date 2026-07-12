# Lovable prompt — Client-Clearing ABM live demo (v2: dark, streaming, network page)

Copy everything below the line into Lovable, attaching the two mockup images
(`api/mockups/dashboard.png`, `api/mockups/network.png`) as style reference.

Run the backend first, from the repo root:

```bash
pip install fastapi uvicorn
uvicorn api.server:app --port 8000
# in a second terminal (brew install cloudflared):
cloudflared tunnel --url http://localhost:8000   # paste the https URL into the app
```

---

Build a two-page dark-theme dashboard called **"Client-Clearing ABM — CCP Systemic Risk
Simulator"**. It runs an agent-based model of a centrally-cleared futures market (E-mini
S&P 500) **live, streamed day by day** from a REST API, and animates the market path, margin
dynamics, and a clearing-network map as the simulation progresses. Match the attached
mockups.

**Theme** (exact): page bg `#0A0E17`, panels/cards `#121A2B` with 1px border `#1E2A44`,
radius 10px. Text `#E6ECF8` primary / `#8DA2C4` secondary. Accent cyan `#38BDF8` (price,
primary buttons), fundamental line `#64748B`, IM area `#38BDF8` at 25% opacity, margin-call
bars `#F59E0B`, danger red `#F43F5E` (defaults, waterfall), frozen amber `#F59E0B`, healthy
green `#34D399`. Font: Inter; numbers in tabular-nums. Recharts for charts, plain SVG for
the network map. No light mode.

## API

Base URL editable via a settings icon in the header (default `http://localhost:8000`).

`GET {base}/meta` → defaults/ranges for the panel (same shape as v1):

```json
{"defaults": {"regime": "stressed", "fv_source": "empirical", "ann_vol_pct": 40.5,
              "recovery": 0.8, "horizon_days": 20, "seed": 42},
 "ranges": {"ann_vol_pct": [10, 90], "recovery": [0.3, 1.0],
            "horizon_days": [3, 75], "seed": [0, 10000]},
 "calibrated": {"synthetic_ann_vol_pct": 40.5, "joint_loss_D": 6.73,
                "empirical_days": {"calm": 75, "stressed": 73}}}
```

`POST {base}/simulate_stream` — the primary endpoint. Body:

```json
{"regime": "stressed", "fv_source": "synthetic", "ann_vol_pct": 40.5,
 "recovery": 0.8, "horizon_days": 42, "seed": 171}
```

Response is **NDJSON** (`application/x-ndjson`): one JSON object per line, streamed while
the model runs (~0.3 s per simulated day). Consume with `fetch` + `ReadableStream`:

```js
const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body});
const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
while (true) {
  const {done, value} = await reader.read(); if (done) break;
  buf += dec.decode(value, {stream: true});
  const lines = buf.split("\n"); buf = lines.pop();
  for (const l of lines) if (l.trim()) handleMessage(JSON.parse(l));
}
```

Three message types:

1. `{"type":"init", "n_days":42, "regime":..., "ann_vol_pct":..., "day_starts":[0,390,...],
   "agents":[{"id":105,"kind":"CCP","cm_id":null,"cash0":1.5},
             {"id":30,"kind":"BCM","cm_id":105,"cash0":6.4,"n_clients":11},
             {"id":0,"kind":"FT","cm_id":102,"cash0":0.5}, ...]}` — the roster: 1 CCP,
   10 BCM + 5 NBCM (all `cm_id` = CCP id), 90 clients (FT/MT/ZI, `cm_id` = their clearing
   member). `cash0` in $B. Build the network layout and empty charts from this.
2. `{"type":"day", "day":3,
    "series": {"t":[...], "mid":[...], "fundamental":[...]},
    "margin_cycles": {"t":[...], "posted_im_usd_b":[...], "margin_calls":[...],
                      "min_member_capital_ratio":[...]},
    "events": {"client_defaults":[{"t":1520,"client_id":17,"kind":"MT","cm_id":104,
                                   "shortfall_usd":1250000,"closeout_loss_usd":310000}],
               "member_defaults":[{"t":9822,"agent_id":104,"kind":"NBCM"}]},
    "agent_states":[{"id":30,"cash":6.375,"pos":2671,"im":4.339,"frozen":0,"defaulted":0,
                     "cr":0.1251}, ...]}` — append series points, append margin cycles,
   append events, replace per-agent state (all 106 agents, every day). `cash`/`im` in $B,
   `pos` in lots, `cr` = capital ratio (members only, null otherwise).
3. `{"type":"done", "summary": {"drawdown_pct":-37.3, "client_defaults":48,
    "member_defaults":5, "nbcm_defaults":3, "deepest_waterfall":3,
    "waterfall_label":"L3 pooled default fund", "mutualised":true, "im_peak_usd_b":61.2,
    "im_mean_usd_b":38.4, "total_df_usd_b":4.1, "ccp_cash_used_usd_b":0.0,
    "runtime_s":12.4, ...}}` — fill the summary cards.

Also `POST {base}/simulate` (same body) returns everything in one response, with
`topology_final` = final `agent_states`; use it as a fallback if streaming errors.
Non-200: show the error `detail` (e.g. synthetic requires stressed). Store every day's
`agent_states` in memory — after the run finishes a **time scrubber** replays the network
day by day.

## Page 1 — Dashboard

**Left sidebar** (sticky, ~300 px):

- **Regime**: segmented Calm / Stressed.
- **Fundamental value**: segmented "Historical ES" / "Synthetic (2-factor SV)". Picking
  Synthetic while Calm auto-switches to Stressed with a caption.
- **FV volatility (annualised)**: slider 10–90 %, default 40.5 %, tick "calibrated 40.5%".
  Enabled only for Synthetic; disabled caption otherwise: "set by history — realised value
  shown in results".
- **Close-out recovery**: slider 0.30–1.00, default 0.80; captions 0.60 "disorderly
  (Nasdaq 2018)", 0.80 "orderly (baseline)".
- **Horizon**: slider 3–75 sessions. **Seed**: number input + dice button.
- **Run simulation** (primary, full width). While streaming it becomes a progress bar
  filling by `day/n_days` with "Day 17 / 42 — 3 client defaults". A Stop button aborts
  the fetch.
- **Presets** (fill panel + auto-run):
  - "Calm baseline" — calm, historical, 20 d, seed 42
  - "COVID replay" — stressed, historical, 73 d, seed 42
  - "Synthetic stress path" — stressed, synthetic, 42 d, seed 171, recovery 0.80
  - "Disorderly close-out" — same but recovery 0.50

**Main column**, top to bottom:

1. **Summary cards**: Drawdown %, Client defaults, Member defaults, Deepest waterfall
   (badge: grey none / amber L1–L2 / red L3–L4 / dark-red L5, shows the label), Peak
   posted IM $B, Default fund $B, CCP capital used $B. Client/member default cards tick
   up live during streaming; the rest fill at `done`.
2. **Waterfall meter**: horizontal 5-segment bar (L1 defaulter resources → L2 CCP
   skin-in-the-game → L3 pooled default fund → L4 survivor cash → L5 CCP capital), filled
   red up to `deepest_waterfall`, animated when it deepens mid-run. Caption: "L3+ = losses
   mutualised across surviving members."
3. **Price chart**: `mid` (cyan) + `fundamental` (thin slate) vs t (x-axis in sessions =
   t/390; faint vertical gridlines at `day_starts`). Client defaults = small red dots on
   the line; member defaults = red diamonds. Grows rightward as days stream in, x-domain
   fixed to the full horizon from the start.
4. **Margin panel** — two stacked charts, shared x:
   - Posted IM ($B, stepped cyan area) — the procyclical ratchet.
   - **Margin-call intensity**: amber bars (calls per hourly cycle) with a 1-day rolling
     mean line; this is the "margin calls cluster in bursts" exhibit.
5. **Capital ratio chart**: minimum member capital ratio (line) with dashed reference
   lines at 8 % (CFTC Reg 1.17, slate) and 4.25 % (Basel LR floor, red). Shade the region
   below 4.25 % faint red.
6. **Cumulative defaults**: step chart, clients (amber) and members (red), built from
   event timestamps.
7. **Events table**: tabs Client / Member defaults (time in sessions, kind, CM, shortfall $,
   close-out loss $). Empty state: "No defaults — clearing absorbed the stress."
8. Footer: "Agent-based model of client clearing — MSc thesis, VU Amsterdam. Synthetic
   paths: two-factor log-vol fundamental, jointly calibrated with agent parameters
   (surrogate-assisted SMM, D = 6.73, 9/10 moments in empirical 95% CI)."

## Page 2 — Clearing network

Radial topology map (plain SVG, ~800 px square, pannable not required):

- **CCP** at centre: large hexagon, cyan outline, label "CCP".
- **15 clearing members** on an inner ring: BCMs as squares, NBCMs as diamonds, radius
  scaled by `sqrt(cash0)`. Spokes to the CCP.
- **90 clients** on an outer ring: small circles (FT/MT/ZI subtly different hues of
  slate/cyan), each placed within an arc segment behind its clearing member, thin edge
  to its member. Five of the BCMs have no clients (own-account only) — they simply have
  no outer arc.
- **State colouring**, updated every streamed day: healthy = default hue; `frozen=1` =
  amber fill; `defaulted=1` = red fill (persists); members whose `cr` < 0.0425 get a
  pulsing amber outline. CCP turns red only if `ccp_cash_used_usd_b > 0`.
- **Hover tooltip** (dark card): id, kind, cash $B, position (lots), posted IM $B,
  capital ratio (members), status. Edge from the hovered node to its CM highlights.
- **Node size pulse** on a default event that day.
- **Time scrubber** under the map: during streaming it follows the live day; afterwards
  it replays any day from the stored `agent_states`. Play/pause button, 2 days/s.
- Legend box: shapes, colours, and the L1–L5 waterfall strip mirrored from page 1.
- A compact stats strip above the map: current day, live client/member default counts,
  posted IM.

Keep all state client-side; no database, no auth. Format $ as `$12.4B`, percentages 1 dp,
days as integers. Charts must not re-mount on each streamed message — append data.
