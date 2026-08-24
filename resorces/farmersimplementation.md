# AgriTrust — Build Plan (Current Plan & Future Plan)

**Team:** 6 people, 1 per role · **Client:** Plain React SPA (no PWA, no offline) · **Auth:** None — role-switcher only · **Escrow:** Mocked state machine (no real payment gateway) · **Backend:** FastAPI · **DB:** Supabase Postgres + PostGIS

---

## 0. Current Scope

**Building now:** Listing creation → CV grading (dynamic coin calibration, blur-retry) → Net Realization comparison → Pooling (spatial clustering) → mocked contract/escrow state machine → mocked gate-in tolerance check → rule-based sell-window advisory.
**Deferred:** see the [Future Plan](#8-future-plan-by-role) section at the end — real auth, real payments, trained ML forecasting, offline sync, i18n, PWA, automatic coin-denomination recognition, and more, broken out per role.

---

## 1. FRONTEND — React SPA

### Stack
- React 18 + Vite, TypeScript, Tailwind CSS
- React Router v6
- TanStack Query for API calls
- MapLibre GL JS + OpenStreetMap tiles for the pooling/discovery map
- `react-webcam` for coin+grain photo capture
- `qrcode.react` (display) + `html5-qrcode` (scan)
- Recharts for the net-realization comparison chart

### No-auth mechanism
A simple screen on load: "Who are you?" → Farmer / FPO Admin / Buyer / Admin. Selection is stored in `localStorage` as `{role, user_id}` (a handful of hardcoded demo user IDs seeded by DB) and sent as headers (`X-Role`, `X-User-Id`) on every API call. No login form, no password, no token refresh.

### Blur-retry flow
Before submitting a captured photo, run a quick client-side blur check (Laplacian variance via a small helper, or just forward to CV and read its response). If the CV service flags the image as `too_blurry`, don't treat it as a hard failure — show "Image too blurry, please retake" and re-open the camera. No EXIF/hardware validation, just this one retry loop.

### Coin selection (dynamic calibration)
Add a dropdown/selector before capture: "Which coin are you using as reference?" (₹1 / ₹2 / ₹5 / ₹10). Send the selected denomination to the Backend/CV service along with the photo, instead of assuming a fixed coin.

### Core Deliverables
1. Role switcher, listing form, camera capture with coin-denomination selector and blur-retry
2. CV grade result screen, net-realization comparison chart + pooling map
3. Buyer accept + mocked escrow flow, QR scan gate-in check

---

## 2. BACKEND / API — FastAPI

### Stack
- FastAPI, Uvicorn, SQLAlchemy 2.0 (async) + Alembic, `asyncpg`, GeoAlchemy2, `httpx`

### No-auth mechanism
A lightweight dependency reads `X-Role` / `X-User-Id` headers and injects them as the "current user" — no verification, no middleware chain, no JWT parsing. Document clearly in the repo README that this is demo-only and would be replaced by real auth in production.

### Core Endpoints
1. `POST /listings` — create listing (skip yield-cap/land-sandbox check entirely, or keep as a one-line stub returning a fixed cap — your call, low effort either way)
2. `POST /listings/{id}/analyze` — forwards photo + selected coin denomination to CV service, stores grade + generates QR hash, surfaces `too_blurry` retry signal from CV back to Frontend unchanged
3. `GET /net-realization` — calls ML service, returns ranked mandis + rule-based HOLD/SELL advisory
4. `POST /pools` — runs `ST_DWithin` query, forms/updates a pool
5. `POST /contracts` — buyer accepts listing/pool → creates contract + mocked `ESCROW_LEDGER` row (`status = FundsLocked` immediately, no real payment call)
6. `POST /contracts/{id}/gate-in` — accepts a photo, calls CV `/analyze` again, compares to original measurements within ±3%, sets `Disbursed` or `DisputeState`

### Core Deliverables
1. Scaffold, DB models, listing + CV orchestration endpoints
2. Net-realization + pooling endpoints
3. Contract/escrow mock + gate-in endpoint, integration testing with Frontend

---

## 3. DEVOPS

### Stack
- GitHub monorepo, Docker Compose (local Backend + CV + ML + local Postgres/PostGIS for dev)
- Supabase project (managed DB only — Auth feature unused)
- Render (Backend, CV, ML as 3 separate free-tier services) + Vercel (Frontend)

### Core Deliverables
1. Repo scaffolded, Supabase project + PostGIS enabled, Docker Compose working, connection strings shared — this is the #1 blocker for everyone else, prioritize it first
2. Render/Vercel deploys live and auto-deploying from `main`
3. Final deploy freeze before demo, quick smoke test of the full flow in prod

---

## 4. DB

### Stack
- Postgres 15 + PostGIS via Supabase, Alembic migrations

### Schema (trimmed ERD)
Keep: `USERS` (no password/auth fields — just `id, phone_number, role, full_name`), `LAND_HOLDINGS`, `FPOS`, `FPO_POOLS`, `COMMODITY_LISTINGS`, `TRADE_CONTRACTS`, `ESCROW_LEDGER`.
Drop: `FPO_MEMBERSHIPS` as a separate table — fold into a simple `fpo_id` FK on `USERS` if a farmer belongs to one FPO, which is enough for now.
Add: `mandi_price_history (commodity, mandi_name, mandi_location geometry, modal_price, arrival_date)` — needed for the rule-based advisory and net-realization ranking.

### Seed Data (critical path)
- 5–10 hardcoded demo users covering all 4 roles (Frontend's role-switcher points at these)
- 1 commodity (Wheat), ~10 mandi price records with varying location/price so ranking looks real
- 4–5 farm locations clustered within 15km of one FPO hub, so the pooling query has something to actually cluster

### Core Deliverables
1. Schema + seed data live — this is the #1 blocker for CV/ML/Backend demo quality, prioritize it first
2. On-call for query/index issues Backend hits

---

## 5. CV

### Stack
- FastAPI microservice, OpenCV (`opencv-python-headless`), scikit-image, NumPy

### Pipeline
1. **Blur check first:** Laplacian variance (`cv2.Laplacian(img, cv2.CV_64F).var()`) against a threshold — if below threshold, return `{status: "too_blurry"}` immediately and skip the rest of the pipeline. No EXIF/hardware checks, just this one gate.
2. **Dynamic coin calibration:** accept a `coin_denomination` param from the request (₹1/₹2/₹5/₹10) and look it up in a small hardcoded diameter table (official RBI diameters — verify exact current figures before hardcoding, roughly: ₹1 ≈ 25mm, ₹2 ≈ 27mm, ₹5 ≈ 23mm, ₹10 ≈ 27mm bimetallic) instead of assuming one fixed coin. Circular Hough Transform (`cv2.HoughCircles`) still detects the coin's pixel radius; `pixels_per_mm = detected_diameter_px / known_diameter_for_selected_coin`.
3. Adaptive thresholding → isolate grains
4. Watershed → separate touching kernels → contours
5. `minAreaRect` per contour → length/width in mm → length/width ratio
6. Simple HSV threshold pass for discoloration (skip the more elaborate LAB defect analysis — HSV alone is enough to demo)
7. Aggregate broken ratio + defect % → map to hardcoded AGMARK Wheat thresholds (from the original spec) → Grade A/B/C/Reject
8. Return SHA-256 hash of the result JSON

**Note on "dynamic" coins:** this makes calibration dynamic (user tells the system which coin, system looks up the right known size) — it is not automatic coin *recognition* (the CV service doesn't visually identify which denomination is in frame on its own). True auto-recognition is possible with OpenCV but needs either a trained classifier on coin appearance (color/size/edge pattern) or multi-coin size-clustering heuristics, both of which are real effort — that's listed under Future Plan below rather than built now.

### Core Deliverables
1. Coin detection + dynamic calibration (denomination lookup table) working on sample photos
2. Full segmentation → grading pipeline with blur-retry gate, exposed as `POST /analyze`
3. Robustness pass on whatever sample photos the demo will actually use — test on the exact set you'll show

---

## 6. ML

### Stack
- FastAPI microservice, Pandas, `geopy` (Haversine distance)

### Current Logic — Phase 1, rule-based (no model training)
1. **Net Realization** (pure formula, keep exact): `Net_m = P_m − ((Distance_m × C_km + C_fixed) / V) − (P_m × (R_cess + R_comm))` — rank mandis from seed data. This alone is enough to power the Solo-vs-Pool comparison screen.
2. **Sell-Window advisory** (heuristic, not ML):
   - Compute trailing 30-day average price and simple linear trend (slope) from `mandi_price_history` via `numpy.polyfit`
   - If trend slope is positive and projected gain over holding cost (same formula as before, storage cost can be a flat assumed rate) → `HOLD`
   - Else → `SELL NOW` with the top-ranked mandi from step 1
   - Honest framing for the demo: "a rule-based Phase 1 — the full spec calls for LightGBM/SARIMAX time-series forecasting, which is Phase 2 (see Future Plan)"

### Core Deliverables
1. Net Realization ranking against seed data
2. Trend heuristic + HOLD/SELL logic
3. Sanity-check outputs look reasonable against the worked example numbers in the original spec

---

## 7. Cross-Cutting API Contracts

Freeze these early so roles aren't blocked waiting on each other:
- `POST /internal/cv/analyze` — request: image, `coin_denomination`; response: either `{status: "too_blurry"}` or `{measurements, grade, hash}`
- `POST /internal/ml/net-realization` — request: location, commodity, volume; response: ranked mandis + `{advisory: "HOLD"|"SELL_NOW", ...}`

---

## 8. Future Plan (by role)

Everything below is explicitly **out of current scope** — captured here so it's not forgotten and can be picked up post-demo.

### Frontend
- Real authentication (phone OTP via Supabase Auth or similar) replacing the role-switcher
- PWA install support + offline sync (IndexedDB queue for spotty rural connectivity)
- Vernacular UI / i18n (Hindi + regional languages)
- Native mobile app (Flutter or React Native) if a true installable app is wanted beyond the web client

### Backend
- Real RBAC/JWT verification replacing the trusted-header approach
- Yield-cap calculation wired to a real (or realistically mocked) land-record lookup, with the FPO endorsement workflow for over-cap listings
- Background job queue (Celery + Redis) for async recalculation and escrow timeout handling, replacing inline request-time logic
- Full audit logging on contract/escrow state transitions

### DevOps
- CI pipeline (lint + test on every PR)
- Error tracking (Sentry) and structured logging/monitoring
- Load testing on the pooling and gate-in endpoints
- Custom domain + SSL, staging vs prod environment separation

### DB
- `FPO_MEMBERSHIPS` as its own table again if a farmer needs to belong to multiple FPOs
- TimescaleDB hypertable for `mandi_price_history` if volume grows beyond what a plain indexed table handles well
- Full production-scale seed/import pipeline from real Agmarknet/e-NAM data dumps

### CV
- Real image integrity checks: EXIF metadata verification, hardware GPS/timestamp attestation, live-capture-only enforcement (disabling gallery upload) — the full anti-fraud hardware-binding described in the original spec
- **Automatic coin denomination recognition** — instead of the user selecting the coin, classify it from the image itself (feasible in OpenCV via a combination of relative size clustering, edge/color features, or a small trained classifier — real effort, not a quick add)
- LAB color-space defect analysis and separate foreign-matter/chaff detection pass, not just HSV
- Separate lightweight/fast "60-second gate-in spot check" variant instead of reusing the full `/analyze` pipeline
- Support for additional commodities beyond Wheat (their own AGMARK threshold matrices)

### ML
- **Phase 2:** Replace the rule-based heuristic with the originally-specified LightGBM (or SARIMAX fallback) time-series forecaster, trained on real historical price + weather data
- Live external data ingestion: Agmarknet/e-NAM price feeds, Open-Meteo/IMD weather feeds, WDRA warehouse capacity data — replacing seeded/mocked data
- Model evaluation/backtesting harness to validate HOLD/SELL advisory accuracy over time
- Road-network-based distance (instead of Haversine) for more accurate freight cost estimation

### Payments/Escrow (cross-cutting)
- Real Razorpay (or equivalent) sandbox integration with webhook-driven settlement, replacing the mocked instant `FundsLocked` transition
- Real bank disbursement rails for the "automated direct disbursement to farmer bank accounts" step
