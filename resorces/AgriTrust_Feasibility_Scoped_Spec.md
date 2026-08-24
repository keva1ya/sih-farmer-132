# AgriTrust: Market Intelligence & Trade-Enablement Platform for Farmers
## Feasibility-Scoped Build Specification (v2)

> This version restructures the original design around one rule: every module is labeled as either **BUILD** (will run live in the demo) or **ROADMAP** (designed and pitched, not built). Judges respect honest scoping far more than an unlabeled wall of ambitious features.

---

## 1. Executive Summary & Problem Framing

Smallholder farmers and Farmer Producer Organizations (FPOs) face systemic friction across agricultural supply chains:

- **Information asymmetry** — farmers see a raw mandi price, not what they'd actually net after freight, labor, and market fees.
- **Weak bargaining power** — fragmented 5–20 quintal lots can't attract institutional buyers who need 200–500 quintal uniform-grade lots.
- **Subjective quality discounting** — informal grading at the farm gate leaves room for arbitrary price cuts by local traders.
- **Settlement uncertainty** — payment delays and disputes with no transparent trail.

**AgriTrust** addresses the three of these that are genuinely solvable in a hackathon timeline — price transparency, aggregation leverage, and a quality-trust signal — and designs (without over-claiming to build) the trust/verification and settlement layers needed for a production version.

---

## 2. What We're Building vs. What We're Designing Toward

This table is the single most important slide in your deck. State it explicitly and early — it preempts the "is this actually working?" question before a judge asks it.

| Module | Hackathon demo | Status |
|---|---|---|
| Net Realization calculator | Real Agmarknet price data, real freight/cost formula | ✅ **BUILD** |
| Sell-now vs. hold advisory | Working forecast model, backtested against real historical prices | ✅ **BUILD** |
| FPO pooling + trade-off simulator | Working spatial grouping on real/sample farmer data, real trade-off math | ✅ **BUILD** |
| Quality check (simplified) | Coin-calibrated single-pass measurement → 3-tier grade | ✅ **BUILD**, explicitly simplified |
| Buyer matching & listings | Working match logic against a mock buyer database | ✅ **BUILD**, buyer data is simulated |
| Payment status tracker | UI shows a deal moving through stages (agreed → dispatched → delivered → paid) | ✅ **BUILD**, no real funds move |
| Identity & land verification | Phone + name only for demo login | ⚪ **ROADMAP** — full API Setu/NGDRS integration |
| Full AGMARK-grade CV assaying | — | ⚪ **ROADMAP** — watershed segmentation, multi-parameter defect analysis |
| Real escrow & automated disbursement | — | ⚪ **ROADMAP** — requires a licensed payment aggregator partnership, flagged as a real regulatory dependency, not just an engineering task |
| Dispute arbitration workflow | — | ⚪ **ROADMAP** — described in the pitch, not functional |

---

## 3. MVP System Architecture

```mermaid
graph TB
    subgraph UI["User Interaction Layer"]
        Mobile["Farmer & FPO Mobile/Web App<br/>Lot listing, price advisory, pooling"]
        Web["Buyer Web Portal<br/>Lot discovery, offers"]
    end

    subgraph API["Application Layer"]
        Gateway["API Gateway<br/>Auth, role-based access"]
    end

    subgraph Core["Core Engines (BUILD)"]
        Mod1["Net Realization &<br/>Sell-Window Advisory"]
        Mod2["FPO Pooling &<br/>Trade-off Simulator"]
        Mod3["Simplified Quality Check<br/>Coin-calibrated CV"]
        Mod4["Buyer Matching<br/>Rule-based scoring"]
        Mod5["Payment Status Tracker<br/>UI-only workflow"]
    end

    subgraph Data["Data Layer"]
        DB[("PostgreSQL<br/>Users, listings, pools, contracts")]
        TS[("Time-series store<br/>Historical mandi & weather")]
        Blob[("Media store<br/>Quality-check photos")]
    end

    Mobile --> Gateway
    Web --> Gateway
    Gateway --> Mod1 --> TS
    Gateway --> Mod2 --> DB
    Gateway --> Mod3 --> Blob
    Gateway --> Mod4 --> DB
    Gateway --> Mod5 --> DB
```

---

## 4. MVP Module Specs

### Module 1: Net Realization & Sell-Window Advisory — ✅ BUILD

Raw mandi prices mislead farmers because they ignore freight, labor, and market fees. This module computes the price a farmer would *actually* net.

```mermaid
flowchart TD
    subgraph Feeds["External Data"]
        P_Feed["Agmarknet / eNAM<br/>Modal prices"]
        W_Feed["IMD<br/>Rainfall forecast"]
    end

    subgraph Engine["Net Realization Engine"]
        Inputs["Farmer location + lot volume"]
        P_Feed --> Calc
        Inputs --> Calc["Net Price = Spot − Freight/qtl − Labor/qtl − Fees"]
        Calc --> Rank["Rank candidate mandis by net payout"]
    end

    subgraph Forecast["Sell-Window Advisory"]
        W_Feed --> Model["Simple time-series model<br/>7/14-day moving average + trend"]
        P_Feed --> Model
        Model --> Compare{"Expected price rise ><br/>holding cost?"}
        Compare -->|Yes| Hold["HOLD advisory<br/>with projected gain"]
        Compare -->|No| Sell["SELL NOW advisory<br/>top-ranked mandi"]
    end
```

**Net Realization formula:**

$$\text{Net Realization}_m = P_m - \left( \frac{\text{Distance}_m \times C_{km} + C_{fixed}}{V} \right) - \left( P_m \times (R_{cess} + R_{comm}) \right)$$

**Sell-window model — build the simple version first:** a 7/14-day moving average and trend line, backtested against last month's real Agmarknet prices so you have an actual accuracy number to quote. Upgrade to Prophet/LightGBM only if time remains — a working simple model beats a fragile advanced one in a live demo.

---

### Module 2: FPO Pooling & Trade-off Simulator — ✅ BUILD

```mermaid
flowchart TD
    subgraph Listings["Fragmented supply"]
        L1["Farmer A: 12 qtl"]
        L2["Farmer B: 18 qtl"]
        L3["Farmer C: 25 qtl"]
    end

    L1 & L2 & L3 --> Group["Group same commodity + grade<br/>within a set radius (haversine distance)"]
    Group --> Pool["Form pool: 55+ quintals"]
    Pool --> Sim["Trade-off simulator"]
    Sim --> M1["Price: institutional premium"]
    Sim --> M2["Freight: shared bulk transport"]
    Sim --> M3["Payout: staged vs immediate"]

    Sim --> Choice{"Farmer opts in?"}
    Choice -->|Yes| Lock["Commit lot to pool"]
    Choice -->|No| Solo["Solo farm-gate sale"]
```

**Build note:** use straightforward haversine-distance grouping in application code rather than a PostGIS extension — it's just as effective for a hackathon demo and removes a setup dependency. PostGIS is a fine upgrade path for the roadmap slide.

**Keep the trade-off table from the original design** — concrete numbers (₹/quintal, % freight savings) are one of your strongest demo moments:

| Dimension | Solo sale | Pooled sale |
|---|---|---|
| Price realization | Baseline spot price | Institutional premium |
| Freight cost | Higher (small lot) | Lower (shared bulk transport) |
| Payment timing | Immediate | Staged, tied to delivery confirmation |
| Rejection risk | 100% individual | Shared across pool |

---

### Module 3: Simplified Quality Check — ✅ BUILD (deliberately scoped down)

The original watershed-segmentation, multi-parameter AGMARK pipeline is a research-level CV problem on its own — too fragile to run live and reliably in front of judges. This version keeps the clever part (coin calibration) and drops the risky part (separating touching/clustered grains).

```mermaid
flowchart TD
    A["Photo: grain sample on a plain sheet<br/>next to a standard coin, spread out (not touching)"] --> B["Detect coin via Hough Circle Transform<br/>Compute pixels-per-mm"]
    B --> C["Detect grain contours<br/>(simple thresholding, no watershed)"]
    C --> D["Measure average length/width<br/>across detected grains"]
    C --> E["HSV color check<br/>estimate % discolored"]
    D --> F["Map to simplified grade: A / B / C"]
    E --> F
```

**Be explicit about this in the pitch:** *"This is a decision-support proxy for farmers and FPOs, not a certified AGMARK lab replacement — full multi-parameter certified grading is on our roadmap."* That one sentence turns a limitation into a sign of maturity.

---

### Module 4: Buyer Matching — ✅ BUILD (mock buyer data)

Rule-based scoring against a small synthetic buyer dataset (15–20 buyers with commodity needs, location, and a reliability score). State plainly that buyer data is simulated — no public buyer-demand dataset exists, so this isn't a gap you created.

### Module 5: Payment Status Tracker — ✅ BUILD (UI only)

A visible deal-status flow — **Agreed → Dispatched → Delivered → Paid** — with no real funds moving. This demonstrates the transaction lifecycle judges want to see without the licensing complexity of an actual escrow/payment integration.

---

## 5. Roadmap Modules (pitch, don't build)

- **Multi-tier identity & land verification** — phone/PAN plus a real API Setu/NGDRS land-record integration, hardware-bound geofencing for photo capture.
- **Full escrow & automated disbursement** — requires partnering with a licensed payment aggregator; explicitly name this as a regulatory dependency in your pitch, not just an engineering one.
- **Production-grade CV grading** — watershed segmentation for touching grains, full AGMARK multi-parameter defect analysis, at scale.
- **FPO arbitration workflow** — structured dispute resolution with SLAs.

Presenting these as a clear-eyed roadmap (not vague hand-waving) is what earns you the "scalable beyond a hackathon" evaluation point without pretending they're built.

---

## 6. Build Priority

1. **Net Realization engine** — pull real Agmarknet data, implement the formula (data lead)
2. **Sell-window advisory** — simple moving-average model, backtested (ML lead)
3. **FPO pooling + trade-off simulator** — haversine grouping, trade-off math (ML/backend lead)
4. **Simplified quality check** — coin calibration + basic contour measurement (CV lead)
5. **Buyer matching + payment status UI** — wire the full demo flow together (backend + frontend)
6. **Rehearsal** — one clean end-to-end story, with the "build vs. roadmap" table ready for Q&A

---

## 7. Data Model (MVP scope only)

```mermaid
erDiagram
    USERS ||--o{ COMMODITY_LISTINGS : creates
    USERS ||--o{ FPO_MEMBERSHIPS : holds
    FPOS ||--o{ FPO_MEMBERSHIPS : contains
    FPOS ||--o{ FPO_POOLS : manages
    FPO_POOLS ||--o{ COMMODITY_LISTINGS : aggregates
    COMMODITY_LISTINGS ||--o{ TRADE_CONTRACTS : fulfills
    USERS ||--o{ TRADE_CONTRACTS : buys

    USERS {
        uuid id PK
        string phone_number UK
        string role
        string full_name
    }

    FPOS {
        uuid id PK
        string fpo_name
        geometry hub_location
    }

    FPO_POOLS {
        uuid id PK
        uuid fpo_id FK
        string target_commodity
        float target_quantity_qtl
        float current_aggregated_qtl
        string pool_status
    }

    COMMODITY_LISTINGS {
        uuid id PK
        uuid seller_id FK
        uuid pool_id FK
        string commodity_name
        float quantity_quintals
        geometry origin_location
        string simplified_grade
        string listing_status
    }

    TRADE_CONTRACTS {
        uuid id PK
        uuid buyer_id FK
        uuid listing_id FK
        float agreed_price_per_qtl
        string payment_status
    }
```

Note what's gone from the original: no `LAND_HOLDINGS` cadastral fields, no `ESCROW_LEDGER` banking fields — `payment_status` is a simple enum (`agreed`/`dispatched`/`delivered`/`paid`), matching what's actually built.

---

## 8. Demo Flow

```mermaid
sequenceDiagram
    actor Farmer
    participant App as AgriTrust
    actor Buyer

    Farmer->>App: List lot (crop, quantity, location)
    Farmer->>App: Capture quality-check photo
    App-->>Farmer: Simplified grade + Net Realization across mandis
    App-->>Farmer: Sell-window advisory (sell now / hold)
    Farmer->>App: Join FPO pool
    App-->>Farmer: Trade-off simulator (solo vs pooled)
    Buyer->>App: Discover pooled lot, send offer
    Farmer->>App: Accept offer
    App-->>Farmer: Payment status: Agreed → Dispatched → Delivered → Paid
```
