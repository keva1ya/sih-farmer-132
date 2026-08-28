# KrishiDrishti — Enhanced SIH PPT Content (matched to the Saarthi reference deck's structure)

This version upgrades every slide to match what the reference deck does well: a real system architecture with role-based access, a detailed process-flow diagram, cited supporting facts, a benefits tree, and a head-to-head competitor table.

---

## Slide 2 — Idea Title & Proposed Solution

**Idea Title:**
> **KrishiDrishti — Net-Realization Price Intelligence & Verified Buyer-Matching for Smallholder Farmers**

**Proposed Solution (left column bullets):**

- **Net-Realization Pricing Engine:** Shows farmers their true take-home price across nearby markets — spot price minus freight, labor, and market fees — not the raw mandi price current tools display.
- **Predictive Sell-Window Advisory:** Backtested price-trend model recommends sell-now vs. hold, with a projected ₹/quintal gain shown before the farmer decides.
- **FPO Trade-off Simulator:** Aggregates fragmented small lots into buyer-ready volumes and shows the farmer a transparent side-by-side comparison — solo sale vs. pooled sale — in rupees, not vague promises.
- **Deterministic Quality Check:** Coin-calibrated computer-vision grading gives an explainable A/B/C grade instead of a subjective trader's eyeball estimate or an unexplainable black-box AI score.
- **Verified Buyer Matching:** Rule-based scoring connects graded, pooled lots to institutional buyers by commodity, grade, and location.

**Innovation vs. invention:** eNAM already solved *price reporting*. KrishiDrishti is the missing *decision layer* on top of it — nobody currently combines net-realization pricing, a predictive sell-window, and a transparent pooling simulator in one farmer-facing tool.

**Diagram: System Architecture (right side) — prompt for Draw.io/Figma/AI diagram tool:**
> "Create a clean system architecture diagram on a white background, top section and bottom section. Top-left: a 'Client System' box with a mobile phone icon and browser icon, labeled 'Farmer & FPO App / Buyer Web Portal', with a two-way arrow labeled 'Req/Res' pointing right to a circular 'Authentication' icon, which connects to a central 'Web Application / Server' hexagon icon. Top-right: a 'Data Retrieval' label above small icons for a database cylinder, a cloud icon, and a lightning-bolt icon, connected to the server. Below the server, draw a table titled 'Restricted Functional Modules' with rows for five roles — Farmer, FPO Admin, Buyer, Institute/Govt Admin, Visitor — each with two columns showing which modules they can access (Listing & Pricing, Pooling, Buyer Discovery, Compliance Dashboard), using a filled circle for full access and a half-filled circle for partial access. To the right of that table, a vertical list of 'Services' with small labeled icons: Maps/Distance API, Agmarknet API, IMD Weather API, SMS/WhatsApp Gateway, Cloud Storage. Use a muted color palette (navy, teal, gray), flat icons, no shadows or gradients, clear thin connecting lines with arrowheads."

---

## Slide 3 — Technical Approach

**Technology Stack (grouped by category, as text on the left):**

**Frontend:** Flutter, Dart, Riverpod (state management), Google Maps SDK

**Backend:** FastAPI (Python), PostgreSQL, Redis (caching), REST APIs

**AI & ML:** Prophet / statsmodels (time-series forecasting), OpenCV (coin-calibrated grain measurement), scikit-learn (rule-scoring weights)

**API Services:** Agmarknet / data.gov.in API, IMD Weather API, Google Maps Distance Matrix API, Twilio (SMS/WhatsApp alerts)

**Deployment:** Docker, GitHub Actions (CI/CD), Render / Railway (hosting for demo)

**Testing & Tools:** Postman, Pytest, GitHub

**Diagram: Detailed Process Flow (right side) — prompt for Draw.io/Figma/AI diagram tool:**
> "Create a detailed process-flow diagram on a white background with 7 rounded rectangle nodes connected by labeled arrows, similar to a user-journey map. Node 1 (top right): 'Farmer Registration & Listing' with a farmer icon. Arrow labeled 'List Produce' down to Node 2: 'Net-Realization & Sell-Window Advisory' with a rupee icon. Arrow labeled 'Join Pool' right to Node 3: 'FPO Pooling Hub' with a group icon. Arrow labeled 'Quality Check' down to Node 4: 'Coin-Calibrated Grading' with a camera icon. Arrow labeled 'Discover Lot' left to Node 5: 'Buyer Matching' with a briefcase icon. Arrow labeled 'Accept Offer' down to Node 6: 'Trade Contract' with a handshake icon. Arrow labeled 'Track Delivery' to Node 7: 'Payment Status Tracker' with a checkmark icon, which loops back with a dashed arrow labeled 'Feedback' to Node 1. Use a consistent icon style, one accent color per node category (green for farmer-side nodes, blue for buyer-side, orange for shared/tracking nodes), clean right-angle or curved connectors, labels on every arrow, no overlapping lines."

---

## Slide 4 — Feasibility and Viability

**Feasibility:**
1. **Technical:** Built entirely on proven, freely available technologies — open government price data, standard web/mobile frameworks, classical (non-training-dependent) computer vision.
2. **Modularization:** Each engine (pricing, pooling, quality check, matching) works independently — a module can fail gracefully without breaking the whole demo.
3. **Market:** Directly addresses a named, funded government priority (farmer income, market linkages) with clear institutional adopters (state agri-marketing boards, NABARD, SFAC).
4. **Economic:** Low initial build cost — core data sources are free and public; no paid API dependency in the MVP path.

**Viability:**
1. Multiple future revenue paths: FPO subscription tiers, buyer-side listing fees, data/analytics services for institutional buyers.
2. Large addressable base: over 10,000 registered FPOs and crores of eNAM-linked farmers nationally.
3. Scalable: architecture generalizes across crops and states without redesign.
4. Long-term value: repository of pooled-lot and price data compounds in usefulness over each season.

**Challenges & Solutions:**

| Challenge | Solution |
|---|---|
| No public buyer-demand dataset | MVP uses a curated FPO/institutional buyer directory; production pursues direct onboarding |
| Live API failure during demo | Local cached Agmarknet/IMD dataset as offline fallback |
| Quality-check reliability in poor lighting | Simplified single-pass measurement with a clear capture instruction, framed explicitly as a decision-support proxy |
| Real payment settlement needs licensing | MVP ships a payment-status tracker; production requires a licensed payment-aggregator partnership |

**★ Supporting Facts for Feasibility and Viability:**
> → A Department of Agriculture & Farmers Welfare review of eNAM found that participating farmers received a **5.5% higher price** selling through electronic markets compared to before adoption, based on primary data collected across states.
>
> → A NABARD study (2020–21 and 2021–22), cited in a reply to Parliament, found FPO-associated farmers saw an **18.75%–31.75% increase in farm productivity** and income growth of **14%–60% in relative terms** post-FPO formation.

---

## Slide 5 — Impact and Benefits

**Diagram: Benefits Tree — prompt for Draw.io/Figma/AI diagram tool:**
> "Create a simple top-down tree diagram on a white background. A single orange rounded rectangle at the top labeled 'Benefits of KrishiDrishti' with a checkmark icon, connecting down via 4 lines to 4 boxes in a row: 'Social' (yellow, people icon), 'Economic' (pink, coin icon), 'Institutional' (blue, building icon), 'Technological' (teal, gear icon). Flat minimal style, evenly spaced, thin connecting lines, no shadows or gradients."

**Benefits of the solution (right column, by category):**

**Social:**
- Strengthens farmer collectivization through FPO participation.
- Reduces distress sales driven by price ignorance, easing financial stress at harvest time.
- Builds trust between farmers and buyers through an objective, transparent quality record.

**Economic:**
- Modeled FPO pooling shows a **+14% institutional price premium** and **65% lower per-quintal freight** versus solo sale — a **₹5,580 net gain on a 15-quintal lot** in our worked example.
- Real-world precedent: NABARD-tracked FPOs already show 14%–60% income growth post-formation; KrishiDrishti's pricing and pooling tools aim to compound this further.
- Reduces post-harvest loss by guiding better sell-timing decisions.

**Institutional:**
- Gives state agri-marketing departments and NABARD/SFAC a data layer they don't currently have — pooled-lot volumes, quality distribution, and buyer-matching outcomes.
- Complements eNAM rather than competing with it — positions naturally as an integration partner, not a replacement.

**Technological:**
- Demonstrates a deterministic, explainable alternative to opaque AI grading — a reusable pattern for other agri-tech quality-check problems.
- Modular architecture allows independent scaling of the pricing engine, matching engine, or quality-check pipeline.

**Potential impact on target audience (quantified, modeled — not yet field-tested):**
- **Improved price realization:** Farmers using Net-Realization comparison across markets could see gains in the range already observed with simpler eNAM price transparency (5.5%), with FPO pooling adding a further modeled 14% premium.
- **Reduced distress sales:** Predictive sell-window advisory could shift a meaningful share of "sell immediately" decisions to a better-timed sale, based on the same economic logic already validated in futures-price-discovery research.
- **Stronger FPO throughput:** A working trade-off simulator could increase voluntary pooling participation by making the ₹ benefit visible and concrete rather than anecdotal.

*(Be explicit in your pitch that these are modeled projections grounded in real eNAM/NABARD data, not yet a field-measured result of KrishiDrishti itself — that honesty is what separates this from an inflated claim.)*

---

## Slide 6 — Research and References

**Existing Platforms (real, verifiable):**
- eNAM (National Agriculture Market): https://enam.gov.in
- DeHaat: https://dehaat.com
- Ninjacart: https://ninjacart.com
- Bijak: https://bijak.in

**Research & Data Sources:**
- Department of Agriculture & Farmers Welfare, *eNAM: A Review of Performance and Prospects* — source of the 5.5% price-realization finding: https://desagri.gov.in
- NABARD FPO productivity study (2020–21, 2021–22), reported via *Business Standard*, Dec 2023: https://www.business-standard.com/industry/agriculture/over-75-of-govt-s-target-to-form-10-000-fpos-reached-in-three-years-123122500613_1.html
- Agmarknet / data.gov.in — open mandi price and arrival-volume dataset: https://agmarknet.gov.in

**Comparison with Existing Systems:**

| Feature | KrishiDrishti | eNAM | DeHaat | Ninjacart |
|---|---|---|---|---|
| Net-Realization pricing (not raw price) | ✅ | ❌ | ❌ | ❌ |
| Predictive sell-window advisory | ✅ | ❌ | ❌ | ❌ |
| FPO pooling + transparent trade-off simulator | ✅ | Partial (manual) | ✅ (village-centre model) | ❌ |
| Explainable, deterministic quality check | ✅ | ❌ (manual assaying) | Partial | ❌ |
| Verified buyer matching | ✅ | ✅ (trader licensing) | ✅ | ✅ |
| Direct farmer-facing app | ✅ | ✅ | ✅ | ❌ (B2B/retail-focused) |
| Free/low-cost for farmers | ✅ | ✅ | Partial | ❌ |

This table is your strongest slide-6 asset — it visually proves you understand the competitive landscape rather than claiming to invent something from nothing.

---

## Notes on using this content

- **Don't skip verifying the two cited statistics yourself** before presenting — pull up the actual desagri.gov.in PDF and the Business Standard article so you can answer confidently if a judge asks "where's that number from?"
- **Keep the "modeled, not field-tested" framing on Slide 5** — it's what makes your impact claims credible rather than inflated, exactly the gap that sinks most hackathon impact slides.
- The competitor table on Slide 6 works only if every ✅ is something you can actually explain in one sentence if asked — rehearse that.
