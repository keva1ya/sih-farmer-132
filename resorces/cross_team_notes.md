# KrishiDrishti — Cross-Team Reference Notes

This file contains information discovered during CV module development that is relevant to other team members.

---

## For ML / Backend Team: CEDA API (Mandi Price Data)

**CEDA (Centre for Economic Data and Analysis)** at Ashoka University provides a cleaned, developer-friendly API for Agmarknet data.

| Detail | Value |
|---|---|
| **Portal** | https://agmarknet.ceda.ashoka.edu.in/ |
| **API Docs** | https://api.ceda.ashoka.edu.in/ |
| **Spec** | OpenAPI (OAS 3.0) |
| **Auth** | API key (free registration) |
| **Coverage** | 300+ agricultural commodities, 2700+ mandis across India |
| **Data** | Daily modal prices, arrivals (quantity), market names, commodity names |
| **Use Case** | Powers the **Net Realization calculation** — rank mandis by net payout after freight/fees |

### What CEDA does NOT provide:
- ❌ Quality grading standards or thresholds (no AGMARK grade data)
- ❌ Weather/rainfall data (use Open-Meteo or IMD for that)
- ❌ Warehouse capacity (use WDRA for that)

### How to integrate:
1. Register at https://agmarknet.ceda.ashoka.edu.in/ to get an API key
2. Use the API to pull live mandi prices for the commodity (e.g., Wheat)
3. Feed into the Net Realization formula: `Net_m = P_m − ((Distance_m × C_km + C_fixed) / V) − (P_m × (R_cess + R_comm))`
4. This replaces the need for hardcoded seed price data in `mandi_price_history`

> **Note:** Agmarknet's own portal has restricted B2B access. CEDA is the easiest legal way to get this data programmatically.

---

## For All Teams: AGMARK Grading Standards — No API Exists

We searched every government data source. **There is no public API for AGMARK grading thresholds.** Standards are published as Gazette of India PDF notifications by the Directorate of Marketing & Inspection (DMI).

| Source Checked | Result |
|---|---|
| DMI (dmi.gov.in) | PDF notifications only |
| FSSAI | Safety minimums, not quality grades |
| FCI | Seasonal procurement circulars (PDFs) |
| API Setu (apisetu.gov.in) | No grading endpoints |
| APEDA HortiNet | Export-only certification |
| data.gov.in | No grading datasets |

**Our approach (CV module):** We hardcoded FCI Uniform Specifications for Wheat in `cv/config.py`. Adding new commodities = adding a new dictionary entry, no code changes required.

**For the pitch:** Frame it as: *"We use FCI Uniform Specifications as our grading baseline — the same standards used for government MSP procurement. In production, we would ingest the official DMI Gazette notifications as they're published."*

---

## For Backend Team: CV Module API Contract

The CV service exposes `POST /analyze` (see `cv/` folder for full implementation).

**Request:** `multipart/form-data` with an image file (JPEG/PNG, max 10MB)

**Response (default):**
```json
{
  "status": "success",
  "grain_count": 47,
  "measurements": { "avg_length_mm": 6.92, "broken_pct": 1.8, ... },
  "grade": "A",
  "hash": "sha256..."
}
```

**Response (with `?detailed=true`):** Includes `per_metric_grades` breakdown.

**Error statuses:** `too_blurry`, `too_tilted`, `coin_not_found`, `coin_not_recognized`, `too_few_grains`, `invalid_format`, `file_too_large`

---

## For Backend Team: Video Upload Endpoint

The CV service will also include `POST /upload-video` for buyer visual records.
- Accepts MP4/MOV, max 30 seconds / 50MB
- No CV processing — just stores the blob and returns a URL
- Link the returned URL to the `COMMODITY_LISTINGS` record
