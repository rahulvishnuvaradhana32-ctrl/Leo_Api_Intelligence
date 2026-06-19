# LEO · Postman collection

Watch LEO's predictive failover live, against the real API.

## Import
1. Open Postman (free) → **Import** → drop in `LEO_API.postman_collection.json`.
2. The collection variable **`base_url`** defaults to the Render deployment
   (`https://leo-api-intelligence.onrender.com`). To hit a local server instead,
   edit `base_url` to `http://localhost:8013` (requires `python scripts/web_server.py`).

## The requests
| Request | What it shows |
|---|---|
| **Health** | service is up (`/health`) |
| **Forecast — Calm** | low signals → `recommended_action: none` (`/v1/forecast`) |
| **Forecast — Outage** | crypto storm → `recommended_action: reroute`, lead time |
| **Reroute demo (live loop)** | the full failover story on `/v1/route` |

## See the reroute live
Open **Reroute demo (live loop)** → **Run** (Collection Runner), or just hit
**Send** repeatedly to step it tick by tick. Each tick feeds `region-a` a rising
then falling risk while `region-b` stays healthy. Open the **Postman Console**
(View → Show Postman Console) to watch:

```
── tick 3  region-a risk = 0.6
   active=region-b  state=REROUTED  action=REROUTE → region-b (least-risk healthy)
   served_by=region-b  record={"idempotency_key":"req-crypto_api-001",...}  checksum=sha256:…
```

The tests assert:
- **data parity** — the `checksum` is identical on every tick, including after
  the reroute → region-b returns the *same record from the same source* (no divergence);
- **✓ rerouted to region-b during the incident**;
- **✓ failed back to region-a after recovery**.

The state machine (`scripts/route_engine.py`) is stateful per `session_id`, so the
loop walks: `serve → pre-warm → REROUTE → serving → recovering (cooldown) → FAIL-BACK`.
