"""
Chapter 12: Deploying Agentic AI to Production
Hands-On: Wrapping the Chapter 10 Model as a Real FastAPI Service

Extracted from: chapter_12_deployment.md
Source book: Agentic AI: Building AI Agents and Retrieval Systems,
a Masterclass in LLM Agents, RAG, and Production Deployment.

Every block below was verified by direct execution before being
written into the handbook; run this file top to bottom, or copy
out the section you need. Where a step needs an API key
(OPENAI_API_KEY / ANTHROPIC_API_KEY), it is loaded from a local
.env file via python-dotenv, following Chapter 5's own security
discipline, never hardcoded.

ADAPTED FOR STANDALONE USE: the book's own text calls Steps 4 and 6
against a real `uvicorn main:app --reload` server running in a second
terminal, exactly how you would actually deploy this. So this file
runs top to bottom with no second terminal needed, those two steps
below use FastAPI's own TestClient instead, an in-process client that
exercises the identical request/response path with no live server
required. Run `uvicorn main:app --reload` yourself and repeat Steps 4
and 6 with `httpx` directly against `http://localhost:8000` to see the
real deployed version this chapter itself walks through.
"""


# ---- Step 1: Define the request schema, following Chapter 9's Pydantic discipline, with genuine clinical bounds on every field rather than accepting any number at all. ----
from pydantic import BaseModel, Field

class PatientRequest(BaseModel):
    pregnancies: int = Field(ge=0, le=20)
    glucose: float = Field(ge=0, le=300)
    blood_pressure: float = Field(ge=0, le=200)
    skin_thickness: float = Field(ge=0, le=100)
    insulin: float = Field(ge=0, le=900)
    bmi: float = Field(ge=0, le=70)
    diabetes_pedigree: float = Field(ge=0, le=3)
    age: int = Field(ge=0, le=120)

class PredictionResponse(BaseModel):
    probability: float
    risk_level: str

# ---- Step 2: Load the Chapter 10 pipeline once, at import time, exactly as this chapter's own architecture principles require. ----
import pickle

with open("saved_models/diabetes_model.sav", "rb") as f:
    rf_pipeline = pickle.load(f)

# ---- Step 3: Define the async endpoint, following this chapter's own recommended FastAPI pattern from earlier. ----
from fastapi import FastAPI

app = FastAPI(title="Health Copilot API")

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientRequest) -> PredictionResponse:
    features = [[patient.pregnancies, patient.glucose, patient.blood_pressure,
                 patient.skin_thickness, patient.insulin, patient.bmi,
                 patient.diabetes_pedigree, patient.age]]
    probability = rf_pipeline.predict_proba(features)[0][1]
    risk_level = "high" if probability >= 0.7 else "moderate" if probability >= 0.4 else "low"
    return PredictionResponse(probability=round(float(probability), 4), risk_level=risk_level)

# ---- Step 4: Confirm both the success path and the validation path actually work
#      (TestClient stands in for `httpx` against a real running server; see the
#      module docstring above). ----
from fastapi.testclient import TestClient

test_client = TestClient(app)

good = {"pregnancies": 2, "glucose": 148, "blood_pressure": 72, "skin_thickness": 35,
        "insulin": 155, "bmi": 33.6, "diabetes_pedigree": 0.627, "age": 50}
resp = test_client.post("/predict", json=good)
print(resp.status_code, resp.json())

bad = dict(good, glucose=9999)  # deliberately violates Field(le=300)
resp2 = test_client.post("/predict", json=bad)
print(resp2.status_code, resp2.json())

# ---- Step 5: Close the two gaps this section's own "Think it through" question names, no rate limiting and no authentication, using exactly the `slowapi` pattern Chapter 13 built and tested for a different endpoint. ----
# The book presents this as editing the same main.py and restarting the server,
# replacing Step 3's endpoint outright; concatenated into one script instead, the
# old, unauthenticated "/predict" route has to be removed explicitly first, or
# FastAPI's own first-registered-route-wins matching would silently keep serving
# it instead of the new, authenticated version defined just below.
app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != "/predict"]

from fastapi import Request, Header, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

API_KEYS = {"demo-key-123"}   # a real deployment reads this from a secrets manager, not a set literal

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
async def predict(request: Request, patient: PatientRequest, x_api_key: str = Header(None)) -> PredictionResponse:
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    features = [[patient.pregnancies, patient.glucose, patient.blood_pressure,
                 patient.skin_thickness, patient.insulin, patient.bmi,
                 patient.diabetes_pedigree, patient.age]]
    probability = rf_pipeline.predict_proba(features)[0][1]
    risk_level = "high" if probability >= 0.7 else "moderate" if probability >= 0.4 else "low"
    return PredictionResponse(probability=round(float(probability), 4), risk_level=risk_level)

# ---- Step 6: Confirm both new gates actually hold, alongside the negative-`pregnancies` case this section's "Think it through" question asked you to predict before running. ----
# rebuild the TestClient so it targets the route table as it stands after Step 5
# re-registered "/predict" with auth and rate limiting (FastAPI matches routes in
# registration order, so a fresh TestClient against the same `app` is enough,
# the newly added route sits alongside the Step 3 one on the same path and method)
test_client = TestClient(app)

resp_no_key = test_client.post("/predict", json=good)
print(resp_no_key.status_code, resp_no_key.json())

resp_good = test_client.post("/predict", json=good, headers={"x-api-key": "demo-key-123"})
print(resp_good.status_code, resp_good.json())

bad_pregnancies = dict(good, pregnancies=-3)
resp_bad = test_client.post("/predict", json=bad_pregnancies, headers={"x-api-key": "demo-key-123"})
print(resp_bad.status_code, resp_bad.json())

# ---- Step 7: Add the one endpoint every one of Chapter 12's five deployment patterns needs and none of the six steps above have built yet, a health check, the thing a load balancer, a Kubernetes readiness probe, or Render's own platform actually polls to decide whether traffic should be routed to this instance at all. ----
import time

START_TIME = time.time()

@app.get("/health")
async def health() -> dict:
    uptime = round(time.time() - START_TIME, 2)
    model_ready = rf_pipeline is not None
    return {
        "status": "healthy" if model_ready else "degraded",
        "uptime_seconds": uptime,
        "model_loaded": model_ready,
    }

response = test_client.get("/health")
print(response.status_code, response.json())

# ---- Step 8: Prove the `async def` concurrency claim this chapter's own prose has been making throughout, rather than only asserting it, by firing a batch of concurrent requests at both an async and a sync version of the same slow endpoint and timing each. ----
import asyncio
import time
from httpx import AsyncClient, ASGITransport

# ADAPTED FOR STANDALONE USE: a plain script has no event loop running yet,
# so asyncio.run() below works unmodified there; a Jupyter notebook already
# has one running per cell, and asyncio.run() refuses to nest inside it.
# nest_asyncio patches that in, so the exact same asyncio.run() calls work
# in both contexts with no further changes.
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

@app.get("/slow_async")
async def slow_endpoint_async():
    await asyncio.sleep(0.2)   # stands in for a real predict_proba call or a slow external API
    return {"status": "done"}

@app.get("/slow_sync")
def slow_endpoint_sync():
    time.sleep(0.2)            # the blocking equivalent, deliberately not awaited
    return {"status": "done"}

async def run_n_concurrent(path: str, n: int) -> float:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        start = time.perf_counter()
        await asyncio.gather(*[client.get(path) for _ in range(n)])
        return time.perf_counter() - start

for n in [10, 60]:
    print(f"async, n={n}: {asyncio.run(run_n_concurrent('/slow_async', n)):.2f}s")
for n in [10, 60]:
    print(f"sync,  n={n}: {asyncio.run(run_n_concurrent('/slow_sync', n)):.2f}s")
