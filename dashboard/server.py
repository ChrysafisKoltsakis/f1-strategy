"""
Simple local dashboard for the live pit-strategy recommender.

Orchestrates the SAME real Kafka/Spark pipeline this project already
has (08_streaming_producer.py + 09_streaming_consumer.py) as subprocesses,
rather than reimplementing the live loop -- so what the browser shows is
a view onto the genuine streaming system, not a simplified stand-in for
it. 09_streaming_consumer.py writes one JSON line per event (lap score,
peak flag, already-acted note, risk comparison) to
data/live_events/{race_id}.jsonl; this server just launches the
producer/consumer and tails that file over Server-Sent Events so a
browser can watch a race replay live.

Deliberately minimal: single race replay at a time, no auth, no
persistence beyond the JSONL file the consumer already writes, plain
CSS. Usability over polish for now.

Requires: the Kafka broker running (see README), JAVA_HOME resolvable
(defaults to ~/.local/jdk if not set), and the trained models already
built (data/models/*.joblib).

Run with: .venv/bin/python dashboard/server.py
Then open: http://localhost:8000
"""
import os
import sys
import json
import time
import socket
import subprocess
import threading
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BRONZE_DIR = PROJECT_ROOT / 'data' / 'bronze'
LIVE_EVENTS_DIR = PROJECT_ROOT / 'data' / 'live_events'
KAFKA_HOST, KAFKA_PORT = 'localhost', 9092
CONSUMER_READY_TIMEOUT_S = 90

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    api_stop()


app = FastAPI(lifespan=lifespan)

# Single-race-at-a-time by design -- this is a local single-user tool, and
# juggling multiple concurrent live replays would add real complexity
# (per-race process tracking, resource contention on the one Spark/Kafka
# setup) for no benefit at this stage.
STATE = {'race_id': None, 'consumer_proc': None, 'producer_proc': None}


def _list_races():
    races = []
    for f in sorted(BRONZE_DIR.glob('*_laps.parquet')):
        race_id = f.stem.replace('_laps', '')
        try:
            event_name = pd.read_parquet(f, columns=['EventName'])['EventName'].iloc[0]
        except Exception:
            event_name = race_id
        races.append({'race_id': race_id, 'event_name': event_name})
    races.sort(key=lambda r: r['race_id'], reverse=True)
    return races


RACES = _list_races()


def _kafka_reachable() -> bool:
    try:
        with socket.create_connection((KAFKA_HOST, KAFKA_PORT), timeout=2):
            return True
    except OSError:
        return False


def _kill(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _consumer_is_ready(events_path: Path) -> bool:
    if not events_path.exists():
        return False
    try:
        with open(events_path) as f:
            return any(json.loads(line).get('type') == 'ready' for line in f if line.strip())
    except (OSError, json.JSONDecodeError):
        return False


def _launch_producer_when_ready(race_id: str, speed_factor: float, consumer_proc, events_path: Path):
    """Runs in a background thread so POST /api/start can return right
    away -- Spark/JVM startup takes 10-20s, and there's no reason to hold
    the HTTP request open for that. The browser's SSE connection just
    shows nothing until the first event lands, which is an honest,
    self-explanatory wait state."""
    deadline = time.time() + CONSUMER_READY_TIMEOUT_S
    while time.time() < deadline:
        if consumer_proc.poll() is not None:
            return  # consumer died before becoming ready
        if _consumer_is_ready(events_path):
            break
        time.sleep(1)
    else:
        return  # timed out
    if STATE.get('race_id') != race_id:
        return  # a stop/restart happened while we were waiting
    producer_proc = subprocess.Popen(
        [sys.executable, '-u', 'notebooks/08_streaming_producer.py', race_id, str(speed_factor)],
        cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    STATE['producer_proc'] = producer_proc


class StartRequest(BaseModel):
    race_id: str
    speed_factor: float = 100


@app.get('/api/races')
def api_races():
    return RACES


@app.get('/api/status')
def api_status():
    running = STATE['consumer_proc'] is not None and STATE['consumer_proc'].poll() is None
    return {'running': running, 'race_id': STATE['race_id'] if running else None}


@app.post('/api/start')
def api_start(body: StartRequest):
    if STATE['consumer_proc'] is not None and STATE['consumer_proc'].poll() is None:
        raise HTTPException(409, 'A replay is already running -- stop it first.')
    if not any(r['race_id'] == body.race_id for r in RACES):
        raise HTTPException(404, f"Unknown race_id '{body.race_id}'.")
    if not _kafka_reachable():
        raise HTTPException(503, f'Kafka broker not reachable at {KAFKA_HOST}:{KAFKA_PORT} -- '
                                  'is the Docker container running? See README.')

    events_path = LIVE_EVENTS_DIR / f"{body.race_id}.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path.exists():
        events_path.unlink()

    env = os.environ.copy()
    java_home = env.get('JAVA_HOME') or str(Path.home() / '.local' / 'jdk')
    env['JAVA_HOME'] = java_home
    env['PATH'] = f"{java_home}/bin:" + env.get('PATH', '')

    consumer_proc = subprocess.Popen(
        [sys.executable, '-u', 'notebooks/09_streaming_consumer.py', body.race_id],
        cwd=PROJECT_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    STATE.update(race_id=body.race_id, consumer_proc=consumer_proc, producer_proc=None)
    threading.Thread(target=_launch_producer_when_ready,
                      args=(body.race_id, body.speed_factor, consumer_proc, events_path),
                      daemon=True).start()
    return {'status': 'starting', 'race_id': body.race_id}


@app.post('/api/stop')
def api_stop():
    _kill(STATE['producer_proc'])
    _kill(STATE['consumer_proc'])
    STATE.update(race_id=None, consumer_proc=None, producer_proc=None)
    return {'status': 'stopped'}


@app.get('/api/stream/{race_id}')
async def api_stream(race_id: str, request: Request):
    import asyncio
    events_path = LIVE_EVENTS_DIR / f"{race_id}.jsonl"

    async def event_gen():
        while not events_path.exists():
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.3)
        with open(events_path) as f:
            while True:
                if await request.is_disconnected():
                    return
                line = f.readline()
                if line:
                    yield f"data: {line}\n\n"
                else:
                    await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type='text/event-stream')


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>F1 Live Pit Strategy</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #111; }
  h1 { font-size: 1.3rem; margin-bottom: 0.2rem; }
  .subtitle { color: #555; margin-bottom: 1rem; max-width: 62rem; line-height: 1.4; }
  .controls { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; }
  select, input[type=number], button { font-size: 0.95rem; padding: 0.35rem 0.5rem; }
  button { cursor: pointer; }
  #status { font-weight: 600; margin-left: 0.4rem; }
  #context { background: #f2f2f2; border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 1rem; display: none; font-size: 0.9rem; }
  .columns { display: flex; gap: 1.5rem; align-items: flex-start; flex-wrap: wrap; }
  .panel { flex: 1 1 420px; min-width: 380px; }
  .panel h2 { font-size: 1rem; margin: 0 0 0.4rem; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border: 1px solid #ddd; padding: 0.3rem 0.5rem; text-align: left; }
  th { background: #fafafa; }
  #feed { max-height: 70vh; overflow-y: auto; display: flex; flex-direction: column; gap: 0.4rem; }
  .card { border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem 0.7rem; font-size: 0.85rem; }
  .card .head { font-weight: 600; }
  .card.pending { border-style: dashed; color: #777; }
  .card.pit-now { background: #ffe9e0; border-color: #e07a4f; }
  .card.wait { background: #e6f0ff; border-color: #5b8def; }
  .card.acted { background: #f0f0f0; color: #666; }
  .hint { color: #888; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>F1 Live Pit Strategy -- Replay Viewer</h1>
<p class="subtitle">
  Replays a real historical race through Kafka in real chronological order, at accelerated speed.
  A trained classifier is queried lap-by-lap as events arrive -- exactly as a live system would use
  it, with no access to laps that haven't "happened" yet. When its pit probability for a driver peaks
  and then drops, that's flagged one lap late (a live system can only tell a value was a peak once it
  sees the next one come in lower) and re-evaluated against the actual cost of pitting now versus
  waiting for a possible Safety Car window later.
</p>
<div class="controls">
  <select id="race"></select>
  speed
  <input id="speed" type="number" value="100" min="5" max="500" style="width:5rem">
  <button id="start">Start replay</button>
  <button id="stop">Stop</button>
  <span id="status">idle</span>
</div>
<div id="context"></div>
<div class="columns">
  <div class="panel">
    <h2>Current state (per driver)</h2>
    <p class="hint">Updates as each driver completes a lap. Sorted alphabetically so rows don't jump around.</p>
    <table id="driverTable"><thead><tr><th>Driver</th><th>Team</th><th>Lap</th><th>P(pit)</th><th>Tyre</th></tr></thead><tbody></tbody></table>
  </div>
  <div class="panel">
    <h2>Live recommendations</h2>
    <p class="hint">Fires one lap after a driver's pit-probability peak. Dashed = just flagged, waiting on the cost comparison.</p>
    <div id="feed"></div>
  </div>
</div>
<script>
let es = null;
const drivers = {};   // driver -> {team, lap, proba, compound, tyre_age}
const cards = {};     // "driver:lap" -> DOM node

function probColor(p) {
  const r = Math.round(255);
  const g = Math.round(255 - p * 170);
  const b = Math.round(255 - p * 200);
  return `rgb(${r},${g},${b})`;
}

function renderDriverTable() {
  const tbody = document.querySelector('#driverTable tbody');
  tbody.innerHTML = '';
  Object.keys(drivers).sort().forEach(d => {
    const row = drivers[d];
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${d}</td><td>${row.team||''}</td><td>${row.lap ?? ''}</td>` +
      `<td style="background:${probColor(row.proba||0)}">${(row.proba??0).toFixed(2)}</td>` +
      `<td>${row.compound||''} (${row.tyre_age ?? '?'})</td>`;
    tbody.appendChild(tr);
  });
}

function addPeakCard(evt) {
  const key = evt.driver + ':' + evt.lap;
  const div = document.createElement('div');
  div.className = 'card pending';
  div.innerHTML = `<div class="head">${evt.driver} -- lap ${evt.lap}</div>` +
    `peak was lap ${evt.peak_lap} (P=${evt.peak_proba.toFixed(2)}) -- evaluating...`;
  document.getElementById('feed').prepend(div);
  cards[key] = div;
}

function resolveCard(evt, kind) {
  const key = evt.driver + ':' + evt.lap;
  const div = cards[key];
  if (!div) return;
  div.classList.remove('pending');
  if (kind === 'already_acted') {
    div.classList.add('acted');
    div.innerHTML += `<br>already pitted at the flagged lap -- no live decision left to evaluate.`;
  } else {
    div.classList.add(evt.verdict === 'pit NOW' ? 'pit-now' : 'wait');
    const discountNote = evt.sc_active_now ? 'SC/VSC active -- discounted' : 'green flag -- full price';
    div.innerHTML += `<br>pit now (${discountNote}): <b>${evt.pit_now_s.toFixed(1)}s</b> &nbsp;|&nbsp; ` +
      `wait (worst): ${evt.wait_worst_s.toFixed(1)}s &nbsp;|&nbsp; ` +
      `wait (SC again, P=${Math.round(evt.p_again*100)}%): ${evt.wait_best_s.toFixed(1)}s &nbsp;|&nbsp; ` +
      `expected wait: ${evt.expected_wait_s.toFixed(1)}s` +
      `<br><b>${evt.verdict}</b> looks better by ${Math.abs(evt.diff_s).toFixed(1)}s on expectation`;
  }
}

function onEvent(evt) {
  if (evt.type === 'context') {
    const c = document.getElementById('context');
    c.style.display = 'block';
    c.innerHTML = `<b>${evt.event_name}</b> (${evt.race_id}) -- ${evt.n_laps} laps. ` +
      (evt.compounds_modeled && evt.compounds_modeled.length
        ? `Compounds modeled: ${evt.compounds_modeled.join(', ')}. Green pit loss ${evt.green_pit_loss.toFixed(1)}s, ` +
          `SC/VSC pit loss ${evt.sc_pit_loss.toFixed(1)}s.`
        : `No usable pit-loss estimate for this race -- recommendations will show probability only, no cost comparison.`);
  } else if (evt.type === 'ready') {
    document.getElementById('status').textContent = 'live -- waiting for laps';
  } else if (evt.type === 'lap_score') {
    drivers[evt.driver] = {team: evt.team, lap: evt.lap, proba: evt.proba, compound: evt.compound, tyre_age: evt.tyre_age};
    renderDriverTable();
  } else if (evt.type === 'peak_flag') {
    addPeakCard(evt);
  } else if (evt.type === 'already_acted') {
    resolveCard(evt, 'already_acted');
  } else if (evt.type === 'risk_comparison') {
    resolveCard(evt, 'risk_comparison');
  }
}

async function loadRaces() {
  const races = await (await fetch('/api/races')).json();
  const sel = document.getElementById('race');
  sel.innerHTML = races.map(r => `<option value="${r.race_id}">${r.event_name} (${r.race_id})</option>`).join('');
}

function connectStream(raceId) {
  if (es) es.close();
  es = new EventSource('/api/stream/' + raceId);
  es.onmessage = (m) => onEvent(JSON.parse(m.data));
}

document.getElementById('start').onclick = async () => {
  const race_id = document.getElementById('race').value;
  const speed_factor = parseFloat(document.getElementById('speed').value) || 100;
  document.getElementById('status').textContent = 'starting (Spark boot takes ~15-20s)...';
  Object.keys(drivers).forEach(k => delete drivers[k]);
  Object.keys(cards).forEach(k => delete cards[k]);
  document.getElementById('feed').innerHTML = '';
  document.querySelector('#driverTable tbody').innerHTML = '';
  document.getElementById('context').style.display = 'none';
  const res = await fetch('/api/start', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({race_id, speed_factor}),
  });
  if (!res.ok) {
    const err = await res.json();
    document.getElementById('status').textContent = 'error: ' + (err.detail || res.statusText);
    return;
  }
  connectStream(race_id);
};

document.getElementById('stop').onclick = async () => {
  await fetch('/api/stop', {method: 'POST'});
  if (es) { es.close(); es = null; }
  document.getElementById('status').textContent = 'stopped';
};

loadRaces();
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
