#!/usr/bin/env python3
import json, logging
from pathlib import Path

log = logging.getLogger("positions")
STATE_FILE = Path("/home/msbel/.openclaw/workspace/trading/logs/positions.json")

def save_positions(trackers: dict):
    state = {}
    for sym, t in trackers.items():
        if t.position > 0:
            state[sym] = {
                "position": t.position,
                "entry_price": t.entry_price,
            }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))

def load_positions() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception as e:
        log.warning(f"Failed to load positions: {e}")
        return {}
