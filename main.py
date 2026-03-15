"""
XAUUSD Order Flow Analysis Backend
FastAPI + Dukascopy data source
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import struct
import lzma
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional
import math

app = FastAPI(title="Order Flow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Dukascopy helpers ──────────────────────────────────────────────────────

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

INSTRUMENT_MAP = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
}

async def fetch_dukascopy_candles(symbol: str, year: int, month: int, day: int, hour: int) -> list[dict]:
    """
    Fetch 1-minute OHLCV candles from Dukascopy for a specific hour.
    Returns list of {time, open, high, low, close, volume}
    """
    # Dukascopy uses 0-based months
    url = f"{DUKA_BASE}/{symbol}/{year:04d}/{(month-1):02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []

    raw = lzma.decompress(resp.content)

    ticks = []
    tick_size = 20  # Dukascopy tick: 4+4+4+4+4 bytes = 20 bytes
    for i in range(0, len(raw) - tick_size + 1, tick_size):
        chunk = raw[i:i + tick_size]
        ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)
        ts = datetime(year, month, day, hour, tzinfo=timezone.utc) + timedelta(milliseconds=ms_offset)
        # For gold: price divisor is 100000 for most Dukascopy pairs; XAUUSD uses 1000
        divisor = 1000.0 if symbol == "XAUUSD" else 100000.0
        mid = (ask_raw + bid_raw) / 2 / divisor
        ticks.append({
            "ts": ts,
            "price": mid,
            "ask_vol": float(ask_vol),
            "bid_vol": float(bid_vol),
        })

    return ticks


def ticks_to_candles(ticks: list[dict], tf_seconds: int = 60) -> list[dict]:
    """Aggregate ticks into OHLCV candles."""
    if not ticks:
        return []

    candles: dict[int, dict] = {}
    for t in ticks:
        bucket = int(t["ts"].timestamp() // tf_seconds) * tf_seconds
        if bucket not in candles:
            candles[bucket] = {
                "time": bucket,
                "open": t["price"],
                "high": t["price"],
                "low": t["price"],
                "close": t["price"],
                "volume": 0.0,
                "buy_vol": 0.0,
                "sell_vol": 0.0,
            }
        c = candles[bucket]
        c["high"] = max(c["high"], t["price"])
        c["low"] = min(c["low"], t["price"])
        c["close"] = t["price"]
        c["volume"] += t["ask_vol"] + t["bid_vol"]
        c["buy_vol"] += t["ask_vol"]
        c["sell_vol"] += t["bid_vol"]

    return sorted(candles.values(), key=lambda x: x["time"])


# ─── Order flow calculations ────────────────────────────────────────────────

def compute_volume_delta(candles: list[dict]) -> list[dict]:
    """
    Compute volume delta (buy pressure - sell pressure) per candle.
    Also compute cumulative delta.
    """
    cum_delta = 0.0
    result = []
    for c in candles:
        delta = c["buy_vol"] - c["sell_vol"]
        cum_delta += delta
        result.append({
            **c,
            "delta": round(delta, 4),
            "cum_delta": round(cum_delta, 4),
        })
    return result


def compute_volume_profile(candles: list[dict], pip_size: float = 0.5) -> list[dict]:
    """
    Build a volume profile: volume traded at each price level.
    pip_size = bucket width in price units (0.5 for gold = 50 cents).
    """
    profile: dict[float, float] = {}
    for c in candles:
        # Distribute candle volume across price range proportionally
        lo_bucket = math.floor(c["low"] / pip_size) * pip_size
        hi_bucket = math.floor(c["high"] / pip_size) * pip_size
        price = lo_bucket
        num_buckets = max(1, round((hi_bucket - lo_bucket) / pip_size) + 1)
        vol_per_bucket = c["volume"] / num_buckets
        while price <= hi_bucket + 0.001:
            key = round(price, 2)
            profile[key] = profile.get(key, 0) + vol_per_bucket
            price = round(price + pip_size, 2)

    if not profile:
        return []

    max_vol = max(profile.values())
    result = [
        {
            "price": p,
            "volume": round(v, 4),
            "pct": round(v / max_vol * 100, 1),
        }
        for p, v in sorted(profile.items())
    ]
    return result


def detect_volume_spikes(candles: list[dict], window: int = 20, multiplier: float = 2.0) -> list[dict]:
    """
    Flag candles where volume > multiplier × rolling average.
    Returns enriched candles with spike flag and avg_vol reference.
    """
    result = []
    for i, c in enumerate(candles):
        start = max(0, i - window)
        window_vols = [x["volume"] for x in candles[start:i]]
        avg_vol = sum(window_vols) / len(window_vols) if window_vols else 0
        is_spike = avg_vol > 0 and c["volume"] > multiplier * avg_vol
        result.append({
            **c,
            "avg_vol": round(avg_vol, 4),
            "is_spike": is_spike,
            "spike_ratio": round(c["volume"] / avg_vol, 2) if avg_vol > 0 else 0,
        })
    return result


def find_poc_and_value_area(profile: list[dict], value_area_pct: float = 0.70) -> dict:
    """
    Find Point of Control (POC) and Value Area (70% of total volume).
    """
    if not profile:
        return {}

    total_vol = sum(p["volume"] for p in profile)
    poc = max(profile, key=lambda x: x["volume"])

    # Value area: expand from POC until 70% of total volume is covered
    sorted_by_vol = sorted(profile, key=lambda x: x["volume"], reverse=True)
    covered = 0.0
    va_prices = []
    for level in sorted_by_vol:
        covered += level["volume"]
        va_prices.append(level["price"])
        if covered >= total_vol * value_area_pct:
            break

    return {
        "poc": poc["price"],
        "poc_volume": round(poc["volume"], 4),
        "vah": round(max(va_prices), 2),  # Value Area High
        "val": round(min(va_prices), 2),  # Value Area Low
        "total_volume": round(total_vol, 4),
    }


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/orderflow/{symbol}")
async def get_order_flow(
    symbol: str,
    hours_back: int = Query(default=3, ge=1, le=12),
    timeframe: int = Query(default=60, description="Candle size in seconds"),
    spike_window: int = Query(default=20),
    spike_multiplier: float = Query(default=2.0),
    profile_pip_size: float = Query(default=0.5),
):
    """
    Main endpoint: returns volume delta, volume profile, and spike detection
    for the requested symbol over the last N hours.
    """
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        return {"error": f"Unknown symbol. Supported: {list(INSTRUMENT_MAP.keys())}"}

    now = datetime.now(timezone.utc)
    all_ticks: list[dict] = []

    # Fetch each hour in parallel
    hours = []
    for h in range(hours_back):
        target = now - timedelta(hours=h + 1)
        hours.append((target.year, target.month, target.day, target.hour))

    async def fetch_hour(args):
        y, mo, d, hr = args
        try:
            return await fetch_dukascopy_candles(symbol, y, mo, d, hr)
        except Exception:
            return []

    results = await asyncio.gather(*[fetch_hour(h) for h in hours])
    for r in results:
        all_ticks.extend(r)

    all_ticks.sort(key=lambda x: x["ts"])

    if not all_ticks:
        return {"error": "No data returned from Dukascopy. Try a different time range or check if markets are open."}

    # Build candles
    candles = ticks_to_candles(all_ticks, tf_seconds=timeframe)

    # Compute order flow metrics
    candles_with_delta = compute_volume_delta(candles)
    candles_with_spikes = detect_volume_spikes(candles_with_delta, window=spike_window, multiplier=spike_multiplier)

    # Volume profile
    profile = compute_volume_profile(candles, pip_size=profile_pip_size)
    poc_info = find_poc_and_value_area(profile)

    # Summary stats
    spikes = [c for c in candles_with_spikes if c["is_spike"]]
    deltas = [c["delta"] for c in candles_with_delta]
    net_delta = sum(deltas)
    bias = "bullish" if net_delta > 0 else "bearish"

    return {
        "symbol": symbol,
        "generated_at": now.isoformat(),
        "candle_count": len(candles),
        "timeframe_seconds": timeframe,
        "summary": {
            "net_delta": round(net_delta, 4),
            "bias": bias,
            "spike_count": len(spikes),
            "last_price": candles[-1]["close"] if candles else None,
            **poc_info,
        },
        "candles": candles_with_spikes,
        "volume_profile": profile,
    }


@app.get("/api/symbols")
async def list_symbols():
    return {"symbols": list(INSTRUMENT_MAP.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
