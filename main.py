from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import struct
import lzma
import asyncio
from datetime import datetime, timezone, timedelta
import math

app = FastAPI(title="Order Flow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

INSTRUMENT_MAP = {
    "XAUUSD": "XAUUSD",
    "EURUSD": "EURUSD",
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
}


def get_last_trading_hours(hours_back):
    """Return trading hours skipping weekends, going back as far as needed."""
    now = datetime.now(timezone.utc)
    hours = []
    offset = 1
    while len(hours) < hours_back and offset < 300:
        target = now - timedelta(hours=offset)
        if target.weekday() < 5:  # Mon=0 ... Fri=4, skip Sat=5 Sun=6
            hours.append((target.year, target.month, target.day, target.hour))
        offset += 1
    return hours


async def fetch_dukascopy_candles(symbol, year, month, day, hour):
    url = f"{DUKA_BASE}/{symbol}/{year:04d}/{(month-1):02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
    raw = lzma.decompress(resp.content)
    ticks = []
    tick_size = 20
    for i in range(0, len(raw) - tick_size + 1, tick_size):
        chunk = raw[i:i + tick_size]
        ms_offset, ask_raw, bid_raw, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)
        ts = datetime(year, month, day, hour, tzinfo=timezone.utc) + timedelta(milliseconds=ms_offset)
        divisor = 1000.0 if symbol == "XAUUSD" else 100000.0
        mid = (ask_raw + bid_raw) / 2 / divisor
        ticks.append({
            "ts": ts,
            "price": mid,
            "ask_vol": float(ask_vol),
            "bid_vol": float(bid_vol),
        })
    return ticks


def ticks_to_candles(ticks, tf_seconds=60):
    if not ticks:
        return []
    candles = {}
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


def compute_volume_delta(candles):
    cum_delta = 0.0
    result = []
    for c in candles:
        delta = c["buy_vol"] - c["sell_vol"]
        cum_delta += delta
        result.append({**c, "delta": round(delta, 4), "cum_delta": round(cum_delta, 4)})
    return result


def compute_volume_profile(candles, pip_size=0.5):
    profile = {}
    for c in candles:
        lo_bucket = math.floor(c["low"] / pip_size) * pip_size
        hi_bucket = math.floor(c["high"] / pip_size) * pip_size
        num_buckets = max(1, round((hi_bucket - lo_bucket) / pip_size) + 1)
        vol_per_bucket = c["volume"] / num_buckets
        price = lo_bucket
        while price <= hi_bucket + 0.001:
            key = round(price, 2)
            profile[key] = profile.get(key, 0) + vol_per_bucket
            price = round(price + pip_size, 2)
    if not profile:
        return []
    max_vol = max(profile.values())
    return [
        {"price": p, "volume": round(v, 4), "pct": round(v / max_vol * 100, 1)}
        for p, v in sorted(profile.items())
    ]


def detect_volume_spikes(candles, window=20, multiplier=2.0):
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


def find_poc_and_value_area(profile, value_area_pct=0.70):
    if not profile:
        return {}
    total_vol = sum(p["volume"] for p in profile)
    poc = max(profile, key=lambda x: x["volume"])
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
        "vah": round(max(va_prices), 2),
        "val": round(min(va_prices), 2),
        "total_volume": round(total_vol, 4),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/orderflow/{symbol}")
async def get_order_flow(
    symbol: str,
    hours_back: int = Query(default=3, ge=1, le=12),
    timeframe: int = Query(default=60),
    spike_window: int = Query(default=20),
    spike_multiplier: float = Query(default=2.0),
    profile_pip_size: float = Query(default=0.5),
):
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        return {"error": f"Unknown symbol. Supported: {list(INSTRUMENT_MAP.keys())}"}

    hours = get_last_trading_hours(hours_back)
    all_ticks = []

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
        return {"error": "No data found. Dukascopy may not have data for this period yet."}

    now = datetime.now(timezone.utc)
    candles = ticks_to_candles(all_ticks, tf_seconds=timeframe)
    candles_with_delta = compute_volume_delta(candles)
    candles_with_spikes = detect_volume_spikes(candles_with_delta, window=spike_window, multiplier=spike_multiplier)
    profile = compute_volume_profile(candles, pip_size=profile_pip_size)
    poc_info = find_poc_and_value_area(profile)

    spikes = [c for c in candles_with_spikes if c["is_spike"]]
    net_delta = sum(c["delta"] for c in candles_with_delta)

    return {
        "symbol": symbol,
        "generated_at": now.isoformat(),
        "candle_count": len(candles),
        "timeframe_seconds": timeframe,
        "data_from": "last trading session (weekend-aware)",
        "summary": {
            "net_delta": round(net_delta, 4),
            "bias": "bullish" if net_delta > 0 else "bearish",
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
