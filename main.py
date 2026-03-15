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
    "XAUUSD":    "XAUUSD",
    "XAGUSD":    "XAGUSD",
    "USOUSD":    "USOUSD",
    "NATGASUSD": "NATGASUSD",
    "EURUSD":    "EURUSD",
    "GBPUSD":    "GBPUSD",
    "USDJPY":    "USDJPY",
    "BTCUSD":    "BTCUSD",
    "SPXUSD":    "SPXUSD",
}

DIVISORS = {
    "XAUUSD":    1000.0,
    "XAGUSD":    1000.0,
    "USOUSD":    1000.0,
    "NATGASUSD": 10000.0,
    "EURUSD":    100000.0,
    "GBPUSD":    100000.0,
    "USDJPY":    1000.0,
    "BTCUSD":    1.0,
    "SPXUSD":    1000.0,
}

# ─── Trading sessions (UTC hours) ──────────────────────────────────────────
SESSIONS = [
    # name, color_key, start_hour_utc, end_hour_utc
    ("Asia",    "asia",    0,  8),
    ("London",  "london",  7, 16),
    ("NY",      "ny",     13, 21),
    ("Off",     "off",    21, 24),
]

def get_session(epoch: int) -> str:
    """Return session name for a UTC epoch timestamp."""
    hour = datetime.utcfromtimestamp(epoch).hour
    # NY and London overlap 13-16 → mark as overlap
    if 13 <= hour < 16:
        return "overlap"
    elif 7 <= hour < 13:
        return "london"
    elif 13 <= hour < 21:
        return "ny"
    elif 0 <= hour < 7:
        return "asia"
    else:
        return "off"

# ─── Delta divergence detection ────────────────────────────────────────────
def detect_delta_divergence(candles: list, lookback: int = 5) -> list:
    """
    Detect bearish and bullish delta divergences.

    Bearish divergence: price makes higher high BUT cumulative delta makes lower high
      → buyers are exhausted, potential reversal down
    Bullish divergence: price makes lower low BUT cumulative delta makes higher low
      → sellers are exhausted, potential reversal up

    lookback: number of candles to look back for swing comparison
    """
    result = []
    for i, c in enumerate(candles):
        divergence = None
        strength   = 0.0

        if i >= lookback:
            window = candles[i - lookback: i + 1]
            prices  = [x["close"] for x in window]
            deltas  = [x["cum_delta"] for x in window]

            prev_high_price = max(prices[:-1])
            prev_high_delta = max(deltas[:-1])
            prev_low_price  = min(prices[:-1])
            prev_low_delta  = min(deltas[:-1])

            cur_price = c["close"]
            cur_delta = c["cum_delta"]

            # Bearish: new price high but delta NOT confirming
            if cur_price > prev_high_price and cur_delta < prev_high_delta:
                price_gain  = (cur_price - prev_high_price) / (prev_high_price + 1e-9)
                delta_drop  = (prev_high_delta - cur_delta) / (abs(prev_high_delta) + 1e-9)
                strength    = round(min(100, (price_gain + delta_drop) * 50), 1)
                divergence  = "bearish"

            # Bullish: new price low but delta NOT confirming
            elif cur_price < prev_low_price and cur_delta > prev_low_delta:
                price_drop  = (prev_low_price - cur_price) / (prev_low_price + 1e-9)
                delta_rise  = (cur_delta - prev_low_delta) / (abs(prev_low_delta) + 1e-9)
                strength    = round(min(100, (price_drop + delta_rise) * 50), 1)
                divergence  = "bullish"

        result.append({
            **c,
            "divergence":          divergence,
            "divergence_strength": strength,
        })
    return result


# ─── Absorption detection ──────────────────────────────────────────────────
def detect_absorption(candles: list, vol_window: int = 20, vol_mult: float = 1.8,
                       range_threshold: float = 0.35) -> list:
    """
    Absorption: high volume candle BUT small price range.
    Signals institutions absorbing the flow — potential reversal.
    vol_mult: volume must be this × avg to qualify
    range_threshold: candle range must be < this fraction of avg range
    """
    result = []
    for i, c in enumerate(candles):
        is_absorption = False
        if i >= vol_window:
            win = candles[i - vol_window:i]
            avg_vol   = sum(x["volume"] for x in win) / len(win)
            avg_range = sum(x["high"] - x["low"] for x in win) / len(win)
            candle_range = c["high"] - c["low"]
            high_vol  = avg_vol > 0 and c["volume"] > vol_mult * avg_vol
            small_rng = avg_range > 0 and candle_range < range_threshold * avg_range
            is_absorption = high_vol and small_rng
        result.append({**c, "is_absorption": is_absorption})
    return result


# ─── Dukascopy helpers ──────────────────────────────────────────────────────
def get_last_trading_hours(hours_back: int):
    now = datetime.now(timezone.utc)
    hours = []
    offset = 1
    while len(hours) < hours_back and offset < 300:
        target = now - timedelta(hours=offset)
        if target.weekday() < 5:
            hours.append((target.year, target.month, target.day, target.hour))
        offset += 1
    return hours


async def fetch_dukascopy_candles(symbol: str, year: int, month: int, day: int, hour: int):
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
        divisor = DIVISORS.get(symbol, 100000.0)
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
                "time":     bucket,
                "open":     t["price"],
                "high":     t["price"],
                "low":      t["price"],
                "close":    t["price"],
                "volume":   0.0,
                "buy_vol":  0.0,
                "sell_vol": 0.0,
                "session":  get_session(bucket),
            }
        c = candles[bucket]
        c["high"]     = max(c["high"], t["price"])
        c["low"]      = min(c["low"],  t["price"])
        c["close"]    = t["price"]
        c["volume"]  += t["ask_vol"] + t["bid_vol"]
        c["buy_vol"] += t["ask_vol"]
        c["sell_vol"]+= t["bid_vol"]
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
            "avg_vol":    round(avg_vol, 4),
            "is_spike":   is_spike,
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
        "poc":         poc["price"],
        "poc_volume":  round(poc["volume"], 4),
        "vah":         round(max(va_prices), 2),
        "val":         round(min(va_prices), 2),
        "total_volume": round(total_vol, 4),
    }


# ─── Session stats summary ─────────────────────────────────────────────────
def compute_session_stats(candles: list) -> dict:
    """Aggregate volume and delta per session."""
    stats = {}
    for c in candles:
        s = c.get("session", "off")
        if s not in stats:
            stats[s] = {"volume": 0.0, "delta": 0.0, "candles": 0, "spikes": 0}
        stats[s]["volume"]  += c.get("volume", 0)
        stats[s]["delta"]   += c.get("delta", 0)
        stats[s]["candles"] += 1
        if c.get("is_spike"):
            stats[s]["spikes"] += 1
    return {k: {x: round(v, 4) if isinstance(v, float) else v
                for x, v in d.items()}
            for k, d in stats.items()}


# ─── Routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/orderflow/{symbol}")
async def get_order_flow(
    symbol:           str,
    hours_back:       int   = Query(default=3, ge=1, le=12),
    timeframe:        int   = Query(default=60),
    spike_window:     int   = Query(default=20),
    spike_multiplier: float = Query(default=2.0),
    profile_pip_size: float = Query(default=0.5),
    divergence_lookback: int = Query(default=5),
):
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        return {"error": f"Unknown symbol. Supported: {list(INSTRUMENT_MAP.keys())}"}

    hours    = get_last_trading_hours(hours_back)
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
        return {"error": "No data found. Markets may be closed — try again Monday."}

    now = datetime.now(timezone.utc)

    # Pipeline: ticks → candles → delta → spikes → divergence → absorption
    candles  = ticks_to_candles(all_ticks, tf_seconds=timeframe)
    candles  = compute_volume_delta(candles)
    candles  = detect_volume_spikes(candles, window=spike_window, multiplier=spike_multiplier)
    candles  = detect_delta_divergence(candles, lookback=divergence_lookback)
    candles  = detect_absorption(candles)

    profile  = compute_volume_profile(candles, pip_size=profile_pip_size)
    poc_info = find_poc_and_value_area(profile)
    session_stats = compute_session_stats(candles)

    spikes       = [c for c in candles if c["is_spike"]]
    divergences  = [c for c in candles if c["divergence"]]
    absorptions  = [c for c in candles if c["is_absorption"]]
    net_delta    = sum(c["delta"] for c in candles)

    return {
        "symbol":           symbol,
        "generated_at":     now.isoformat(),
        "candle_count":     len(candles),
        "timeframe_seconds": timeframe,
        "data_from":        "last trading session (weekend-aware)",
        "summary": {
            "net_delta":        round(net_delta, 4),
            "bias":             "bullish" if net_delta > 0 else "bearish",
            "spike_count":      len(spikes),
            "divergence_count": len(divergences),
            "absorption_count": len(absorptions),
            "last_price":       candles[-1]["close"] if candles else None,
            **poc_info,
        },
        "session_stats":    session_stats,
        "candles":          candles,
        "volume_profile":   profile,
    }


@app.get("/api/symbols")
async def list_symbols():
    return {"symbols": list(INSTRUMENT_MAP.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
