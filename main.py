from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, struct, lzma, asyncio, math
from datetime import datetime, timezone, timedelta

app = FastAPI(title="Order Flow API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

INSTRUMENT_MAP = {
    "XAUUSD":    "XAUUSD",
    "XAGUSD":    "XAGUSD",
    "USOUSD":    "WTICMDUSD",
    "BRTUSD":    "BRENTCMDUSD",
    "NATGASUSD": "NATGASUSD",
    "EURUSD":    "EURUSD",
    "GBPUSD":    "GBPUSD",
    "USDJPY":    "USDJPY",
    "BTCUSD":    "BTCUSD",
    "SPXUSD":    "SPX500",
    "NDXUSD":    "NAS100",
}

DIVISORS = {
    "XAUUSD": 1000.0, "XAGUSD": 1000.0, "WTICMDUSD": 1000.0,
    "BRENTCMDUSD": 1000.0, "NATGASUSD": 10000.0,
    "EURUSD": 100000.0, "GBPUSD": 100000.0, "USDJPY": 1000.0,
    "BTCUSD": 1.0, "SPX500": 1000.0, "NAS100": 1000.0,
}

def get_session(epoch):
    h = datetime.utcfromtimestamp(epoch).hour
    if 13 <= h < 16: return "overlap"
    if 7  <= h < 13: return "london"
    if 13 <= h < 21: return "ny"
    if 0  <= h <  7: return "asia"
    return "off"

def get_last_trading_hours(hours_back):
    now, hours, offset = datetime.now(timezone.utc), [], 1
    while len(hours) < hours_back and offset < 300:
        t = now - timedelta(hours=offset)
        if t.weekday() < 5:
            hours.append((t.year, t.month, t.day, t.hour))
        offset += 1
    return hours

async def fetch_raw_ticks(duka_sym, year, month, day, hour):
    url = f"{DUKA_BASE}/{duka_sym}/{year:04d}/{(month-1):02d}/{day:02d}/{hour:02d}h_ticks.bi5"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200: return []
    div = DIVISORS.get(duka_sym, 100000.0)
    raw = lzma.decompress(r.content)
    ticks = []
    for i in range(0, len(raw) - 19, 20):
        ms, ask_r, bid_r, av, bv = struct.unpack(">IIIff", raw[i:i+20])
        ts = datetime(year, month, day, hour, tzinfo=timezone.utc) + timedelta(milliseconds=ms)
        ticks.append({
            "ts": ts,
            "price": (ask_r + bid_r) / 2 / div,
            "ask_vol": float(av),
            "bid_vol": float(bv),
        })
    return ticks

# ── Simple candle builder — no tick storage, no footprint ────────────────────
def ticks_to_candles(ticks, tf_seconds=60):
    if not ticks: return []
    candles = {}
    for t in ticks:
        b = int(t["ts"].timestamp() // tf_seconds) * tf_seconds
        if b not in candles:
            candles[b] = {
                "time": b, "open": t["price"], "high": t["price"],
                "low":  t["price"], "close": t["price"],
                "volume": 0.0, "buy_vol": 0.0, "sell_vol": 0.0,
                "session": get_session(b),
            }
        c = candles[b]
        c["high"]    = max(c["high"], t["price"])
        c["low"]     = min(c["low"],  t["price"])
        c["close"]   = t["price"]
        c["volume"]  += t["ask_vol"] + t["bid_vol"]
        c["buy_vol"] += t["ask_vol"]
        c["sell_vol"]+= t["bid_vol"]
    return sorted(candles.values(), key=lambda x: x["time"])

def compute_volume_delta(candles):
    cum, out = 0.0, []
    for c in candles:
        d = c["buy_vol"] - c["sell_vol"]
        cum += d
        out.append({**c, "delta": round(d, 4), "cum_delta": round(cum, 4)})
    return out

def compute_volume_profile(candles, pip_size=0.5):
    profile = {}
    for c in candles:
        lo = math.floor(c["low"] / pip_size) * pip_size
        hi = math.floor(c["high"] / pip_size) * pip_size
        n  = max(1, round((hi - lo) / pip_size) + 1)
        vpb = c["volume"] / n
        p = lo
        while p <= hi + 0.001:
            k = round(p, 2)
            profile[k] = profile.get(k, 0) + vpb
            p = round(p + pip_size, 2)
    if not profile: return []
    mv = max(profile.values())
    return [{"price": p, "volume": round(v, 4), "pct": round(v / mv * 100, 1)}
            for p, v in sorted(profile.items())]

def detect_volume_spikes(candles, window=20, mult=2.0):
    out = []
    for i, c in enumerate(candles):
        vols = [x["volume"] for x in candles[max(0, i - window):i]]
        avg  = sum(vols) / len(vols) if vols else 0
        spike = avg > 0 and c["volume"] > mult * avg
        out.append({**c, "avg_vol": round(avg, 4), "is_spike": spike,
                    "spike_ratio": round(c["volume"] / avg, 2) if avg > 0 else 0})
    return out

def detect_delta_divergence(candles, lookback=5):
    out = []
    for i, c in enumerate(candles):
        div, strength = None, 0.0
        if i >= lookback:
            w  = candles[i - lookback:i + 1]
            pp = [x["close"] for x in w]
            dp = [x["cum_delta"] for x in w]
            php, phd = max(pp[:-1]), max(dp[:-1])
            plp, pld = min(pp[:-1]), min(dp[:-1])
            cp, cd = c["close"], c["cum_delta"]
            if cp > php and cd < phd:
                strength = round(min(100, ((cp-php)/(php+1e-9) + (phd-cd)/(abs(phd)+1e-9)) * 50), 1)
                div = "bearish"
            elif cp < plp and cd > pld:
                strength = round(min(100, ((plp-cp)/(plp+1e-9) + (cd-pld)/(abs(pld)+1e-9)) * 50), 1)
                div = "bullish"
        out.append({**c, "divergence": div, "divergence_strength": strength})
    return out

def detect_absorption(candles, vw=20, vm=1.8, rt=0.35):
    out = []
    for i, c in enumerate(candles):
        ab = False
        if i >= vw:
            win = candles[i - vw:i]
            av = sum(x["volume"] for x in win) / len(win)
            ar = sum(x["high"] - x["low"] for x in win) / len(win)
            ab = av > 0 and c["volume"] > vm * av and ar > 0 and (c["high"] - c["low"]) < rt * ar
        out.append({**c, "is_absorption": ab})
    return out

def find_poc_and_value_area(profile, pct=0.70):
    if not profile: return {}
    tv  = sum(p["volume"] for p in profile)
    poc = max(profile, key=lambda x: x["volume"])
    sv  = sorted(profile, key=lambda x: x["volume"], reverse=True)
    cov, vp = 0.0, []
    for lvl in sv:
        cov += lvl["volume"]; vp.append(lvl["price"])
        if cov >= tv * pct: break
    return {"poc": poc["price"], "poc_volume": round(poc["volume"], 4),
            "vah": round(max(vp), 2), "val": round(min(vp), 2),
            "total_volume": round(tv, 4)}

def compute_session_stats(candles):
    stats = {}
    for c in candles:
        s = c.get("session", "off")
        if s not in stats: stats[s] = {"volume": 0.0, "delta": 0.0, "candles": 0, "spikes": 0}
        stats[s]["volume"]  += c.get("volume", 0)
        stats[s]["delta"]   += c.get("delta", 0)
        stats[s]["candles"] += 1
        if c.get("is_spike"): stats[s]["spikes"] += 1
    return {k: {x: round(v, 4) if isinstance(v, float) else v for x, v in d.items()}
            for k, d in stats.items()}

def pearson_correlation(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3: return 0.0
    xs, ys = xs[-n:], ys[-n:]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    da  = math.sqrt(sum((a - mx) ** 2 for a in xs))
    db  = math.sqrt(sum((b - my) ** 2 for b in ys))
    if da == 0 or db == 0: return 0.0
    return round(num / (da * db), 4)

# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Main order flow — lean response, NO heatmap ──────────────────────────────
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

    duka_sym  = INSTRUMENT_MAP[symbol]
    hours     = get_last_trading_hours(hours_back)
    all_ticks = []

    async def fh(args):
        y, mo, d, hr = args
        try: return await fetch_raw_ticks(duka_sym, y, mo, d, hr)
        except: return []

    for r in await asyncio.gather(*[fh(h) for h in hours]):
        all_ticks.extend(r)
    all_ticks.sort(key=lambda x: x["ts"])

    if not all_ticks:
        return {"error": "No data found. Markets may be closed — try again Monday."}

    now     = datetime.now(timezone.utc)
    candles = ticks_to_candles(all_ticks, tf_seconds=timeframe)
    candles = compute_volume_delta(candles)
    candles = detect_volume_spikes(candles, window=spike_window, multiplier=spike_multiplier)
    candles = detect_delta_divergence(candles, lookback=divergence_lookback)
    candles = detect_absorption(candles)

    profile    = compute_volume_profile(candles, pip_size=profile_pip_size)
    poc_info   = find_poc_and_value_area(profile)
    sess_stats = compute_session_stats(candles)

    spikes    = [c for c in candles if c["is_spike"]]
    divs      = [c for c in candles if c["divergence"]]
    absorbs   = [c for c in candles if c["is_absorption"]]
    net_delta = sum(c["delta"] for c in candles)

    return {
        "symbol":          symbol,
        "generated_at":    now.isoformat(),
        "candle_count":    len(candles),
        "timeframe_seconds": timeframe,
        "data_from":       "last trading session (weekend-aware)",
        "summary": {
            "net_delta":        round(net_delta, 4),
            "bias":             "bullish" if net_delta > 0 else "bearish",
            "spike_count":      len(spikes),
            "divergence_count": len(divs),
            "absorption_count": len(absorbs),
            "last_price":       candles[-1]["close"] if candles else None,
            **poc_info,
        },
        "session_stats":  sess_stats,
        "candles":        candles,
        "volume_profile": profile,
    }


# ── Heatmap — separate endpoint ───────────────────────────────────────────────
@app.get("/api/heatmap/{symbol}")
async def get_heatmap(
    symbol:      str,
    hours_back:  int = Query(default=2, ge=1, le=6),
    timeframe:   int = Query(default=60),
    price_buckets: int = Query(default=30, ge=10, le=50),
):
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        return {"error": "Unknown symbol"}

    duka_sym  = INSTRUMENT_MAP[symbol]
    hours     = get_last_trading_hours(hours_back)
    all_ticks = []

    async def fh(args):
        y, mo, d, hr = args
        try: return await fetch_raw_ticks(duka_sym, y, mo, d, hr)
        except: return []

    for r in await asyncio.gather(*[fh(h) for h in hours]):
        all_ticks.extend(r)
    all_ticks.sort(key=lambda x: x["ts"])
    if not all_ticks:
        return {"error": "No data found"}

    candles = ticks_to_candles(all_ticks, tf_seconds=timeframe)

    all_p = [p for c in candles for p in [c["high"], c["low"]]]
    min_p, max_p = min(all_p), max(all_p)
    rng = max_p - min_p
    if rng <= 1e-9:
        return {"symbol": symbol, "cells": []}

    bs = rng / price_buckets
    cells = {}
    for c in candles:
        lo_b = max(0, int((c["low"]  - min_p) / bs))
        hi_b = min(price_buckets - 1, int((c["high"] - min_p) / bs))
        n = max(1, hi_b - lo_b + 1)
        vpb = c["volume"] / n
        for b in range(lo_b, hi_b + 1):
            key = (c["time"], b)
            cells[key] = cells.get(key, 0) + vpb

    if not cells:
        return {"symbol": symbol, "cells": []}

    max_v = max(cells.values())
    result = [
        {
            "time":     t,
            "price":    round(min_p + b * bs + bs / 2, 4),
            "price_lo": round(min_p + b * bs, 4),
            "price_hi": round(min_p + (b + 1) * bs, 4),
            "pct":      round(v / max_v * 100, 1),
        }
        for (t, b), v in cells.items()
    ]
    return {"symbol": symbol, "cells": result, "min_price": round(min_p, 4), "max_price": round(max_p, 4)}


# ── Footprint — separate endpoint ─────────────────────────────────────────────
@app.get("/api/footprint/{symbol}")
async def get_footprint(
    symbol:      str,
    hours_back:  int = Query(default=2, ge=1, le=4),
    timeframe:   int = Query(default=60),
    fp_levels:   int = Query(default=10, ge=5, le=20),
    num_candles: int = Query(default=30, ge=5, le=50),
):
    symbol = symbol.upper()
    if symbol not in INSTRUMENT_MAP:
        return {"error": "Unknown symbol"}

    duka_sym  = INSTRUMENT_MAP[symbol]
    hours     = get_last_trading_hours(hours_back)
    all_ticks = []

    async def fh(args):
        y, mo, d, hr = args
        try: return await fetch_raw_ticks(duka_sym, y, mo, d, hr)
        except: return []

    for r in await asyncio.gather(*[fh(h) for h in hours]):
        all_ticks.extend(r)
    all_ticks.sort(key=lambda x: x["ts"])
    if not all_ticks:
        return {"error": "No data found"}

    # Build only simple candles first, keep last N
    simple = ticks_to_candles(all_ticks, tf_seconds=timeframe)
    recent_times = {c["time"] for c in simple[-num_candles:]}

    # Re-aggregate only the ticks belonging to those N candles
    fp_candles = {}
    for t in all_ticks:
        b = int(t["ts"].timestamp() // timeframe) * timeframe
        if b not in recent_times:
            continue
        if b not in fp_candles:
            fp_candles[b] = {
                "time": b, "open": t["price"], "high": t["price"],
                "low":  t["price"], "close": t["price"],
                "volume": 0.0, "buy_vol": 0.0, "sell_vol": 0.0,
                "session": get_session(b),
                "_ticks": [],
            }
        c = fp_candles[b]
        c["high"]    = max(c["high"], t["price"])
        c["low"]     = min(c["low"],  t["price"])
        c["close"]   = t["price"]
        c["volume"]  += t["ask_vol"] + t["bid_vol"]
        c["buy_vol"] += t["ask_vol"]
        c["sell_vol"]+= t["bid_vol"]
        c["_ticks"].append(t)

    result = []
    for c in sorted(fp_candles.values(), key=lambda x: x["time"]):
        rng = c["high"] - c["low"]
        footprint = []
        if rng > 1e-9:
            bs = rng / fp_levels
            for lvl in range(fp_levels):
                p_lo = c["low"] + lvl * bs
                p_hi = p_lo + bs
                is_last = lvl == fp_levels - 1
                bv = sv = 0.0
                for t in c["_ticks"]:
                    in_b = (p_lo <= t["price"] <= p_hi) if is_last else (p_lo <= t["price"] < p_hi)
                    if in_b:
                        bv += t["ask_vol"]
                        sv += t["bid_vol"]
                footprint.append({
                    "price":    round((p_lo + p_hi) / 2, 4),
                    "price_lo": round(p_lo, 4),
                    "price_hi": round(p_hi, 4),
                    "buy_vol":  round(bv, 4),
                    "sell_vol": round(sv, 4),
                    "delta":    round(bv - sv, 4),
                    "total":    round(bv + sv, 4),
                })
        result.append({
            "time": c["time"], "open": c["open"], "high": c["high"],
            "low": c["low"], "close": c["close"],
            "volume": round(c["volume"], 4),
            "buy_vol": round(c["buy_vol"], 4),
            "sell_vol": round(c["sell_vol"], 4),
            "session": c["session"],
            "footprint": footprint,
        })

    return {"symbol": symbol, "candles": result, "generated_at": datetime.now(timezone.utc).isoformat()}


# ── Correlation — separate endpoint ───────────────────────────────────────────
@app.get("/api/correlation")
async def get_correlation(
    hours_back: int = Query(default=3, ge=1, le=6),
    timeframe:  int = Query(default=300),
):
    symbols = list(INSTRUMENT_MAP.keys())
    hours   = get_last_trading_hours(hours_back)

    async def fetch_closes(sym):
        duka = INSTRUMENT_MAP[sym]
        ticks = []
        async def fh(args):
            y, mo, d, hr = args
            try: return await fetch_raw_ticks(duka, y, mo, d, hr)
            except: return []
        for r in await asyncio.gather(*[fh(h) for h in hours]):
            ticks.extend(r)
        ticks.sort(key=lambda x: x["ts"])
        if not ticks: return sym, []
        return sym, [c["close"] for c in ticks_to_candles(ticks, tf_seconds=timeframe)]

    results = await asyncio.gather(*[fetch_closes(s) for s in symbols])
    closes  = dict(results)

    matrix = {}
    for s1 in symbols:
        matrix[s1] = {}
        for s2 in symbols:
            if s1 == s2: matrix[s1][s2] = 1.0
            elif closes.get(s1) and closes.get(s2):
                matrix[s1][s2] = pearson_correlation(closes[s1], closes[s2])
            else: matrix[s1][s2] = None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbols":      symbols,
        "matrix":       matrix,
        "last_prices":  {s: closes[s][-1] if closes.get(s) else None for s in symbols},
    }


@app.get("/api/symbols")
async def list_symbols():
    return {"symbols": list(INSTRUMENT_MAP.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
