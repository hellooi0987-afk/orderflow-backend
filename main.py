from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx, struct, lzma, asyncio, math, statistics
from datetime import datetime, timezone, timedelta

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DUKA_BASE = "https://datafeed.dukascopy.com/datafeed"

INSTRUMENT_MAP = {
    "XAUUSD":"XAUUSD",    "XAGUSD":"XAGUSD",
    "USOUSD":"WTICMDUSD", "BRTUSD":"BRENTCMDUSD",
    "NATGASUSD":"NATGASUSD",
    "EURUSD":"EURUSD",    "GBPUSD":"GBPUSD",  "USDJPY":"USDJPY",
    "BTCUSD":"BTCUSD",    "SPXUSD":"SPX500",  "NDXUSD":"NAS100",
}

DIVISORS = {
    "XAUUSD":1000,"XAGUSD":1000,"WTICMDUSD":1000,"BRENTCMDUSD":1000,
    "NATGASUSD":10000,"EURUSD":100000,"GBPUSD":100000,
    "USDJPY":1000,"BTCUSD":1,"SPX500":1000,"NAS100":1000,
}

def get_session(h):
    if 13<=h<16: return "overlap"
    if 7<=h<13:  return "london"
    if 13<=h<21: return "ny"
    if 0<=h<7:   return "asia"
    return "off"

def last_trading_hours(n):
    now,out,off = datetime.now(timezone.utc),[],1
    while len(out)<n and off<300:
        t = now-timedelta(hours=off)
        if t.weekday()<5: out.append((t.year,t.month,t.day,t.hour))
        off+=1
    return out

async def get_ticks(sym,y,mo,d,h):
    url = f"{DUKA_BASE}/{sym}/{y:04d}/{mo-1:02d}/{d:02d}/{h:02d}h_ticks.bi5"
    try:
        async with httpx.AsyncClient(timeout=12) as c:
            r = await c.get(url, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code!=200: return []
        div = DIVISORS.get(sym,100000)
        raw = lzma.decompress(r.content)
        out = []
        for i in range(0,len(raw)-19,20):
            ms,ar,br,av,bv = struct.unpack(">IIIff",raw[i:i+20])
            ts = datetime(y,mo,d,h,tzinfo=timezone.utc)+timedelta(milliseconds=ms)
            out.append({"ts":ts,"price":(ar+br)/2/div,"av":float(av),"bv":float(bv)})
        return out
    except: return []

def build_candles(ticks,tf):
    if not ticks: return []
    c = {}
    for t in ticks:
        b = int(t["ts"].timestamp()//tf)*tf
        if b not in c:
            h = datetime.utcfromtimestamp(b).hour
            c[b] = {"time":b,"open":t["price"],"high":t["price"],"low":t["price"],
                    "close":t["price"],"volume":0.0,"buy":0.0,"sell":0.0,"session":get_session(h)}
        x = c[b]
        x["high"]=max(x["high"],t["price"]); x["low"]=min(x["low"],t["price"])
        x["close"]=t["price"]; x["volume"]+=t["av"]+t["bv"]
        x["buy"]+=t["av"];     x["sell"]+=t["bv"]
    return sorted(c.values(),key=lambda x:x["time"])

# ── Footprint: buy/sell vol per price tick inside each candle ──────────────
def build_footprint(ticks, tf, pip_size=0.5):
    """
    Returns list of candles where each candle has a 'levels' dict:
    { price_level: { buy, sell, delta } }
    """
    buckets = {}
    for t in ticks:
        b = int(t["ts"].timestamp()//tf)*tf
        if b not in buckets: buckets[b] = []
        buckets[b].append(t)

    result = []
    for b in sorted(buckets.keys()):
        tlist = buckets[b]
        levels = {}
        for t in tlist:
            pl = round(math.floor(t["price"]/pip_size)*pip_size, 4)
            if pl not in levels: levels[pl] = {"buy":0.0,"sell":0.0,"delta":0.0}
            levels[pl]["buy"]  += t["av"]
            levels[pl]["sell"] += t["bv"]
        # Compute delta per level and find dominant
        max_vol = 0.0
        for pl,lv in levels.items():
            lv["delta"] = round(lv["buy"]-lv["sell"],6)
            lv["buy"]   = round(lv["buy"],6)
            lv["sell"]  = round(lv["sell"],6)
            max_vol = max(max_vol, lv["buy"]+lv["sell"])
        # Convert to sorted list for serialisation
        lvl_list = [{"price":pl,"buy":round(lv["buy"],6),"sell":round(lv["sell"],6),
                     "delta":round(lv["delta"],6),
                     "pct":round((lv["buy"]+lv["sell"])/max_vol*100,1) if max_vol>0 else 0}
                    for pl,lv in sorted(levels.items())]
        result.append({"time":b,"levels":lvl_list,"max_vol":round(max_vol,4)})
    return result

# ── Heatmap: volume clustered by time × price ──────────────────────────────
def build_heatmap(ticks, tf, pip_size=0.5, max_cols=60):
    """
    Returns { times:[], prices:[], cells:[[vol,...]], max_vol }
    times = sorted list of candle epochs (limited to max_cols most recent)
    prices = sorted list of price levels
    cells[price_idx][time_idx] = volume
    """
    # Build time→price→vol grid
    grid = {}
    for t in ticks:
        b = int(t["ts"].timestamp()//tf)*tf
        pl = round(math.floor(t["price"]/pip_size)*pip_size, 4)
        if b not in grid: grid[b] = {}
        if pl not in grid[b]: grid[b][pl] = 0.0
        grid[b][pl] += t["av"]+t["bv"]

    times  = sorted(grid.keys())[-max_cols:]
    prices_set = set()
    for b in times:
        prices_set.update(grid[b].keys())
    prices = sorted(prices_set)

    if not times or not prices: return {"times":[],"prices":[],"cells":[],"max_vol":0}

    cells = []
    max_vol = 0.0
    for pl in prices:
        row = []
        for b in times:
            v = grid.get(b,{}).get(pl,0.0)
            row.append(round(v,4))
            max_vol = max(max_vol,v)
        cells.append(row)

    return {"times":times,"prices":prices,"cells":cells,"max_vol":round(max_vol,4)}

# ── Correlation: Pearson across symbols ───────────────────────────────────
def pearson(a, b):
    n = min(len(a),len(b))
    if n < 5: return 0.0
    a,b = a[-n:],b[-n:]
    try:
        return round(statistics.correlation(a,b), 3)
    except: return 0.0

def add_delta(candles):
    cum=0.0; out=[]
    for c in candles:
        d=c["buy"]-c["sell"]; cum+=d
        out.append({**c,"delta":round(d,4),"cum_delta":round(cum,4)})
    return out

def add_spikes(candles,w=20,m=2.0):
    out=[]
    for i,c in enumerate(candles):
        vols=[x["volume"] for x in candles[max(0,i-w):i]]
        avg=sum(vols)/len(vols) if vols else 0
        sp=avg>0 and c["volume"]>m*avg
        out.append({**c,"avg_vol":round(avg,4),"is_spike":sp,
                    "spike_ratio":round(c["volume"]/avg,2) if avg>0 else 0})
    return out

def add_divergence(candles,lb=5):
    out=[]
    for i,c in enumerate(candles):
        div=None; st=0.0
        if i>=lb:
            w=candles[i-lb:i+1]
            pp=[x["close"] for x in w]; dp=[x["cum_delta"] for x in w]
            cp,cd=c["close"],c["cum_delta"]
            if cp>max(pp[:-1]) and cd<max(dp[:-1]):
                st=round(min(100,((cp-max(pp[:-1]))/(max(pp[:-1])+1e-9)+(max(dp[:-1])-cd)/(abs(max(dp[:-1]))+1e-9))*50),1); div="bearish"
            elif cp<min(pp[:-1]) and cd>min(dp[:-1]):
                st=round(min(100,((min(pp[:-1])-cp)/(min(pp[:-1])+1e-9)+(cd-min(dp[:-1]))/(abs(min(dp[:-1]))+1e-9))*50),1); div="bullish"
        out.append({**c,"divergence":div,"div_strength":st})
    return out

def add_absorption(candles,w=20):
    out=[]
    for i,c in enumerate(candles):
        ab=False
        if i>=w:
            win=candles[i-w:i]
            av=sum(x["volume"] for x in win)/len(win)
            ar=sum(x["high"]-x["low"] for x in win)/len(win)
            ab=av>0 and c["volume"]>1.8*av and ar>0 and (c["high"]-c["low"])<0.35*ar
        out.append({**c,"is_absorption":ab})
    return out

def volume_profile(candles,ps=0.5):
    pf={}
    for c in candles:
        lo=math.floor(c["low"]/ps)*ps; hi=math.floor(c["high"]/ps)*ps
        n=max(1,round((hi-lo)/ps)+1); vpb=c["volume"]/n; p=lo
        while p<=hi+0.001:
            k=round(p,2); pf[k]=pf.get(k,0)+vpb; p=round(p+ps,2)
    if not pf: return []
    mv=max(pf.values())
    return [{"price":p,"volume":round(v,4),"pct":round(v/mv*100,1)} for p,v in sorted(pf.items())]

def poc_va(pf,pct=0.70):
    if not pf: return {}
    tv=sum(p["volume"] for p in pf)
    poc=max(pf,key=lambda x:x["volume"])
    sv=sorted(pf,key=lambda x:x["volume"],reverse=True)
    cov,vp=0.0,[]
    for lvl in sv:
        cov+=lvl["volume"]; vp.append(lvl["price"])
        if cov>=tv*pct: break
    return {"poc":poc["price"],"poc_vol":round(poc["volume"],4),"vah":round(max(vp),2),"val":round(min(vp),2)}

def session_stats(candles):
    s={}
    for c in candles:
        k=c.get("session","off")
        if k not in s: s[k]={"volume":0.0,"delta":0.0,"candles":0,"spikes":0}
        s[k]["volume"]+=c.get("volume",0); s[k]["delta"]+=c.get("delta",0)
        s[k]["candles"]+=1
        if c.get("is_spike"): s[k]["spikes"]+=1
    return {k:{x:round(v,4) if isinstance(v,float) else v for x,v in d.items()} for k,d in s.items()}

@app.get("/api/health")
async def health():
    return {"status":"ok","time":datetime.utcnow().isoformat()}

@app.get("/api/orderflow/{symbol}")
async def orderflow(
    symbol:str,
    hours_back:int=Query(3,ge=1,le=6),
    timeframe:int=Query(60),
    spike_window:int=Query(20),
    spike_mult:float=Query(2.0),
    pip_size:float=Query(0.5),
    include_footprint:bool=Query(False),
    include_heatmap:bool=Query(False),
):
    sym = symbol.upper()
    if sym not in INSTRUMENT_MAP: return {"error":"Unknown symbol"}
    duka = INSTRUMENT_MAP[sym]
    hours = last_trading_hours(hours_back)
    all_ticks = []
    results = await asyncio.gather(*[get_ticks(duka,y,mo,d,h) for y,mo,d,h in hours])
    for r in results: all_ticks.extend(r)
    all_ticks.sort(key=lambda x:x["ts"])
    if not all_ticks: return {"error":"No data. Markets closed — try Monday."}

    candles = build_candles(all_ticks,timeframe)
    candles = add_delta(candles)
    candles = add_spikes(candles,spike_window,spike_mult)
    candles = add_divergence(candles)
    candles = add_absorption(candles)

    pf  = volume_profile(candles,pip_size)
    ki  = poc_va(pf)
    ss  = session_stats(candles)
    net = sum(c["delta"] for c in candles)

    resp = {
        "symbol":sym,"candle_count":len(candles),"timeframe_seconds":timeframe,
        "data_from":"last trading session (weekend-aware)",
        "summary":{
            "net_delta":round(net,4),"bias":"bullish" if net>0 else "bearish",
            "spike_count":sum(1 for c in candles if c["is_spike"]),
            "divergence_count":sum(1 for c in candles if c["divergence"]),
            "absorption_count":sum(1 for c in candles if c["is_absorption"]),
            "last_price":candles[-1]["close"] if candles else None,**ki,
        },
        "session_stats":ss,"candles":candles,"volume_profile":pf,
    }

    if include_footprint:
        resp["footprint"] = build_footprint(all_ticks, timeframe, pip_size)
    if include_heatmap:
        resp["heatmap"] = build_heatmap(all_ticks, timeframe, pip_size)

    return resp

@app.get("/api/correlation")
async def correlation(
    hours_back:int=Query(3,ge=1,le=6),
    timeframe:int=Query(300),
):
    """Compute Pearson correlation matrix across all symbols using close prices."""
    symbols = list(INSTRUMENT_MAP.keys())
    hours   = last_trading_hours(hours_back)

    async def fetch_sym(sym):
        duka = INSTRUMENT_MAP[sym]
        ticks = []
        results = await asyncio.gather(*[get_ticks(duka,y,mo,d,h) for y,mo,d,h in hours])
        for r in results: ticks.extend(r)
        ticks.sort(key=lambda x:x["ts"])
        if not ticks: return sym, []
        candles = build_candles(ticks, timeframe)
        return sym, [c["close"] for c in candles]

    pairs = await asyncio.gather(*[fetch_sym(s) for s in symbols])
    prices = {s:p for s,p in pairs if p}

    # Align all series to same length (use last N common candles)
    min_len = min((len(v) for v in prices.values()), default=0)
    if min_len < 5:
        return {"error":"Not enough data for correlation. Try more hours or check markets are open."}

    aligned = {s:v[-min_len:] for s,v in prices.items()}
    syms = list(aligned.keys())

    matrix = {}
    for a in syms:
        matrix[a] = {}
        for b in syms:
            matrix[a][b] = pearson(aligned[a],aligned[b])

    return {"symbols":syms,"matrix":matrix,"candles_used":min_len}

@app.get("/api/symbols")
async def symbols():
    return {"symbols":list(INSTRUMENT_MAP.keys())}

if __name__=="__main__":
    import uvicorn, os
    uvicorn.run("main:app",host="0.0.0.0",port=int(os.getenv("PORT",8000)))
