"""
Kite Option-Selling Dashboard — local backend
------------------------------------------------
Run:  python backend.py
Then open: http://localhost:5000

What this does
- Logs you into Kite Connect (daily login, token expires every day - that's Kite's design, not a bug here)
- Pulls your F&O stock universe from Kite's instrument list, ranks by historical volatility / ATR
- Also supports index options directly: NIFTY, BANKNIFTY, FINNIFTY — just type the symbol
- Lets you pick WHICH expiry (current month, next month, etc.) rather than only the nearest one
- Shows the full live option chain, lets you build and adjust a delta-based Iron Condor OR a naked
  Strangle (you control target delta, hedge width, and lot count before committing)
- Tracks entered trades with live daily P&L, re-estimated probability of success, and max loss
- Lets you actually PLACE the real orders for a tracked position in your Zerodha account — but only
  after an explicit confirmation step showing exactly what will be sent, and gives you an order list
  with cancel/modify so you stay in control the whole time
- Optional lightweight news headlines per stock as a basic event-risk / "threat intelligence" signal

IMPORTANT
- Nothing here is investment advice. Verify every number on your broker terminal before trading.
- Kite access tokens expire every day at ~6am IST. You will need to log in again each trading day.
- Naked strangles carry theoretically unlimited risk on the call side.
- ORDER EXECUTION IS REAL. Placing orders through this tool sends real orders to your live Zerodha
  account using real money. Nothing is placed without you explicitly confirming on the preview screen.
  If one leg of a multi-leg order fails, you may be left holding a partial, unhedged position — the
  tool stops immediately on the first failure and tells you to check your Zerodha app right away.
"""

import os
import math
import time
import json
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_from_directory, redirect

try:
    from kiteconnect import KiteConnect
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install kiteconnect flask numpy requests")

import numpy as np
import requests

# ---------------------------------------------------------------------------
# CONFIG — fill these in from https://developers.kite.trade (your app)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("KITE_API_KEY", "b4j9bna5hdew1hh4")
API_SECRET = os.environ.get("KITE_API_SECRET", "mbrdjydzd9ckisvrp4tsqbtkkgojpzue")
REDIRECT_URL = "http://localhost:5000/api/callback"   # set this exact URL in your Kite app settings

# If your network does TLS interception (common on office/government networks — you'll see
# "self-signed certificate in certificate chain" errors), set this env var to allow the news
# feature specifically to fall back to an unverified request. This does NOT affect Kite API calls
# at all (those always stay fully verified) — it only relaxes verification for public news RSS
# feeds, which carry no credentials or sensitive data.
ALLOW_INSECURE_NEWS = os.environ.get("ALLOW_INSECURE_NEWS", "false").lower() == "true"

RISK_FREE_RATE = 0.07
MIN_DAYS_TO_EXPIRY = 7
DEFAULT_TARGET_DELTA = 0.18
DEFAULT_WING_WIDTH_PCT = 0.05
CHAIN_STRIKE_RANGE_PCT = 0.25

# Index symbols you can type directly (in addition to any F&O stock) — maps to the exact
# Kite quote key Kite uses for that index's live spot price.
INDEX_SYMBOLS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
}

app = Flask(__name__, static_folder="static", static_url_path="")
kite = KiteConnect(api_key=API_KEY)

SESSION = {"access_token": None, "logged_in_at": None}
INSTRUMENT_CACHE = {"nfo": None, "nse": None, "fetched_at": None}
SCREENER_CACHE = {"results": None, "fetched_at": None}

POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "positions.json")
_positions_lock = threading.Lock()


def load_positions():
    if not os.path.exists(POSITIONS_FILE):
        return []
    try:
        with open(POSITIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_positions(positions):
    with _positions_lock:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)


def find_position(pos_id):
    for p in load_positions():
        if p["id"] == pos_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = (S - K) if opt_type == "CE" else (K - S)
        return max(0.0, intrinsic)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return 1.0 if (opt_type == "CE" and S > K) else (0.0 if opt_type == "CE" else (-1.0 if S < K else 0.0))
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if opt_type == "CE" else (norm_cdf(d1) - 1)


def implied_vol(price, S, K, T, opt_type, r=RISK_FREE_RATE):
    if price <= 0 or T <= 0:
        return 0.0
    lo, hi = 0.001, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, r, mid, opt_type)
        if p > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def extract_price(quote):
    """Fall back to bid/ask midpoint when last_price is 0 (illiquid/deep-OTM contracts that
    haven't traded today but still have resting orders) instead of silently dropping the strike."""
    if not quote:
        return None
    ltp = quote.get("last_price", 0)
    if ltp and ltp > 0:
        return ltp
    depth = quote.get("depth", {}) or {}
    buys = [b for b in depth.get("buy", []) if b.get("price", 0) > 0]
    sells = [s for s in depth.get("sell", []) if s.get("price", 0) > 0]
    bid = buys[0]["price"] if buys else None
    ask = sells[0]["price"] if sells else None
    if bid and ask:
        return (bid + ask) / 2
    if ask:
        return ask
    if bid:
        return bid
    return None


def compute_margin(legs_for_margin, quantity, product="NRML"):
    """legs_for_margin: list of {'tradingsymbol': ..., 'transaction_type': 'BUY'/'SELL'}.
    Returns (margin_amount_or_None, error_message_or_None). Uses Kite's basket margin API
    where available (accounts for the margin benefit of hedged combos like an iron condor);
    falls back to summing individual order margins if the basket endpoint isn't available."""
    order_params = []
    for lg in legs_for_margin:
        order_params.append({
            "exchange": "NFO", "tradingsymbol": lg["tradingsymbol"],
            "transaction_type": lg["transaction_type"], "variety": "regular",
            "product": product, "order_type": "MARKET", "quantity": quantity,
        })
    try:
        if hasattr(kite, "basket_order_margins"):
            resp = kite.basket_order_margins(order_params, consider_positions=False)
            total = None
            if isinstance(resp, dict):
                section = resp.get("final") or resp.get("initial") or resp
                if isinstance(section, dict):
                    total = section.get("total")
            if total is None:
                return None, "Unexpected response shape from basket margin API"
            return round(total, 2), None
        else:
            resp = kite.order_margins(order_params)
            total = sum(r.get("total", 0) for r in resp)
            return round(total, 2), None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Kite login flow
# ---------------------------------------------------------------------------
@app.route("/api/login-url")
def login_url():
    return jsonify({"url": kite.login_url()})


@app.route("/api/callback")
def callback():
    request_token = request.args.get("request_token")
    if not request_token:
        return "Login failed: no request_token received.", 400
    data = kite.generate_session(request_token, api_secret=API_SECRET)
    SESSION["access_token"] = data["access_token"]
    SESSION["logged_in_at"] = datetime.now().isoformat()
    kite.set_access_token(SESSION["access_token"])
    return redirect("/")


@app.route("/api/session-status")
def session_status():
    return jsonify({"logged_in": SESSION["access_token"] is not None,
                     "logged_in_at": SESSION["logged_in_at"]})


def require_session():
    if not SESSION["access_token"]:
        return False
    kite.set_access_token(SESSION["access_token"])
    return True


# ---------------------------------------------------------------------------
# Instrument cache
# ---------------------------------------------------------------------------
def get_instruments(force=False):
    now = datetime.now()
    if (force or INSTRUMENT_CACHE["fetched_at"] is None or
            now - INSTRUMENT_CACHE["fetched_at"] > timedelta(hours=6)):
        INSTRUMENT_CACHE["nfo"] = kite.instruments("NFO")
        INSTRUMENT_CACHE["nse"] = kite.instruments("NSE")
        INSTRUMENT_CACHE["fetched_at"] = now
    return INSTRUMENT_CACHE["nfo"], INSTRUMENT_CACHE["nse"]


def fo_stock_universe(force=False):
    """Individual F&O stocks only (indices are offered separately via INDEX_SYMBOLS)."""
    nfo, _ = get_instruments(force=force)
    names = set()
    for ins in nfo:
        if ins["segment"] == "NFO-OPT" and ins["name"] not in INDEX_SYMBOLS:
            names.add(ins["name"])
    return sorted(names)


@app.route("/api/refresh-data", methods=["POST"])
def refresh_data():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    get_instruments(force=True)
    SCREENER_CACHE["results"] = None
    SCREENER_CACHE["fetched_at"] = None
    return jsonify({"ok": True, "message": "Instrument cache cleared. Re-run the screener to refresh rankings."})


# ---------------------------------------------------------------------------
# Volatility screener
# ---------------------------------------------------------------------------
def historical_vol_and_atr(nse_token, days=60):
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days + 20)
    candles = kite.historical_data(nse_token, from_date, to_date, "day")
    if len(candles) < 10:
        return None, None, None
    closes = np.array([c["close"] for c in candles[-days:]])
    highs = np.array([c["high"] for c in candles[-days:]])
    lows = np.array([c["low"] for c in candles[-days:]])
    returns = np.diff(np.log(closes))
    hv_annualized = float(np.std(returns) * math.sqrt(252) * 100)
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    atr = float(np.mean(tr[-14:]))
    last_close = float(closes[-1])
    atr_pct = atr / last_close * 100
    return hv_annualized, atr_pct, last_close


@app.route("/api/screener")
def screener():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    limit = int(request.args.get("limit", 25))
    force = request.args.get("force", "false").lower() == "true"
    universe = fo_stock_universe(force=force)
    _, nse = get_instruments(force=force)
    symbol_to_token = {i["tradingsymbol"]: i["instrument_token"] for i in nse if i["exchange"] == "NSE"}

    results = []
    for name in universe:
        token = symbol_to_token.get(name)
        if not token:
            continue
        try:
            hv, atr_pct, ltp = historical_vol_and_atr(token)
        except Exception:
            continue
        if hv is None:
            continue
        results.append({"symbol": name, "ltp": round(ltp, 2),
                         "hv_annualized_pct": round(hv, 2), "atr_pct_of_price": round(atr_pct, 2)})
        if len(results) >= 300:
            break

    results.sort(key=lambda r: r["hv_annualized_pct"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["total"] = len(results)

    SCREENER_CACHE["results"] = results
    SCREENER_CACHE["fetched_at"] = datetime.now()

    return jsonify({"count": len(results), "stocks": results[:limit],
                     "note": "Lower hv_annualized_pct / atr_pct_of_price = calmer stock. "
                             "Still check upcoming results dates and F&O ban list yourself before trading."})


def get_stock_rank(symbol):
    if not SCREENER_CACHE["results"]:
        return None
    for r in SCREENER_CACHE["results"]:
        if r["symbol"] == symbol:
            return {"rank": r["rank"], "total": r["total"],
                     "hv_annualized_pct": r["hv_annualized_pct"], "atr_pct_of_price": r["atr_pct_of_price"],
                     "screener_age_minutes": round((datetime.now() - SCREENER_CACHE["fetched_at"]).total_seconds() / 60, 1)}
    return None


# ---------------------------------------------------------------------------
# Expiry list + option chain (supports stocks AND indices, any expiry you pick)
# ---------------------------------------------------------------------------
def get_spot_price(symbol):
    """Returns (spot_price, error_dict_or_None). Handles both index symbols and F&O stocks."""
    symbol = symbol.upper()
    if symbol in INDEX_SYMBOLS:
        key = INDEX_SYMBOLS[symbol]
        try:
            quote = kite.quote([key])[key]
            return quote["last_price"], None
        except Exception as e:
            return None, {"error": f"Could not fetch {symbol} index quote: {e}"}
    _, nse = get_instruments()
    nse_match = [i for i in nse if i["exchange"] == "NSE" and i["tradingsymbol"] == symbol]
    if not nse_match:
        return None, {"error": f"{symbol} not found on NSE and not a recognized index (NIFTY/BANKNIFTY/FINNIFTY/MIDCPNIFTY)"}
    quote = kite.quote([f"NSE:{symbol}"])[f"NSE:{symbol}"]
    return quote["last_price"], None


@app.route("/api/expiries/<symbol>")
def expiries(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    symbol = symbol.upper()
    nfo, _ = get_instruments()
    opts = [i for i in nfo if i["name"] == symbol and i["segment"] == "NFO-OPT"]
    if not opts:
        return jsonify({"error": f"No options found for {symbol}"}), 404
    today = datetime.now().date()
    exp_list = sorted({o["expiry"] for o in opts if (o["expiry"] - today).days >= 1})
    return jsonify({"symbol": symbol, "expiries": [str(e) for e in exp_list]})


def get_chain_for_symbol(symbol, expiry_str=None):
    """Returns (data_dict, None) or (None, error_dict). data_dict has spot/expiry/T/lot_size/chain."""
    symbol = symbol.upper()
    spot, err = get_spot_price(symbol)
    if err:
        return None, err

    nfo, _ = get_instruments()
    opts = [i for i in nfo if i["name"] == symbol and i["segment"] == "NFO-OPT"]
    if not opts:
        return None, {"error": f"No options found for {symbol}"}

    today = datetime.now().date()
    all_expiries = sorted({o["expiry"] for o in opts})

    if expiry_str:
        try:
            target_expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            return None, {"error": f"Invalid expiry format '{expiry_str}', expected YYYY-MM-DD"}
        if target_expiry not in all_expiries:
            return None, {"error": f"{expiry_str} is not a valid expiry for {symbol}. "
                                    f"Available: {', '.join(str(e) for e in all_expiries[:6])}"}
        expiry = target_expiry
    else:
        valid = [e for e in all_expiries if (e - today).days >= MIN_DAYS_TO_EXPIRY]
        if not valid:
            return None, {"error": "No expiry beyond minimum days-to-expiry filter"}
        expiry = valid[0]

    T = max((expiry - today).days, 0) / 365.0
    chain = [o for o in opts if o["expiry"] == expiry]
    lot_size = chain[0]["lot_size"]
    inst_keys = [f"NFO:{o['tradingsymbol']}" for o in chain]
    quotes = kite.quote(inst_keys)

    enriched = []
    for o in chain:
        key = f"NFO:{o['tradingsymbol']}"
        q = quotes.get(key)
        ltp = extract_price(q)
        if ltp is None:
            continue
        oi = q.get("oi", 0) if q else 0
        iv = implied_vol(ltp, spot, o["strike"], T, o["instrument_type"])
        delta = bs_delta(spot, o["strike"], T, RISK_FREE_RATE, iv, o["instrument_type"])
        enriched.append({**o, "ltp": ltp, "oi": oi, "iv": round(iv * 100, 1), "delta": round(delta, 3)})

    return {"spot": spot, "expiry": expiry, "T": T, "lot_size": lot_size, "chain": enriched,
            "all_expiries": [str(e) for e in all_expiries]}, None


@app.route("/api/optionchain/<symbol>")
def option_chain(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    expiry_str = request.args.get("expiry")
    data, err = get_chain_for_symbol(symbol, expiry_str)
    if err:
        return jsonify(err), 404

    spot = data["spot"]
    lo = spot * (1 - CHAIN_STRIKE_RANGE_PCT)
    hi = spot * (1 + CHAIN_STRIKE_RANGE_PCT)
    filtered = [o for o in data["chain"] if lo <= o["strike"] <= hi]
    calls = sorted([o for o in filtered if o["instrument_type"] == "CE"], key=lambda x: x["strike"])
    puts = sorted([o for o in filtered if o["instrument_type"] == "PE"], key=lambda x: x["strike"])

    def slim(o):
        return {"strike": o["strike"], "ltp": o["ltp"], "oi": o["oi"], "iv_pct": o["iv"], "delta": o["delta"]}

    return jsonify({
        "symbol": symbol.upper(), "spot": spot, "expiry": str(data["expiry"]), "lot_size": data["lot_size"],
        "all_expiries": data["all_expiries"],
        "calls": [slim(o) for o in calls], "puts": [slim(o) for o in puts]
    })


# ---------------------------------------------------------------------------
# Strategy builder — Iron Condor or Naked Strangle, adjustable delta/wing/lots/expiry
# ---------------------------------------------------------------------------
def build_strategy(symbol, target_delta=DEFAULT_TARGET_DELTA, wing_width_pct=DEFAULT_WING_WIDTH_PCT,
                    strategy_type="iron_condor", expiry_str=None, lots=1):
    data, err = get_chain_for_symbol(symbol, expiry_str)
    if err:
        return err
    spot, expiry, T, lot_size, chain = data["spot"], data["expiry"], data["T"], data["lot_size"], data["chain"]
    today = datetime.now().date()
    quantity = lot_size * max(1, int(lots))

    calls = sorted([e for e in chain if e["instrument_type"] == "CE"], key=lambda x: x["strike"])
    puts = sorted([e for e in chain if e["instrument_type"] == "PE"], key=lambda x: x["strike"])

    def closest_by_delta(options, target, sign):
        best, best_diff = None, None
        for o in options:
            diff = abs(o["delta"] - sign * target)
            if best_diff is None or diff < best_diff:
                best, best_diff = o, diff
        return best

    short_call = closest_by_delta(calls, target_delta, +1)
    short_put = closest_by_delta(puts, target_delta, -1)
    if not short_call or not short_put:
        return {"error": "Could not find suitable strikes near target delta"}

    def leg(o):
        return {"strike": o["strike"], "ltp": o["ltp"], "delta": o["delta"], "tradingsymbol": o["tradingsymbol"]}

    if strategy_type == "naked_strangle":
        net_credit = short_call["ltp"] + short_put["ltp"]
        result = {
            "symbol": symbol.upper(), "spot": spot, "expiry": str(expiry),
            "days_to_expiry": (expiry - today).days, "lot_size": lot_size, "lots": lots, "quantity": quantity,
            "strategy_type": "naked_strangle",
            "legs": {"sell_call": leg(short_call), "sell_put": leg(short_put)},
            "net_credit_per_share": round(net_credit, 2),
            "max_profit": round(net_credit * quantity, 2),
            "max_loss": None,
            "breakeven_upper": round(short_call["strike"] + net_credit, 2),
            "breakeven_lower": round(short_put["strike"] - net_credit, 2),
            "note": "NAKED STRANGLE: max loss is theoretically UNLIMITED on the call side and large "
                    "(capped only by the stock going to zero) on the put side. Margin requirements are "
                    "typically much higher than for an iron condor. Educational calculation only — not advice."
        }
    else:
        wing_call_strike_target = short_call["strike"] * (1 + wing_width_pct)
        wing_put_strike_target = short_put["strike"] * (1 - wing_width_pct)
        calls_above = [o for o in calls if o["strike"] > short_call["strike"]]
        puts_below = [o for o in puts if o["strike"] < short_put["strike"]]

        if not calls_above:
            return {"error": f"No strike available above {short_call['strike']} to use as a call hedge"}
        if not puts_below:
            return {"error": f"No strike available below {short_put['strike']} to use as a put hedge"}

        long_call = min(calls_above, key=lambda o: abs(o["strike"] - wing_call_strike_target))
        long_put = min(puts_below, key=lambda o: abs(o["strike"] - wing_put_strike_target))

        net_credit = (short_call["ltp"] + short_put["ltp"]) - (long_call["ltp"] + long_put["ltp"])
        call_wing = long_call["strike"] - short_call["strike"]
        put_wing = short_put["strike"] - long_put["strike"]
        max_loss_per_share = max(call_wing, put_wing) - net_credit
        max_profit_per_share = net_credit

        available_call_strikes = sorted({o["strike"] for o in calls_above})[:6]
        available_put_strikes = sorted({o["strike"] for o in puts_below}, reverse=True)[:6]

        result = {
            "symbol": symbol.upper(), "spot": spot, "expiry": str(expiry),
            "days_to_expiry": (expiry - today).days, "lot_size": lot_size, "lots": lots, "quantity": quantity,
            "strategy_type": "iron_condor",
            "legs": {"sell_call": leg(short_call), "buy_call": leg(long_call),
                     "sell_put": leg(short_put), "buy_put": leg(long_put)},
            "net_credit_per_share": round(net_credit, 2),
            "max_profit": round(max_profit_per_share * quantity, 2),
            "max_loss": round(max_loss_per_share * quantity, 2),
            "breakeven_upper": round(short_call["strike"] + net_credit, 2),
            "breakeven_lower": round(short_put["strike"] - net_credit, 2),
            "available_call_strikes_above_short": available_call_strikes,
            "available_put_strikes_below_short": available_put_strikes,
            "note": "Educational calculation only — not a trade recommendation. "
                    "Verify prices, margin, and lot size on your broker terminal before placing any order. "
                    "Also manually check: upcoming results date, F&O ban list, and news for this stock."
        }

    result["target_delta_used"] = target_delta
    result["wing_width_pct_used"] = wing_width_pct if strategy_type == "iron_condor" else None
    result["rank_info"] = get_stock_rank(symbol.upper())
    result["all_expiries"] = data["all_expiries"]

    legs_for_margin = [{"tradingsymbol": lg["tradingsymbol"],
                         "transaction_type": "SELL" if k.startswith("sell") else "BUY"}
                        for k, lg in result["legs"].items()]
    margin_required, margin_error = compute_margin(legs_for_margin, quantity)
    result["margin_required"] = margin_required
    result["margin_error"] = margin_error
    return result


@app.route("/api/strategy/<symbol>")
def strategy(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    target_delta = float(request.args.get("target_delta", DEFAULT_TARGET_DELTA))
    wing_width_pct = float(request.args.get("wing_width_pct", DEFAULT_WING_WIDTH_PCT))
    strategy_type = request.args.get("strategy_type", "iron_condor")
    expiry_str = request.args.get("expiry")
    lots = int(request.args.get("lots", 1))
    result = build_strategy(symbol, target_delta, wing_width_pct, strategy_type, expiry_str, lots)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ---------------------------------------------------------------------------
# Watchlist / Trade Section
# ---------------------------------------------------------------------------
@app.route("/api/watchlist/add", methods=["POST"])
def watchlist_add():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol = body.get("symbol", "").upper()
    target_delta = float(body.get("target_delta", DEFAULT_TARGET_DELTA))
    wing_width_pct = float(body.get("wing_width_pct", DEFAULT_WING_WIDTH_PCT))
    strategy_type = body.get("strategy_type", "iron_condor")
    expiry_str = body.get("expiry")
    lots = int(body.get("lots", 1))

    built = build_strategy(symbol, target_delta, wing_width_pct, strategy_type, expiry_str, lots)
    if "error" in built:
        return jsonify(built), 404

    today_str = datetime.now().date().isoformat()
    position = {
        "id": f"{symbol}_{int(time.time())}",
        "symbol": symbol,
        "added_on": today_str,
        "entry_spot": built["spot"],
        "expiry": built["expiry"],
        "lot_size": built["lot_size"],
        "lots": built["lots"],
        "quantity": built["quantity"],
        "strategy_type": built["strategy_type"],
        "legs": built["legs"],
        "entry_net_credit_per_share": built["net_credit_per_share"],
        "entry_max_profit": built["max_profit"],
        "entry_max_loss": built["max_loss"],
        "entry_margin_required": built.get("margin_required"),
        "entry_margin_error": built.get("margin_error"),
        "breakeven_upper": built["breakeven_upper"],
        "breakeven_lower": built["breakeven_lower"],
        "broker_orders": [],
        "history": [{"date": today_str, "spot": built["spot"],
                     "pnl": 0.0, "current_debit_per_share": built["net_credit_per_share"]}],
    }
    positions = load_positions()
    positions.append(position)
    save_positions(positions)
    return jsonify({"ok": True, "position": position})


@app.route("/api/watchlist/<pos_id>", methods=["DELETE"])
def watchlist_remove(pos_id):
    positions = load_positions()
    positions = [p for p in positions if p["id"] != pos_id]
    save_positions(positions)
    return jsonify({"ok": True})


def mark_to_market(position):
    strategy_type = position.get("strategy_type", "iron_condor")
    leg_keys = ["sell_call", "buy_call", "sell_put", "buy_put"] if strategy_type == "iron_condor" \
        else ["sell_call", "sell_put"]
    quantity = position.get("quantity", position["lot_size"])

    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite.quote(inst_keys)

    prices, missing_legs = {}, []
    for k in leg_keys:
        key = f"NFO:{position['legs'][k]['tradingsymbol']}"
        price = extract_price(quotes.get(key))
        prices[k] = price
        if price is None:
            missing_legs.append(f"{k} ({position['legs'][k]['tradingsymbol']})")

    if missing_legs:
        return {"__error__": "No usable price for: " + ", ".join(missing_legs) +
                              ". Contract may be expired/delisted, or market closed with no resting orders."}

    if strategy_type == "iron_condor":
        current_debit_per_share = (prices["sell_call"] + prices["sell_put"]) - (prices["buy_call"] + prices["buy_put"])
    else:
        current_debit_per_share = prices["sell_call"] + prices["sell_put"]

    pnl_per_share = position["entry_net_credit_per_share"] - current_debit_per_share
    pnl = round(pnl_per_share * quantity, 2)
    current_position_value = round(current_debit_per_share * quantity, 2)

    spot, err = get_spot_price(position["symbol"])
    if err:
        return {"__error__": err["error"]}

    today = datetime.now().date()
    expiry_date = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    days_left = (expiry_date - today).days
    T_remaining = max(days_left, 0) / 365.0

    zone = "safe"
    if spot > position["breakeven_upper"] or spot < position["breakeven_lower"]:
        zone = "breached"
    elif days_left <= 2:
        zone = "near_expiry"

    probability_of_success = None
    if days_left > 0:
        call_strike = position["legs"]["sell_call"]["strike"]
        put_strike = position["legs"]["sell_put"]["strike"]
        iv_call = implied_vol(prices["sell_call"], spot, call_strike, T_remaining, "CE")
        iv_put = implied_vol(prices["sell_put"], spot, put_strike, T_remaining, "PE")
        delta_call = bs_delta(spot, call_strike, T_remaining, RISK_FREE_RATE, iv_call, "CE")
        delta_put = bs_delta(spot, put_strike, T_remaining, RISK_FREE_RATE, iv_put, "PE")
        prob_call_itm = max(0.0, min(1.0, delta_call))
        prob_put_itm = max(0.0, min(1.0, abs(delta_put)))
        probability_of_success = round(max(0.0, 1 - prob_call_itm - prob_put_itm) * 100, 1)
    else:
        probability_of_success = 100.0 if zone == "safe" else 0.0

    leg_details = {}
    for k in leg_keys:
        entry_price = position["legs"][k]["ltp"]
        current_price = prices[k]
        is_sell = k.startswith("sell")
        # sold leg profits when price falls; bought leg profits when price rises
        per_share = (entry_price - current_price) if is_sell else (current_price - entry_price)
        leg_details[k] = {
            "tradingsymbol": position["legs"][k]["tradingsymbol"],
            "strike": position["legs"][k]["strike"],
            "entry_price": entry_price, "current_price": round(current_price, 2),
            "pnl": round(per_share * quantity, 2)
        }

    return {
        "spot": spot, "pnl": pnl, "current_debit_per_share": round(current_debit_per_share, 2),
        "current_position_value": current_position_value, "legs_current": leg_details,
        "days_left": days_left, "zone": zone, "probability_of_success_pct": probability_of_success,
        "pct_of_max_profit": round((pnl / position["entry_max_profit"] * 100), 1) if position["entry_max_profit"] else None
    }


@app.route("/api/watchlist")
def watchlist():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_positions()
    today_str = datetime.now().date().isoformat()
    out, changed = [], False
    for p in positions:
        mtm = mark_to_market(p)
        if mtm and "__error__" in mtm:
            out.append({**p, "mtm_error": mtm["__error__"]})
            continue
        if not p["history"] or p["history"][-1]["date"] != today_str:
            p["history"].append({"date": today_str, "spot": mtm["spot"], "pnl": mtm["pnl"],
                                  "current_debit_per_share": mtm["current_debit_per_share"]})
            changed = True
        out.append({**p, "current": mtm})
    if changed:
        save_positions(positions)
    return jsonify({"positions": out})


# ---------------------------------------------------------------------------
# Order execution — preview (no side effects) then confirm (places real orders)
# ---------------------------------------------------------------------------
def leg_keys_for(position):
    return ["sell_call", "buy_call", "sell_put", "buy_put"] if position.get("strategy_type") == "iron_condor" \
        else ["sell_call", "sell_put"]


@app.route("/api/execute/<pos_id>/preview")
def execute_preview(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404

    quantity = position.get("quantity", position["lot_size"])
    orders = []
    for k in leg_keys_for(position):
        leg = position["legs"][k]
        txn = "SELL" if k.startswith("sell") else "BUY"
        orders.append({
            "leg": k, "tradingsymbol": leg["tradingsymbol"], "transaction_type": txn,
            "quantity": quantity, "reference_price": leg["ltp"],
        })
    return jsonify({
        "position_id": pos_id, "symbol": position["symbol"], "orders": orders,
        "default_product": "NRML", "default_order_type": "MARKET",
        "warning": "These orders are NOT yet placed. Review carefully — you can edit quantity/price or "
                   "remove a leg entirely below — then confirm to send them to your live Zerodha account. "
                   "Removing a hedge leg (a BUY order) from an Iron Condor leaves that side of the "
                   "position with unlimited-style risk, same as a naked strangle."
    })


@app.route("/api/execute/<pos_id>/confirm", methods=["POST"])
def execute_confirm(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    if not body.get("confirmed"):
        return jsonify({"error": "Confirmation flag not set — nothing was placed."}), 400

    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404

    product = body.get("product", "NRML")
    order_type = body.get("order_type", "MARKET")

    # Use the exact list of orders the user reviewed/edited in the UI if provided (each item may
    # have its own quantity and, for LIMIT orders, its own price). Falls back to the position's
    # default legs if the frontend didn't send an explicit list, for backward compatibility.
    custom_orders = body.get("orders")
    if custom_orders:
        legs_to_place = custom_orders
    else:
        default_quantity = position.get("quantity", position["lot_size"])
        legs_to_place = [{
            "leg": k, "tradingsymbol": position["legs"][k]["tradingsymbol"],
            "transaction_type": "SELL" if k.startswith("sell") else "BUY",
            "quantity": default_quantity, "price": position["legs"][k]["ltp"],
        } for k in leg_keys_for(position)]

    if not legs_to_place:
        return jsonify({"error": "No legs left to place — every leg was removed in the review screen."}), 400

    results = []
    for item in legs_to_place:
        txn_type = kite.TRANSACTION_TYPE_SELL if item["transaction_type"] == "SELL" else kite.TRANSACTION_TYPE_BUY
        quantity = int(item.get("quantity") or 1)
        try:
            kwargs = dict(
                variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                tradingsymbol=item["tradingsymbol"], transaction_type=txn_type,
                quantity=quantity, product=getattr(kite, f"PRODUCT_{product}"),
                order_type=getattr(kite, f"ORDER_TYPE_{order_type}"),
                validity=kite.VALIDITY_DAY,
            )
            if order_type == "LIMIT" and item.get("price"):
                kwargs["price"] = float(item["price"])
            order_id = kite.place_order(**kwargs)
            results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                             "transaction_type": item["transaction_type"], "quantity": quantity,
                             "status": "placed", "order_id": order_id})
        except Exception as e:
            results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                             "transaction_type": item["transaction_type"], "quantity": quantity,
                             "status": "failed", "error": str(e)})
            # Stop immediately on first failure — continuing could leave you with a partial,
            # unhedged position. Better to stop and let you check/complete manually.
            break

    positions = load_positions()
    for p in positions:
        if p["id"] == pos_id:
            p["broker_orders"] = p.get("broker_orders", []) + results
    save_positions(positions)

    any_failed = any(r["status"] == "failed" for r in results)
    placed_count = sum(1 for r in results if r["status"] == "placed")
    total_legs = len(legs_to_place)
    partial = any_failed and placed_count > 0

    return jsonify({
        "results": results,
        "partial_failure": partial,
        "note": ("PARTIAL EXECUTION: some legs placed, one failed. You may now hold an incomplete, "
                 "unhedged position. Open your Zerodha app / Kite web IMMEDIATELY to check your actual "
                 "positions and orders, and manually complete or exit as needed."
                 if partial else
                 "All legs failed — nothing was placed." if any_failed and placed_count == 0 else
                 f"All {placed_count}/{total_legs} legs placed successfully. Verify fills in your Zerodha app.")
    })


@app.route("/api/orders")
def list_orders():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        orders = kite.orders()
        return jsonify({"orders": orders})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/orders/<order_id>/cancel", methods=["POST"])
def cancel_order_route(order_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/orders/<order_id>/modify", methods=["POST"])
def modify_order_route(order_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    try:
        kwargs = {"variety": kite.VARIETY_REGULAR, "order_id": order_id}
        if body.get("quantity"):
            kwargs["quantity"] = int(body["quantity"])
        if body.get("price"):
            kwargs["price"] = float(body["price"])
        if body.get("order_type"):
            kwargs["order_type"] = getattr(kite, f"ORDER_TYPE_{body['order_type']}")
        kite.modify_order(**kwargs)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# Price chart
# ---------------------------------------------------------------------------
INTERVAL_MAX_DAYS = {
    "5minute": 30, "15minute": 30, "60minute": 90, "day": 720,
}


@app.route("/api/chart/<symbol>")
def chart(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    symbol = symbol.upper()
    interval = request.args.get("interval", "day")
    if interval not in INTERVAL_MAX_DAYS:
        return jsonify({"error": f"Unsupported interval '{interval}'. Use one of: {', '.join(INTERVAL_MAX_DAYS)}"}), 400

    if symbol in INDEX_SYMBOLS:
        _, nse = get_instruments()
        token = None
        wanted = INDEX_SYMBOLS[symbol].split(":")[1]
        for i in nse:
            if i["segment"] == "INDICES" and i["tradingsymbol"] == wanted:
                token = i["instrument_token"]
                break
        if not token:
            return jsonify({"error": f"Could not resolve chart instrument token for {symbol}"}), 404
    else:
        _, nse = get_instruments()
        nse_match = [i for i in nse if i["exchange"] == "NSE" and i["tradingsymbol"] == symbol]
        if not nse_match:
            return jsonify({"error": f"{symbol} not found on NSE"}), 404
        token = nse_match[0]["instrument_token"]

    requested_days = int(request.args.get("days", 60))
    max_days = INTERVAL_MAX_DAYS[interval]
    days = min(requested_days, max_days)
    clamped = requested_days > max_days

    to_date = datetime.now()
    from_date = to_date - timedelta(days=days + (15 if interval == "day" else 3))
    candles = kite.historical_data(token, from_date, to_date, interval)

    def fmt_date(c):
        d = c["date"]
        return d.isoformat() if hasattr(d, "isoformat") else str(d)

    return jsonify({
        "symbol": symbol, "interval": interval, "clamped_to_days": days if clamped else None,
        "candles": [{"t": fmt_date(c), "o": c["open"], "h": c["high"], "l": c["low"], "c": c["close"]} for c in candles],
    })


# ---------------------------------------------------------------------------
# News / event-risk headlines
# ---------------------------------------------------------------------------
def _fetch_rss(url, headers, verify=True, timeout=10):
    resp = requests.get(url, headers=headers, timeout=timeout, verify=verify)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:5]
    headlines = []
    for it in items:
        headlines.append({
            "title": (it.findtext("title") or "").strip(),
            "link": (it.findtext("link") or "").strip(),
            "pub_date": (it.findtext("pubDate") or "").strip(),
        })
    return headlines


@app.route("/api/news/<symbol>")
def news(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    symbol = symbol.upper()
    sources = [
        ("Google News",
         f"https://news.google.com/rss/search?q={requests.utils.quote(symbol + ' NSE share')}&hl=en-IN&gl=IN&ceid=IN:en"),
        ("Yahoo Finance",
         f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}.NS&region=IN&lang=en-IN"),
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
               "Accept": "application/rss+xml, application/xml, text/xml, */*"}
    errors = []
    ssl_issue_seen = False

    for name, url in sources:
        try:
            headlines = _fetch_rss(url, headers, verify=True)
            if headlines:
                return jsonify({"symbol": symbol, "source": name, "headlines": headlines,
                                 "note": "Best-effort headline scan, not verified analysis. "
                                         "Read the actual articles before treating this as an event-risk signal."})
            errors.append(f"{name} returned no items")
        except requests.exceptions.SSLError as e:
            ssl_issue_seen = True
            errors.append(f"{name}: SSL certificate verification failed")
            if ALLOW_INSECURE_NEWS:
                try:
                    headlines = _fetch_rss(url, headers, verify=False)
                    if headlines:
                        return jsonify({"symbol": symbol, "source": name + " (unverified TLS)", "headlines": headlines,
                                         "note": "Fetched with certificate verification disabled because "
                                                 "ALLOW_INSECURE_NEWS is set — your network is doing TLS "
                                                 "interception. This is only used for public headlines, never "
                                                 "for Kite API calls. Best-effort scan, not verified analysis."})
                except Exception as e2:
                    errors.append(f"{name} (insecure retry) also failed: {e2}")
        except Exception as e:
            errors.append(f"{name} failed: {e}")

    guidance = ""
    if ssl_issue_seen:
        guidance = (" This looks like your network (office/government firewall, antivirus, or a proxy) is "
                     "intercepting HTTPS traffic with its own certificate — common on corporate/government "
                     "networks. Kite API calls aren't affected since those go through Kite's own SDK. To fix "
                     "properly: ask your IT team for the organization's root CA certificate and set it via the "
                     "REQUESTS_CA_BUNDLE environment variable. As a quick workaround for this headlines feature "
                     "only (not recommended on untrusted networks), you can set ALLOW_INSECURE_NEWS=true as an "
                     "environment variable before running backend.py.")

    return jsonify({"symbol": symbol, "headlines": [],
                     "error": "Could not fetch news from any source. " + " | ".join(errors) + guidance})


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    if "PUT_YOUR" in API_KEY or "PUT_YOUR" in API_SECRET:
        print("!! Set KITE_API_KEY and KITE_API_SECRET (env vars, or edit backend.py) before running.")
    print(f"Set your Kite app's Redirect URL to: {REDIRECT_URL}")
    if ALLOW_INSECURE_NEWS:
        print("!! ALLOW_INSECURE_NEWS is on — news headline fetches will skip TLS verification on failure.")
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
