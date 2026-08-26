"""
Kite Option-Selling Dashboard — local backend
------------------------------------------------
Run:  python backend.py
Then open: https://algo2.wecon.in

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
- REQUIRES kiteconnect >= 5.1.1 (`pip install --upgrade kiteconnect`). Exchanges now reject MARKET/
  SL-M orders placed via the API without a market_protection value (SEBI's retail algo-trading rules);
  this file always sends one, but older SDK versions don't accept the parameter at all — see the
  try/except around kite.place_order() in place_basket_orders() for the fallback behavior.
"""

import os
import math
import time
import json
import threading
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

from flask import Flask, request, jsonify, send_from_directory, redirect

try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install kiteconnect flask numpy requests")

import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# CONFIG — fill these in from https://developers.kite.trade (your app)
# SECURITY: set these as real environment variables (or a .env file loaded before
# this process starts) — do NOT hardcode real keys/secrets directly in this file,
# especially if this file is ever shared, committed to git, or pasted anywhere.
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("KITE_API_KEY", "b4j9bna5hdew1hh4")
API_SECRET = os.environ.get("KITE_API_SECRET", "mbrdjydzd9ckisvrp4tsqbtkkgojpzue")
REDIRECT_URL = os.environ.get("REDIRECT_URL", "https://algo2.wecon.in/api/callback")

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

# --- Double Calendar Spread defaults ---
# A double calendar is: SELL a near-term call + SELL a near-term put (usually a bit OTM each side),
# and BUY a far-term call + far-term put at the SAME two strikes. It's a net-DEBIT, defined-risk
# trade that profits from the near leg decaying faster than the far leg (positive theta, long vega) —
# the "sweet spot" is the underlying sitting between the two short strikes at near expiry.
DEFAULT_CALENDAR_OTM_PCT = 0.03      # each strike this far OTM from spot, in "otm_pct" strike mode
DEFAULT_CALENDAR_TARGET_DELTA = 0.25 # used instead of otm_pct in "delta" strike mode
CALENDAR_TARGET_GAP_DAYS = 30        # preferred day-gap between near and far expiry when auto-picking
CALENDAR_CURVE_POINTS = 41           # number of spot points sampled for the payoff curve
CALENDAR_CURVE_RANGE_PCT = 0.15      # curve spans spot x (1 +/- this), i.e. +/-15% around current spot
# Exit-suggestion thresholds for tracked calendar positions (informational only, never auto-exits)
CALENDAR_STOP_LOSS_DEBIT_MULTIPLE = 0.5   # suggest exit if loss reaches this multiple of debit paid
CALENDAR_NEAR_EXPIRY_DAYS_WARNING = 3     # suggest exit/roll when this close to near-leg expiry (gamma risk)

# --- Exit / stop-loss suggestion rule (informational only — this tool never auto-exits) ---
# Trigger a suggested-exit flag when EITHER condition is met, whichever occurs first:
#   1) total position loss reaches this multiple of the premium originally received, or
#   2) either short leg's delta magnitude rises to at least this threshold.
STOP_LOSS_PREMIUM_MULTIPLE = 2.0
STOP_LOSS_DELTA_THRESHOLD = 0.35

# --- Approximate Zerodha F&O options charges (informational estimate only) ---
# These are commonly published rates as of this writing — brokerage/tax rules DO change over
# time (STT rates in particular have changed via budget announcements before). Verify current
# rates at https://zerodha.com/charges and your actual contract note before relying on this for
# anything beyond a rough planning estimate. All values are editable here.
CHARGES = {
    "brokerage_flat": 20.0,          # per executed order, or 0.03% of turnover, whichever is LOWER
    "brokerage_pct": 0.0003,
    "stt_sell_pct": 0.001,           # Securities Transaction Tax, options SELL side, on premium turnover
    "exchange_txn_pct": 0.0003503,   # NSE F&O exchange transaction charge, on premium turnover (both sides)
    "sebi_pct": 0.0000001,           # SEBI turnover fee (₹10 per crore == 0.0001%), both sides
    "gst_pct": 0.18,                 # GST on (brokerage + exchange txn charges + SEBI fee)
    "stamp_duty_buy_pct": 0.00003,   # stamp duty, BUY side only, on premium turnover
}

# --- Stock-picking screener v2: IV-rank, liquidity, ban-list, news (all best-effort) ---
# Kite has no historical-IV endpoint, so a genuine IV Rank/Percentile has to be built up by us,
# one snapshot per day, in a small local file. Until enough days have accumulated, iv_rank will
# be null and we fall back to a same-day cross-sectional IV percentile (how rich this stock's IV
# is TODAY relative to the other F&O stocks scanned today) so the field is never just empty.
# NSE/BSE trade on IST (UTC+5:30) regardless of what timezone this server's OS happens to be set
# to. Every "today", market-hours check, and historical-data window in this file needs to line up
# with the EXCHANGE's clock, not the server's -- so all of that goes through this helper instead of
# the bare now_ist(), which would silently use the server's local timezone and could otherwise
# leave charts/scans looking "stuck" a few hours behind (or ahead) if the server isn't set to IST.
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist():
    """Current wall-clock time in IST, as a naive datetime (matching what Kite's historical-data
    API expects, and what date/market-hours comparisons elsewhere in this file assume)."""
    return datetime.now(IST).replace(tzinfo=None)


IV_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "iv_history.json")
IVR_LOOKBACK_DAYS = 252
IVR_MIN_HISTORY_DAYS = 60        # genuine rank only after a meaningful history is available

# Iron-Condor-specific screening. The ATM pair is still useful as a first liquidity gate,
# but the final ranking is based on the ACTUAL four legs proposed for the condor.
MIN_ATM_TOTAL_OI = 500
MAX_ATM_SPREAD_PCT = 4.0
IC_MIN_LEG_OI = 25
IC_MAX_LEG_SPREAD_PCT = 15.0
IC_MIN_TOTAL_VOLUME = 10
IC_MIN_SHORT_OI = 200
IC_MAX_SHORT_SPREAD_PCT = 7.0
IC_MIN_SHORT_VOLUME = 10
IC_MIN_CREDIT_TO_MAX_LOSS = 0.08
IC_MIN_CUSHION_EM = 0.75
IC_CHAIN_STRIKE_RANGE_PCT = 0.30
IC_PREFERRED_DTE_LOW = 21
IC_PREFERRED_DTE_HIGH = 35
IC_MIN_DTE = 14
IC_MAX_DTE = 50
IC_DEFAULT_SHORT_DELTA_LOW = 0.15
IC_DEFAULT_SHORT_DELTA_HIGH = 0.20
IC_WING_WIDTHS_PCT = (0.02, 0.025, 0.035, 0.05, 0.07, 0.10)

# Final IC score: volatility richness 25, range quality 20, expected-move cushion 20,
# four-leg liquidity 15, trade economics 15, event risk 5.
IC_SCORE_WEIGHTS = {
    # Delta symmetry is deliberately explicit: a 0.18-delta IC should not silently
    # become a 0.25/0.18 structure merely because the latter collects more premium.
    "iv": 0.20, "range": 0.15, "cushion": 0.25,
    "liquidity": 0.10, "economics": 0.15, "delta": 0.10, "event": 0.05,
}
SCORE_WEIGHTS = {"iv_richness": 0.40, "calmness": 0.35, "liquidity": 0.25}  # legacy display

NEWS_FOR_TOP_N = 10
FO_BAN_LIST_URL = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"


def load_iv_history():
    if not os.path.exists(IV_HISTORY_FILE):
        return {}
    try:
        with open(IV_HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_iv_history(history):
    with open(IV_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def update_iv_history_and_get_rank(symbol, atm_iv_pct, history, today_str):
    """Appends today's ATM IV snapshot (idempotent per day so re-running the screener the same
    day doesn't distort the series), trims to the lookback window, and returns
    (iv_rank_pct_or_None, days_of_history)."""
    series = history.setdefault(symbol, [])
    series[:] = [pt for pt in series if pt["date"] != today_str]
    series.append({"date": today_str, "iv": atm_iv_pct})
    series.sort(key=lambda p: p["date"])
    if len(series) > IVR_LOOKBACK_DAYS:
        del series[:-IVR_LOOKBACK_DAYS]

    if len(series) < IVR_MIN_HISTORY_DAYS:
        return None, len(series)
    values = [p["iv"] for p in series]
    below_or_equal = sum(1 for v in values if v <= atm_iv_pct)
    rank_pct = 100.0 * below_or_equal / len(values)
    return round(rank_pct, 1), len(series)


def get_fo_ban_list():
    """Best-effort fetch of NSE's daily F&O ban list. NSE's site actively blocks plain
    requests without a real browser session/cookie handshake, and the URL/format can change —
    if this fails, we say so explicitly rather than silently treating everything as 'not banned'.
    Returns (set_of_symbols_or_None, error_message_or_None)."""
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/csv,*/*",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=8)  # cookie warm-up
        resp = session.get(FO_BAN_LIST_URL, headers=headers, timeout=8)
        resp.raise_for_status()
        lines = [l.strip() for l in resp.text.splitlines() if l.strip()]
        symbols = set()
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            for p in parts:
                if p.isupper() and p.isalnum() and len(p) > 1:
                    symbols.add(p)
        return symbols, None
    except Exception as e:
        return None, (f"Could not fetch NSE F&O ban list ({e}). Verify manually at "
                       f"https://www.nseindia.com/companies-listing/corporate-filings-actions "
                       f"before trading — this filter is best-effort only.")


def _quote_cache_get(keys):
    now = time.monotonic()
    out = {}
    for k in keys:
        item = _QUOTE_CACHE.get(k)
        if item and now - item["ts"] <= _QUOTE_CACHE_TTL:
            out[k] = item["quote"]
    return out


def kite_quote_bulk(keys, *, chunk_size=500, retries=1, force_refresh=False):
    """Rate-limited, cached wrapper around Kite's /quote endpoint.

    Kite supports up to 500 instruments per /quote call, but the endpoint is limited to
    1 request/sec. A 429 is followed by a cooldown before retrying. This function is
    intentionally used by the screeners so one scan cannot DOS its own API session.
    """
    global _QUOTE_LAST_AT, _QUOTE_COOLDOWN_UNTIL
    keys = list(dict.fromkeys(k for k in keys if k))
    if not keys:
        return {}
    result = {} if force_refresh else _quote_cache_get(keys)
    missing = list(keys) if force_refresh else [k for k in keys if k not in result]
    if not missing:
        return result

    for start in range(0, len(missing), max(1, min(int(chunk_size), 500))):
        chunk = missing[start:start + max(1, min(int(chunk_size), 500))]
        attempt = 0
        while True:
            with _QUOTE_LOCK:
                now = time.monotonic()
                wait_for = max(_QUOTE_LAST_AT + _QUOTE_MIN_INTERVAL - now,
                               _QUOTE_COOLDOWN_UNTIL - now, 0.0)
                if wait_for > 0:
                    time.sleep(wait_for)
                try:
                    raw = kite.quote(chunk)
                    _QUOTE_LAST_AT = time.monotonic()
                    raw = raw or {}
                    for k, q in raw.items():
                        _QUOTE_CACHE[k] = {"quote": q, "ts": time.monotonic()}
                        result[k] = q
                    break
                except Exception as e:
                    msg = str(e).lower()
                    is_429 = "too many requests" in msg or getattr(e, "code", None) == 429
                    _QUOTE_LAST_AT = time.monotonic()
                    if is_429 and attempt < retries:
                        _QUOTE_COOLDOWN_UNTIL = time.monotonic() + _QUOTE_429_COOLDOWN
                        attempt += 1
                        logger.warning("Kite quote rate limit hit; cooling down %.1fs before retry", _QUOTE_429_COOLDOWN)
                        time.sleep(_QUOTE_429_COOLDOWN)
                        continue
                    logger.warning("Kite quote batch failed (%d instruments): %s", len(chunk), e)
                    break
    return result


def seed_spot_cache_from_prices(rows):
    now = time.monotonic()
    for r in rows:
        sym = str(r.get("symbol", "")).upper()
        price = r.get("ltp") or r.get("last_close")
        if sym and price:
            _QUOTE_CACHE[f"NSE:{sym}"] = {
                "quote": {"last_price": float(price)}, "ts": now
            }


def pick_atm_contracts(nfo_opts_for_symbol, last_close, today):
    """Given this symbol's NFO-OPT instruments, last close price, and today's date, picks the
    nearest valid expiry (same MIN_DAYS_TO_EXPIRY rule as the strategy builder) and the strike
    closest to last_close. Returns (ce_tradingsymbol, pe_tradingsymbol, strike, expiry, T) or None."""
    if not nfo_opts_for_symbol:
        return None
    all_expiries = sorted({o["expiry"] for o in nfo_opts_for_symbol})
    valid = [e for e in all_expiries if (e - today).days >= MIN_DAYS_TO_EXPIRY]
    if not valid:
        return None
    expiry = valid[0]
    chain = [o for o in nfo_opts_for_symbol if o["expiry"] == expiry]
    strikes = sorted({o["strike"] for o in chain})
    if not strikes:
        return None
    atm_strike = min(strikes, key=lambda k: abs(k - last_close))
    ce = next((o for o in chain if o["strike"] == atm_strike and o["instrument_type"] == "CE"), None)
    pe = next((o for o in chain if o["strike"] == atm_strike and o["instrument_type"] == "PE"), None)
    if not ce or not pe:
        return None
    T = max((expiry - today).days, 0) / 365.0
    return ce["tradingsymbol"], pe["tradingsymbol"], atm_strike, expiry, T


def quote_spread_pct(q):
    """Bid-ask spread as % of mid, from a Kite quote's depth. None if depth unavailable."""
    if not q:
        return None
    depth = q.get("depth", {}) or {}
    buys = [b for b in depth.get("buy", []) if b.get("price", 0) > 0]
    sells = [s for s in depth.get("sell", []) if s.get("price", 0) > 0]
    if not buys or not sells:
        return None
    bid, ask = buys[0]["price"], sells[0]["price"]
    mid = (bid + ask) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid * 100.0


def get_atm_iv_and_liquidity_bulk(candidates, nfo):
    """candidates: list of {'symbol', 'last_close'}. Batches Kite quote() calls (chunks of 200
    instruments, well under Kite's per-call limit) instead of one call per stock, since fetching
    a full option chain per stock (like get_chain_for_symbol does for a single symbol) would mean
    hundreds of extra round-trips here. Returns {symbol: {atm_iv_pct, atm_oi_total, spread_pct,
    expiry}} — a symbol is omitted if its ATM contracts couldn't be resolved or quoted."""
    today = now_ist().date()
    opts_by_symbol = {}
    for o in nfo:
        if o["segment"] == "NFO-OPT":
            opts_by_symbol.setdefault(o["name"], []).append(o)

    picks = {}  # symbol -> (ce_ts, pe_ts, strike, expiry, T)
    needed_keys = []
    for c in candidates:
        sym = c["symbol"]
        pick = pick_atm_contracts(opts_by_symbol.get(sym, []), c["last_close"], today)
        if pick:
            picks[sym] = pick
            ce_ts, pe_ts, _, _, _ = pick
            needed_keys.append(f"NFO:{ce_ts}")
            needed_keys.append(f"NFO:{pe_ts}")

    quotes = kite_quote_bulk(needed_keys, chunk_size=500, retries=1)

    out = {}
    for sym, (ce_ts, pe_ts, strike, expiry, T) in picks.items():
        ce_q = quotes.get(f"NFO:{ce_ts}")
        pe_q = quotes.get(f"NFO:{pe_ts}")
        ce_ltp, pe_ltp = extract_price(ce_q), extract_price(pe_q)
        if ce_ltp is None or pe_ltp is None:
            continue
        last_close = next(c["last_close"] for c in candidates if c["symbol"] == sym)
        ce_iv = implied_vol(ce_ltp, last_close, strike, T, "CE")
        pe_iv = implied_vol(pe_ltp, last_close, strike, T, "PE")
        atm_iv_pct = (ce_iv + pe_iv) / 2 * 100
        ce_oi = (ce_q or {}).get("oi", 0) or 0
        pe_oi = (pe_q or {}).get("oi", 0) or 0
        spreads = [s for s in (quote_spread_pct(ce_q), quote_spread_pct(pe_q)) if s is not None]
        spread_pct = round(sum(spreads) / len(spreads), 2) if spreads else None
        out[sym] = {"atm_iv_pct": round(atm_iv_pct, 1), "atm_oi_total": int(ce_oi + pe_oi),
                     "atm_spread_pct": spread_pct, "atm_expiry": str(expiry)}
    return out


def _percentile_rank(value, all_values):
    """0-100, higher = higher value relative to the group. None-safe."""
    vals = [v for v in all_values if v is not None]
    if value is None or not vals:
        return 50.0  # neutral when data is missing, rather than silently zero-weighting it
    below_or_equal = sum(1 for v in vals if v <= value)
    return 100.0 * below_or_equal / len(vals)


# --- Event calendar (informational only) ---
# A hand-maintained list of known macro event dates that commonly move markets, used to warn
# against opening NEW positions right around them, and to flag existing positions that run into
# one before expiry. RBI MPC and FOMC dates below were sourced from RBI/Federal Reserve published
# calendars — always re-verify at rbi.org.in and federalreserve.gov since schedules can shift.
# Election result days and geopolitical events are NOT reliably predictable in advance and are not
# auto-populated — add them yourself via POST /api/event-calendar/add as they become known.
EVENT_CALENDAR_FILE = os.path.join(os.path.dirname(__file__), "event_calendar.json")
ENTRY_WARNING_WINDOW_DAYS = 2   # warn on new entries if an event falls within this many days
TRADE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "trade_history.json")

# Index symbols you can type directly (in addition to any F&O stock) — maps to the exact
# Kite quote key Kite uses for that index's live spot price.
INDEX_SYMBOLS = {
    "NIFTY": "NSE:NIFTY 50",
    "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE",
    "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
    "SENSEX": "BSE:SENSEX",
}

INDEX_OPTION_EXCHANGE = {"SENSEX": "BFO", "NIFTY": "NFO", "BANKNIFTY": "NFO", "FINNIFTY": "NFO", "MIDCPNIFTY": "NFO"}

OPTION_EXCHANGE_SEGMENT = {"NFO": "NFO-OPT", "BFO": "BFO-OPT"}

app = Flask(__name__, static_folder="static", static_url_path="")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kite_dashboard")
kite = KiteConnect(api_key=API_KEY)

# Zerodha Quote API protection. Quote is limited to 1 request/second and a maximum of
# 500 instruments per request. Keep one process-wide gate so Screen 1, Screen 2 and
# other quote consumers cannot accidentally burst the API.
_QUOTE_LOCK = threading.Lock()
_QUOTE_LAST_AT = 0.0
_QUOTE_COOLDOWN_UNTIL = 0.0
_QUOTE_CACHE = {}  # key -> {"quote": dict, "ts": monotonic_seconds}
_QUOTE_CACHE_TTL = float(os.environ.get("KITE_QUOTE_CACHE_TTL", "3.0"))
_QUOTE_MIN_INTERVAL = float(os.environ.get("KITE_QUOTE_MIN_INTERVAL", "1.05"))
_QUOTE_429_COOLDOWN = float(os.environ.get("KITE_QUOTE_429_COOLDOWN", "10.5"))

SESSION = {"access_token": None, "logged_in_at": None}
INSTRUMENT_CACHE = {"nfo": None, "nse": None, "bfo": None, "bse": None, "fetched_at": None}
SCREENER_CACHE = {"results": None, "fetched_at": None}
IC_SCREENER_CACHE = {"results": None, "fetched_at": None}

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
# Event calendar (informational, hand-maintained)
# ---------------------------------------------------------------------------
def load_event_calendar():
    if not os.path.exists(EVENT_CALENDAR_FILE):
        return []
    try:
        with open(EVENT_CALENDAR_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_event_calendar(events):
    with open(EVENT_CALENDAR_FILE, "w") as f:
        json.dump(events, f, indent=2)


def seed_event_calendar_if_missing():
    """Seeds a starter calendar the first time this runs. Sourced from officially published
    RBI and US Federal Reserve calendars as of this writing — verify/update at rbi.org.in and
    federalreserve.gov, since meeting schedules can shift and this list isn't auto-refreshed."""
    if os.path.exists(EVENT_CALENDAR_FILE):
        return
    events = [
        # RBI Monetary Policy Committee — FY 2026-27 schedule (published by RBI)
        {"date": "2026-08-05", "label": "RBI MPC Policy Announcement", "type": "rbi_policy", "source": "RBI FY26-27 calendar"},
        {"date": "2026-10-07", "label": "RBI MPC Policy Announcement", "type": "rbi_policy", "source": "RBI FY26-27 calendar"},
        {"date": "2026-12-04", "label": "RBI MPC Policy Announcement", "type": "rbi_policy", "source": "RBI FY26-27 calendar"},
        # US Federal Reserve FOMC — 2026 schedule (decision announced on 2nd day, ~2pm ET)
        {"date": "2026-07-29", "label": "FOMC Rate Decision", "type": "fed_policy", "source": "federalreserve.gov 2026 calendar"},
        {"date": "2026-09-16", "label": "FOMC Rate Decision", "type": "fed_policy", "source": "federalreserve.gov 2026 calendar"},
        {"date": "2026-10-28", "label": "FOMC Rate Decision", "type": "fed_policy", "source": "federalreserve.gov 2026 calendar"},
        {"date": "2026-12-09", "label": "FOMC Rate Decision", "type": "fed_policy", "source": "federalreserve.gov 2026 calendar"},
        # Union Budget — fixed Feb 1 convention in India since 2017
        {"date": "2027-02-01", "label": "Union Budget Day", "type": "budget", "source": "fixed annual convention"},
    ]
    save_event_calendar(events)


def get_upcoming_events(days_ahead=45):
    events = load_event_calendar()
    today = now_ist().date()
    upcoming = []
    for idx, e in enumerate(events):
        try:
            d = datetime.strptime(e["date"], "%Y-%m-%d").date()
        except Exception:
            continue
        days_away = (d - today).days
        if 0 <= days_away <= days_ahead:
            upcoming.append({**e, "days_away": days_away, "index": idx})
    upcoming.sort(key=lambda e: e["days_away"])
    return upcoming


def get_entry_warning():
    """Checks for any flagged event within ENTRY_WARNING_WINDOW_DAYS — used to warn (not block)
    against opening a brand-new position right around a known macro event."""
    near = [e for e in get_upcoming_events(days_ahead=ENTRY_WARNING_WINDOW_DAYS)]
    if not near:
        return None
    labels = ", ".join(f"{e['label']} ({e['date']})" for e in near)
    return (f"Heads up: {labels} within the next {ENTRY_WARNING_WINDOW_DAYS} days. Many traders avoid "
            f"opening new option-selling positions right around major policy/event days due to volatility risk. "
            f"This is informational only — the tool does not block the trade.")


def get_event_before_expiry(expiry_str):
    """For an existing tracked position — any flagged event between today and its expiry."""
    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    except Exception:
        return None
    today = now_ist().date()
    days_to_expiry = max((expiry_date - today).days, 0)
    events = get_upcoming_events(days_ahead=days_to_expiry)
    return events[0] if events else None


# ---------------------------------------------------------------------------
# Trade history (archived on full close)
# ---------------------------------------------------------------------------
def load_trade_history():
    if not os.path.exists(TRADE_HISTORY_FILE):
        return []
    try:
        with open(TRADE_HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_trade_history(history):
    with open(TRADE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


def archive_closed_position(position, close_results):
    history = load_trade_history()
    est_realized_pnl = sum(
        r.get("estimated_realized_pnl", 0) for r in close_results if r["status"] == "placed"
    )
    exit_orders_for_charges = [
        {"price": r.get("reference_price") or 0, "quantity": r.get("quantity", 0),
         "transaction_type": r.get("transaction_type", "SELL")}
        for r in close_results if r["status"] == "placed"
    ]
    exit_charges = estimate_charges(exit_orders_for_charges)
    entry_charges_total = position.get("entry_estimated_charges") or 0
    round_trip_charges = round(entry_charges_total + exit_charges["total"], 2)
    net_realized_after_charges = round(est_realized_pnl - round_trip_charges, 2)

    history.append({
        "id": position["id"], "symbol": position["symbol"], "strategy_type": position.get("strategy_type"),
        "added_on": position.get("added_on"), "closed_on": now_ist().date().isoformat(),
        "entry_max_profit": position.get("entry_max_profit"), "entry_max_loss": position.get("entry_max_loss"),
        "estimated_realized_pnl": round(est_realized_pnl, 2),
        "entry_charges": entry_charges_total, "estimated_exit_charges": exit_charges["total"],
        "estimated_round_trip_charges": round_trip_charges,
        "net_realized_pnl_after_charges": net_realized_after_charges,
        "close_orders": close_results,
        "note": "estimated_realized_pnl is based on quoted prices at close time, not confirmed fill "
                "prices — check your Zerodha contract note for the exact realized P&L and charges.",
    })
    save_trade_history(history)


# Seed the event calendar once at import time — works whether launched via
# `python backend.py` directly or imported by Gunicorn (`gunicorn backend:app`).
seed_event_calendar_if_missing()


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


# ---------------------------------------------------------------------------
# Greeks — Gamma / Theta / Vega (Delta/Price/IV are above) + portfolio aggregation, used by the
# Dynamic Delta-Neutral Adjustment Engine further down. Reuses the exact same bs_price / bs_delta /
# norm_cdf already defined above so every Greek across the whole app is priced identically.
#
# Sign convention: a SHORT leg (you sold it) contributes the NEGATIVE of the raw per-unit option
# Greek to the portfolio; a LONG leg (bought, e.g. a hedge) contributes the raw (positive) Greek.
# This is expressed by passing `quantity` already signed (negative = short, positive = long) — every
# Greek is multiplied by that signed quantity, so the sign logic lives in exactly one place.
# ---------------------------------------------------------------------------
def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _d1_d2(S, K, T, r, sigma):
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_gamma(S, K, T, r, sigma):
    """Identical for calls and puts at the same strike/expiry. Per 1-point move in the underlying."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    """Per 1 percentage point (0.01) change in IV — the convention traders actually quote."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_theta(S, K, T, r, sigma, opt_type):
    """Per calendar day (annualized theta / 365)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if opt_type == "CE":
        term2 = -r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
    return (term1 + term2) / 365.0


@dataclass
class LegGreeks:
    tradingsymbol: str
    role: str            # "sell_call" | "sell_put" | "buy_call" | "buy_put" | ...
    opt_type: str         # "CE" | "PE"
    strike: float
    quantity: int          # SIGNED: negative for short legs, positive for long legs
    ltp: float
    delta: float
    gamma: float
    theta: float
    vega: float
    mtm: float


@dataclass
class PortfolioGreeks:
    symbol: str
    spot: float
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    mtm: float = 0.0
    legs: List[LegGreeks] = field(default_factory=list)

    def as_dict(self):
        return {
            "symbol": self.symbol, "spot": self.spot,
            "net_delta": round(self.net_delta, 2), "net_gamma": round(self.net_gamma, 4),
            "net_theta": round(self.net_theta, 2), "net_vega": round(self.net_vega, 2),
            "mtm": round(self.mtm, 2),
            "legs": [vars(l) for l in self.legs],
        }


class PortfolioGreeksEngine:
    """Computes per-leg and portfolio-level Greeks from plain dicts. The caller is responsible for
    fetching live spot/LTP/IV and passing them in."""

    def __init__(self, risk_free_rate=RISK_FREE_RATE):
        self.r = risk_free_rate

    def leg_greeks(self, *, opt_type, strike, spot, T, iv, ltp, quantity, role, tradingsymbol,
                   entry_premium=None):
        delta = bs_delta(spot, strike, T, self.r, iv, opt_type) * quantity
        gamma = bs_gamma(spot, strike, T, self.r, iv) * quantity
        theta = bs_theta(spot, strike, T, self.r, iv, opt_type) * quantity
        vega = bs_vega(spot, strike, T, self.r, iv) * quantity
        mtm = 0.0
        if entry_premium is not None:
            # Short leg profits when ltp falls below entry premium; long leg profits when ltp rises.
            per_unit_pnl = (entry_premium - ltp) if quantity < 0 else (ltp - entry_premium)
            mtm = per_unit_pnl * abs(quantity)
        return LegGreeks(tradingsymbol, role, opt_type, strike, quantity, ltp,
                          round(delta, 4), round(gamma, 6), round(theta, 4), round(vega, 4),
                          round(mtm, 2))

    def portfolio_greeks(self, symbol, spot, legs):
        """legs: iterable of dicts, each with opt_type, strike, T, iv, ltp, quantity (signed), role,
        tradingsymbol, and optionally entry_premium (for MTM)."""
        pg = PortfolioGreeks(symbol=symbol, spot=spot)
        for leg in legs:
            lg = self.leg_greeks(**leg)
            pg.legs.append(lg)
            pg.net_delta += lg.delta
            pg.net_gamma += lg.gamma
            pg.net_theta += lg.theta
            pg.net_vega += lg.vega
            pg.mtm += lg.mtm
        return pg


# ---------------------------------------------------------------------------
# Risk Management — configurable gates for the Delta Neutral Adjustment Engine. Every check here is
# a pure function of (limits, current numbers) -> (allowed: bool, reason: str); nothing in this
# section places or cancels orders, it only decides whether the Adjustment Engine / Execution code
# further down is ALLOWED to act.
# ---------------------------------------------------------------------------
@dataclass
class RiskLimits:
    max_adjustments_per_day: int = 6
    max_loss_per_position: float = 15000.0        # Rs, absolute MTM loss on a single position
    max_daily_mtm_loss: float = 25000.0            # Rs, absolute MTM loss across ALL positions today
    min_premium_for_adjustment: float = 8.0        # Rs; don't roll/adjust into a leg worth less than this
    profit_targets_pct: Tuple[float, ...] = (25.0, 50.0, 70.0, 90.0)   # staged profit-booking levels
    stop_loss_pct: float = 200.0                   # % of credit received; exit if MTM loss exceeds this


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def can_adjust(self, *, adjustments_today: int, position_mtm: float, daily_mtm: float,
                    proposed_leg_premium: Optional[float] = None) -> Tuple[bool, str]:
        if adjustments_today >= self.limits.max_adjustments_per_day:
            return False, f"Max adjustments/day reached ({self.limits.max_adjustments_per_day})"
        if position_mtm <= -abs(self.limits.max_loss_per_position):
            return False, f"Position MTM loss (Rs {position_mtm:.0f}) exceeds max loss per position"
        if daily_mtm <= -abs(self.limits.max_daily_mtm_loss):
            return False, f"Daily MTM loss (Rs {daily_mtm:.0f}) exceeds max daily loss for this symbol"
        if proposed_leg_premium is not None and proposed_leg_premium < self.limits.min_premium_for_adjustment:
            return False, (f"Proposed leg premium Rs {proposed_leg_premium:.2f} is below the minimum "
                            f"Rs {self.limits.min_premium_for_adjustment} required to bother adjusting")
        return True, "OK"

    def breached_daily_loss(self, daily_mtm: float) -> bool:
        return daily_mtm <= -abs(self.limits.max_daily_mtm_loss)

    def profit_target_hit(self, credit_received: float, current_mtm: float) -> Optional[float]:
        """Returns the highest configured profit-target % that's been reached, or None. Positions
        are meant to be closed in FULL the moment any configured target is hit."""
        if credit_received <= 0:
            return None
        pct_captured = (current_mtm / credit_received) * 100.0
        hit = [t for t in sorted(self.limits.profit_targets_pct) if pct_captured >= t]
        return max(hit) if hit else None

    def stop_loss_hit(self, credit_received: float, current_mtm: float) -> bool:
        if credit_received <= 0:
            return False
        loss_pct = (-current_mtm / credit_received) * 100.0
        return loss_pct >= self.limits.stop_loss_pct


# ---------------------------------------------------------------------------
# Adjustment Engine — Scenario A/B logic and a transparent multi-factor scorer (this IS the "AI
# Recommendation Engine" from the spec, implemented as an inspectable weighted-score model rather
# than an opaque trained model, so every number that drives a decision is loggable and auditable).
#
# --- Why net delta goes NEGATIVE when the market moves UP (easy to get backwards) ---
# Selling a call is a short-delta position (lose as price rises, like being short the underlying);
# selling a put is a long-delta position (lose as price falls, like being long the underlying). In a
# roughly delta-neutral short strangle/condor, both are sized to net close to zero. If the underlying
# RISES: the short call moves closer to the money -> its delta magnitude grows -> your (negative)
# delta exposure from that leg grows more negative; the short put moves further OTM -> its delta
# shrinks toward zero -> your (positive) exposure from that leg shrinks. Both effects push net
# portfolio delta MORE NEGATIVE as spot rises (consistent with losing money as price rises = negative
# delta). The mirror is true on the way down:
#     net_delta very NEGATIVE  <=>  market has moved UP, the CALL side is under stress  (Scenario A)
#     net_delta very POSITIVE  <=>  market has moved DOWN, the PUT side is under stress (Scenario B)
# ---------------------------------------------------------------------------
@dataclass
class AdjustmentCandidate:
    action: str                  # "roll_put_up" | "roll_call_down" | "roll_call_further_otm" |
                                  # "roll_put_further_otm" | "convert_iron_fly" | "add_hedge" | "no_action"
    description: str
    legs_to_close: list = field(default_factory=list)   # leg dicts to buy back / sell to close
    legs_to_open: list = field(default_factory=list)    # leg dicts describing the new legs
    expected_delta_after: float = 0.0
    additional_premium: float = 0.0     # Rs collected (positive) or paid (negative) net of this adjustment
    margin_impact: float = 0.0          # Rs, additional margin this adjustment is expected to require
    risk_reduction_score: float = 0.0   # 0-1: how much closer to delta-neutral this gets you
    probability_of_profit: float = 0.0  # 0-1
    expected_drawdown: float = 0.0      # Rs, rough worst-case add-on risk from taking this action
    score: float = 0.0
    reasoning: str = ""


class AdjustmentEngine:
    def __init__(self, delta_threshold=10.0, gamma_threshold=None, weights=None):
        self.delta_threshold = delta_threshold
        self.gamma_threshold = gamma_threshold
        # Weighted composite score — every factor normalized to a comparable 0..1-ish scale before
        # weighting, so no single factor dominates purely because of its raw units (Rs vs a
        # probability vs a percentage). Weights are configurable from Settings.
        self.weights = weights or {
            "expected_profit": 0.30, "risk_reduction": 0.30, "additional_premium": 0.15,
            "margin_impact": 0.10, "probability_of_profit": 0.10, "expected_drawdown": 0.05,
        }

    def needs_adjustment(self, net_delta: float, net_gamma: Optional[float] = None):
        """Ignore small delta changes — only trigger on a genuine, configured threshold breach."""
        if abs(net_delta) < self.delta_threshold:
            return False, "Delta within threshold, no action needed"
        return True, f"Net delta {net_delta:+.1f} exceeds threshold +/-{self.delta_threshold}"

    def scenario(self, net_delta: float) -> str:
        """See class docstring above for the sign derivation. "up" = call side under stress (market
        has risen); "down" = put side under stress (market has fallen)."""
        return "up" if net_delta < 0 else "down"

    def generate_candidates(self, position, portfolio_greeks, candidate_fetcher: Callable) -> List[AdjustmentCandidate]:
        """Builds every viable AdjustmentCandidate for the current breach, per the priority list:
        roll the threatened short strike further away, OR roll the calmer side's short strike closer
        (collects more premium and adds offsetting delta), OR convert to an Iron Fly if that
        materially flattens delta, OR add a standalone hedge if no roll alone brings delta back in
        range. `candidate_fetcher(scenario, position, portfolio_greeks) -> list[dict]` supplies the
        actual tradable strikes/premiums (live option chain)."""
        scenario = self.scenario(portfolio_greeks.net_delta)
        raw = candidate_fetcher(scenario, position, portfolio_greeks) or []
        candidates = [AdjustmentCandidate(**c) for c in raw]
        if not candidates:
            candidates.append(AdjustmentCandidate(
                action="no_action",
                description="No viable roll/hedge candidate found within the configured strike/delta range",
                expected_delta_after=portfolio_greeks.net_delta))
        return candidates

    def _score_candidate(self, cand: AdjustmentCandidate) -> AdjustmentCandidate:
        expected_profit_norm = min(max(cand.additional_premium / 500.0, -1.0), 1.0)
        risk_reduction_norm = min(max(cand.risk_reduction_score, 0.0), 1.0)
        additional_premium_norm = min(max(cand.additional_premium / 500.0, -1.0), 1.0)
        margin_norm = 1.0 - min(max(cand.margin_impact / 50000.0, 0.0), 1.0)
        pop_norm = min(max(cand.probability_of_profit, 0.0), 1.0)
        drawdown_norm = 1.0 - min(max(cand.expected_drawdown / 20000.0, 0.0), 1.0)

        w = self.weights
        score = (w["expected_profit"] * expected_profit_norm
                 + w["risk_reduction"] * risk_reduction_norm
                 + w["additional_premium"] * additional_premium_norm
                 + w["margin_impact"] * margin_norm
                 + w["probability_of_profit"] * pop_norm
                 + w["expected_drawdown"] * drawdown_norm)
        cand.score = round(score, 4)
        cand.reasoning = (
            f"{cand.action}: expected_profit={expected_profit_norm:.2f}, risk_reduction={risk_reduction_norm:.2f}, "
            f"premium={additional_premium_norm:.2f}, margin={margin_norm:.2f}, pop={pop_norm:.2f}, "
            f"drawdown={drawdown_norm:.2f} -> composite score {cand.score}"
        )
        return cand

    def recommend(self, position, portfolio_greeks, candidate_fetcher: Callable):
        """Full pipeline: threshold check -> generate candidates -> score -> pick the best.
        Returns (recommended: AdjustmentCandidate | None, all_candidates: list, trigger_reason: str)."""
        trigger, reason = self.needs_adjustment(portfolio_greeks.net_delta, portfolio_greeks.net_gamma)
        if not trigger:
            return None, [], reason
        candidates = self.generate_candidates(position, portfolio_greeks, candidate_fetcher)
        for c in candidates:
            self._score_candidate(c)
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0] if candidates else None
        return best, candidates, reason


# ---------------------------------------------------------------------------
# Execution wrapper — thin and deliberately dependency-injected: it takes backend's OWN
# place_basket_orders function as an argument rather than calling the broker directly, so there is
# exactly one place in this file that ever calls the real broker order-placement API.
# ---------------------------------------------------------------------------
@dataclass
class ExecutionResult:
    ok: bool
    orders: list = field(default_factory=list)
    error: str = None


class AdjustmentExecutor:
    def __init__(self, place_orders_fn: Callable[[List[Dict[str, Any]], str, str], list], product="NRML"):
        self.place_orders_fn = place_orders_fn
        self.product = product

    def _build_legs(self, candidate):
        legs = []
        for leg in candidate.legs_to_close:
            legs.append({
                "leg": f"close_{leg['role']}", "tradingsymbol": leg["tradingsymbol"],
                "transaction_type": "BUY" if leg["quantity"] < 0 else "SELL",
                "quantity": abs(leg["quantity"]),
            })
        for leg in candidate.legs_to_open:
            legs.append({
                "leg": f"open_{leg['role']}", "tradingsymbol": leg["tradingsymbol"],
                "transaction_type": "SELL" if leg["role"].startswith("sell") else "BUY",
                "quantity": abs(leg["quantity"]),
            })
        return legs

    def execute(self, candidate, execution_mode="track"):
        """In "track" mode (default, matches the auto-trade engine's paper-trading pattern), no real
        order is sent — fills are simulated immediately so downstream P&L/logging behaves exactly as
        it would live."""
        legs = self._build_legs(candidate)
        if not legs:
            return ExecutionResult(ok=True, orders=[])
        if execution_mode == "track":
            simulated = [{"status": "placed", "order_id": f"TRACK-ADJ-{i}", **leg}
                         for i, leg in enumerate(legs)]
            return ExecutionResult(ok=True, orders=simulated)
        results = self.place_orders_fn(legs, self.product, "MARKET")
        failed = [r for r in results if r.get("status") != "placed"]
        return ExecutionResult(ok=not failed, orders=results,
                                error=None if not failed else f"{len(failed)} leg(s) failed to place")


# ---------------------------------------------------------------------------
# Adjustment logging — structured, append-only JSON-lines log of every adjustment considered/taken.
# ---------------------------------------------------------------------------
class AdjustmentLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self._lock = threading.Lock()
        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def log_adjustment(self, *, ts, symbol, spot, delta_before, delta_after, action,
                        premium_collected, reason, execution_mode, candidates_considered=None):
        entry = {
            "ts": ts, "symbol": symbol, "spot": spot,
            "delta_before": round(delta_before, 2), "delta_after": round(delta_after, 2),
            "action": action,
            "premium_collected": round(premium_collected, 2) if premium_collected else 0.0,
            "reason": reason, "execution_mode": execution_mode,
            "candidates_considered": candidates_considered or [],
        }
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def read_recent(self, limit=200):
        if not os.path.exists(self.log_path):
            return []
        with self._lock:
            with open(self.log_path, "r") as f:
                lines = f.readlines()
        out = []
        for line in lines[-limit:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        out.reverse()
        return out



# ---------------------------------------------------------------------------
# Delta Neutral Engine config + JSON-file persisted state — mirrors the AUTOTRADE_DEFAULTS /
# load_autotrade_state pattern already used above, so both subsystems are operated the same way
# (arm/disarm, a fixed set of configurable keys, daily counters that roll over at day-change).
# ---------------------------------------------------------------------------
DELTA_ENGINE_STATE_FILE = os.path.join(os.path.dirname(__file__), "delta_engine_state.json")
_delta_engine_state_lock = threading.Lock()

DELTA_ENGINE_DEFAULTS = {
    "enabled": False,                # armed or not — monitoring/adjustment never runs unless True
    # "track" (default, paper — no real orders) or "live". Mirrors the autotrade execution_mode
    # safety pattern exactly: NOT settable via the bulk config route, only via a dedicated ack-gated
    # route, so a stray "Save settings" click can never flip this to real orders.
    "execution_mode": "track",
    "symbols": ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"],
    "delta_threshold": 10.0,         # absolute net portfolio delta that triggers an adjustment
    "gamma_threshold": None,         # optional secondary confirmation; None = delta alone triggers
    "max_adjustments_per_day": 6,
    "max_loss_per_position": 15000.0,
    "max_daily_mtm_loss": 25000.0,
    "min_premium_for_adjustment": 8.0,
    "profit_targets_pct": [25, 50, 70, 90],
    "stop_loss_pct": 200.0,
    "hedge_distance_pct": 2.5,
    "delta_range_low": 0.15,
    "delta_range_high": 0.20,
    "expiry_selection": "nearest",   # "nearest" | "next"
    "spot_poll_seconds": 1,          # underlying spot LTP refresh cadence (the "tick" loop)
    "greeks_poll_seconds": 5,        # option-chain/premium refresh cadence (heavier call, rate-limited)
    # Every risk counter below is keyed BY SYMBOL, not pooled -- a bad day on BANKNIFTY never eats
    # into NIFTY's adjustment budget or vice versa, and each is evaluated independently.
    "adjustments_today_by_symbol": {},   # {"NIFTY": 2, "BANKNIFTY": 0, ...}
    "daily_mtm_by_symbol": {},           # {"NIFTY": -1200.0, ...}
    "paused_symbols_today": [],          # symbols whose OWN daily-loss limit tripped -- new adjustments
                                          # are skipped for just that symbol for the rest of the day;
                                          # profit-target/stop-loss closing still applies to it as normal
    "day": None,
    "last_recommendation": None,
    "last_scan_at": None,
    "last_error": None,
    "disarm_reason": None,
}

DELTA_ENGINE_CONFIGURABLE_KEYS = (
    "symbols", "delta_threshold", "gamma_threshold", "max_adjustments_per_day",
    "max_loss_per_position", "max_daily_mtm_loss", "min_premium_for_adjustment",
    "profit_targets_pct", "stop_loss_pct", "hedge_distance_pct", "delta_range_low",
    "delta_range_high", "expiry_selection", "spot_poll_seconds", "greeks_poll_seconds",
)


def dn_load_state():
    if not os.path.exists(DELTA_ENGINE_STATE_FILE):
        return dict(DELTA_ENGINE_DEFAULTS)
    try:
        with open(DELTA_ENGINE_STATE_FILE, "r") as f:
            state = json.load(f)
        merged = dict(DELTA_ENGINE_DEFAULTS)
        merged.update(state)
        return merged
    except Exception:
        return dict(DELTA_ENGINE_DEFAULTS)


def dn_save_state(state):
    with _delta_engine_state_lock:
        with open(DELTA_ENGINE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Trading-logic enhancements: IV/HV, Expected Move, Probability of Touch,
# Trend Detection (EMA/ADX/RSI), Volatility Regime. All heuristic / best-effort —
# these support decision-making, they don't replace it.
# ---------------------------------------------------------------------------
def classify_iv_hv(iv_pct, hv_pct):
    """IV/HV ratio + label. Rich IV relative to how much the stock actually moves is the
    core edge in option selling — HV alone or IV alone can both be misleading."""
    if not iv_pct or not hv_pct:
        return None
    ratio = iv_pct / hv_pct
    if ratio > 1.30:
        label = "Excellent"
    elif ratio >= 1.10:
        label = "Good"
    elif ratio >= 1.0:
        label = "Fair"
    else:
        label = "Avoid"
    return {"iv_pct": iv_pct, "hv_pct": hv_pct, "ratio": round(ratio, 2), "label": label}


def get_iv_trend_from_history(symbol, history):
    """5/10/20-trading-day IV trend from the same iv_history.json used for IV Rank."""
    series = sorted(history.get(symbol, []), key=lambda p: p["date"])
    if len(series) < 2:
        return None
    today_iv = series[-1]["iv"]

    def n_ago(n):
        idx = len(series) - 1 - n
        return series[idx]["iv"] if idx >= 0 else None

    iv_5, iv_10, iv_20 = n_ago(5), n_ago(10), n_ago(20)
    trend = "Stable"
    if iv_5 is not None:
        if today_iv > iv_5 * 1.05:
            trend = "Rising"
        elif today_iv < iv_5 * 0.95:
            trend = "Falling"
    return {"iv_now": today_iv, "iv_5d_ago": iv_5, "iv_10d_ago": iv_10, "iv_20d_ago": iv_20, "trend": trend}


def expected_move(spot, atm_iv_pct, days_to_expiry):
    """Expected Move = Spot x IV x sqrt(DTE/365). The standard 1-sigma range option sellers use
    to decide whether a strike has enough of a cushion."""
    if spot is None or atm_iv_pct is None or days_to_expiry is None:
        return None
    T = max(days_to_expiry, 0) / 365.0
    em = spot * (atm_iv_pct / 100.0) * math.sqrt(T)
    return {"expected_move": round(em, 2), "expected_move_pct": round(em / spot * 100, 2) if spot else None,
            "upper": round(spot + em, 2), "lower": round(spot - em, 2)}


def probability_of_touch(delta):
    """Standard trading-desk approximation: POT is roughly 2x the delta of the strike (since
    touching the strike at any point is roughly twice as likely as finishing beyond it at expiry)."""
    if delta is None:
        return None
    return round(min(100.0, abs(delta) * 2 * 100), 1)


def _ema(values, period):
    if len(values) < period:
        return None
    ema = float(np.mean(values[:period]))
    alpha = 2.0 / (period + 1)
    for v in values[period:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - 100 / (1 + rs))


def _rma(values, period):
    """Wilder's smoothed moving average, used by ADX."""
    if len(values) < period:
        return np.array([])
    rma = np.zeros(len(values) - period + 1)
    rma[0] = np.mean(values[:period])
    alpha = 1.0 / period
    for i in range(1, len(rma)):
        rma[i] = rma[i - 1] + alpha * (values[period - 1 + i] - rma[i - 1])
    return rma


def _adx(highs, lows, closes, period=14):
    if len(closes) < period * 2 + 1:
        return None
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr_rma, plus_rma, minus_rma = _rma(tr, period), _rma(plus_dm, period), _rma(minus_dm, period)
    n = min(len(atr_rma), len(plus_rma), len(minus_rma))
    if n == 0:
        return None
    atr_safe = np.where(atr_rma[-n:] == 0, 1e-9, atr_rma[-n:])
    plus_di = 100 * plus_rma[-n:] / atr_safe
    minus_di = 100 * minus_rma[-n:] / atr_safe
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-9, (plus_di + minus_di))
    if len(dx) < period:
        return float(np.mean(dx))
    adx_series = _rma(dx, period)
    return float(adx_series[-1]) if len(adx_series) else float(np.mean(dx))


def resolve_token_for_symbol(symbol):
    """Shared instrument-token lookup for stocks AND indices, including BSE SENSEX."""
    symbol = symbol.upper()
    if symbol in INDEX_SYMBOLS:
        wanted = INDEX_SYMBOLS[symbol].split(":")[1]
        exchange = INDEX_SYMBOLS[symbol].split(":")[0]
        instruments = get_bse_instruments() if exchange == "BSE" else get_instruments()[1]
        for i in instruments:
            if i.get("segment") == "INDICES" and i.get("tradingsymbol") == wanted:
                return i["instrument_token"], None
        return None, f"Could not resolve index token for {symbol}"
    _, nse = get_instruments()
    matches = [i for i in nse if i["exchange"] == "NSE" and i["tradingsymbol"] == symbol]
    if not matches:
        return None, f"{symbol} not found on NSE"
    return matches[0]["instrument_token"], None


def classify_trend_regime(ema20, ema50, ema100, adx, rsi):
    trending = adx is not None and adx >= 25
    bullish_stack = ema50 is not None and ema20 > ema50 and (ema100 is None or ema50 > ema100)
    bearish_stack = ema50 is not None and ema20 < ema50 and (ema100 is None or ema50 < ema100)
    if trending and bullish_stack and rsi is not None and rsi > 55:
        return "Strong Uptrend", True
    if trending and bearish_stack and rsi is not None and rsi < 45:
        return "Strong Downtrend", True
    if adx is not None and adx < 20 and rsi is not None and 40 <= rsi <= 60:
        return "Range Bound", False
    if adx is not None and 20 <= adx < 25:
        return "Transitioning", False
    return "Volatile / Mixed", False


def get_trend_regime(symbol):
    """EMA20/50/100 + ADX14 + RSI14 off ~220 days of daily candles. Premium selling (Iron
    Condor/Strangle) works best in Range Bound markets — strong trends should generally be
    avoided or handled with directional spreads instead."""
    token, err = resolve_token_for_symbol(symbol)
    if err:
        return {"error": err}
    to_date = now_ist()
    from_date = to_date - timedelta(days=220)
    try:
        candles = kite.historical_data(token, from_date, to_date, "day")
    except Exception as e:
        return {"error": f"Historical data fetch failed: {e}"}
    if len(candles) < 30:
        return {"error": "Not enough historical data to classify trend (need 30+ trading days)"}
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    ema20, ema50, ema100 = _ema(closes, 20), _ema(closes, 50), _ema(closes, 100)
    adx, rsi = _adx(highs, lows, closes, 14), _rsi(closes, 14)
    if ema20 is None or adx is None or rsi is None:
        return {"error": "Not enough historical data for a reliable trend read"}
    regime, avoid_selling = classify_trend_regime(ema20, ema50, ema100, adx, rsi)
    return {"symbol": symbol.upper(), "ema20": round(ema20, 2),
            "ema50": round(ema50, 2) if ema50 is not None else None,
            "ema100": round(ema100, 2) if ema100 is not None else None,
            "adx14": round(adx, 1), "rsi14": round(rsi, 1),
            "regime": regime, "avoid_premium_selling": avoid_selling,
            "note": "Heuristic (EMA slope + ADX strength + RSI), not a guaranteed signal."}


def get_india_vix():
    try:
        q = kite_quote_bulk(["NSE:INDIA VIX"])["NSE:INDIA VIX"]
        return q["last_price"], None
    except Exception as e:
        return None, str(e)


def classify_volatility_regime(vix, iv_rank_pct):
    """Commonly-cited India VIX bands. Thresholds are approximate conventions, not a rule
    from any exchange — re-check against current market context."""
    if vix is None:
        return {"label": "Unknown", "recommendation": "Suitable",
                "note": "India VIX unavailable right now; regime not classified."}
    if vix < 12:
        label = "Low Volatility"
    elif vix < 18:
        label = "Normal"
    elif vix < 25:
        label = "High Volatility"
    else:
        label = "Extreme"
    if label == "Low Volatility":
        rec = "Reduce Size" if (iv_rank_pct is not None and iv_rank_pct < 30) else "Suitable"
    elif label == "Normal":
        rec = "Suitable"
    elif label == "High Volatility":
        rec = "Reduce Size"
    else:
        rec = "Avoid"
    return {"label": label, "india_vix": round(vix, 2), "recommendation": rec}


def suggest_strategy_family(iv_rank_pct, trend):
    """Simple rule table: strong trend -> directional spread; otherwise pick the non-directional
    structure that fits the current IV regime."""
    trend_label = trend.get("regime") if trend and not trend.get("error") else None
    if trend_label in ("Strong Uptrend", "Strong Downtrend"):
        base = "Bull Put Spread (directional credit spread)" if trend_label == "Strong Uptrend" \
            else "Bear Call Spread (directional credit spread)"
        return {"suggested": base,
                "reason": f"{trend_label} detected — avoid non-directional premium selling "
                          f"(Iron Condor/Strangle) into a strong trend."}
    if iv_rank_pct is None:
        return {"suggested": None,
                "reason": "IV rank unavailable (run the Screener first) — can't classify IV regime yet."}
    if iv_rank_pct >= 70:
        return {"suggested": "Short Strangle or Iron Fly",
                "reason": f"IV rank {iv_rank_pct}% is high — rich premium supports a more aggressive structure."}
    if iv_rank_pct >= 40:
        return {"suggested": "Iron Condor",
                "reason": f"IV rank {iv_rank_pct}% is moderate — a defined-risk Iron Condor is the standard fit."}
    return {"suggested": "Single-side Credit Spread, or skip",
            "reason": f"IV rank {iv_rank_pct}% is low — premium is thin here."}


def recommended_position_size(capital, risk_pct, max_loss_per_lot):
    if not capital or not risk_pct or not max_loss_per_lot or max_loss_per_lot <= 0:
        return None
    max_risk_amount = capital * risk_pct / 100.0
    lots = int(max_risk_amount // max_loss_per_lot)
    return {"max_risk_amount": round(max_risk_amount, 2), "recommended_lots": max(lots, 0)}


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


def quote_stats(q):
    """Return best bid/ask, mid, spread %, top-level quantity and quote volume from a Kite quote."""
    if not q:
        return {"bid": None, "ask": None, "mid": None, "spread_pct": None, "bid_qty": 0, "ask_qty": 0,
                "volume": 0, "oi": 0}
    depth = q.get("depth", {}) or {}
    buys = [x for x in depth.get("buy", []) if x.get("price", 0) > 0]
    sells = [x for x in depth.get("sell", []) if x.get("price", 0) > 0]
    bid = buys[0]["price"] if buys else None
    ask = sells[0]["price"] if sells else None
    mid = (bid + ask) / 2 if bid is not None and ask is not None else extract_price(q)
    spread = ((ask - bid) / mid * 100) if bid is not None and ask is not None and mid else None
    return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread,
            "bid_qty": int(buys[0].get("quantity", 0)) if buys else 0,
            "ask_qty": int(sells[0].get("quantity", 0)) if sells else 0,
            "volume": int(q.get("volume", 0) or 0), "oi": int(q.get("oi", 0) or 0)}


def option_quality(o, q):
    st = quote_stats(q)
    return {**o, "ltp": extract_price(q), "bid": st["bid"], "ask": st["ask"],
            "mid": st["mid"], "spread_pct": round(st["spread_pct"], 2) if st["spread_pct"] is not None else None,
            "volume": st["volume"], "oi": st["oi"], "bid_qty": st["bid_qty"], "ask_qty": st["ask_qty"]}


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def model_expiry_probability_between(spot, lower, upper, iv_pct, days):
    """Approximate risk-neutral probability of expiry spot between two prices using lognormal BS distribution."""
    if not all(v is not None for v in (spot, lower, upper, iv_pct, days)) or spot <= 0 or lower <= 0 or upper <= lower or iv_pct <= 0:
        return None
    T = max(days, 1) / 365.0
    sigma = iv_pct / 100.0
    vol = sigma * math.sqrt(T)
    mu = math.log(spot) + (RISK_FREE_RATE - 0.5 * sigma * sigma) * T
    if vol <= 0:
        return None
    z_lo = (math.log(lower) - mu) / vol
    z_hi = (math.log(upper) - mu) / vol
    return round(max(0.0, min(1.0, normal_cdf(z_hi) - normal_cdf(z_lo))) * 100, 1)


def score_band(value, bands):
    for threshold, score in bands:
        if value <= threshold:
            return score
    return bands[-1][1]


def ic_event_risk(symbol, expiry_str, headlines=None):
    """Only use data actually available to this application: seeded/manual event calendar + headline scan.
    Zerodha does not provide an earnings calendar through Kite Connect, so this is deliberately not presented
    as an earnings-date guarantee."""
    flags = []
    expiry_event = get_event_before_expiry(expiry_str)
    if expiry_event:
        flags.append({"type": "calendar", "label": expiry_event.get("label"), "date": expiry_event.get("date")})
    risky_words = ("result", "earnings", "dividend", "bonus", "split", "merger", "demerger", "acquisition",
                   "buyback", "order", "regulatory", "court", "approval", "rating", "downgrade", "upgrade")
    for h in headlines or []:
        title = (h.get("title") or "").lower()
        if any(w in title for w in risky_words):
            flags.append({"type": "headline", "label": h.get("title"), "date": h.get("pub_date")})
    score = 100
    if any(x["type"] == "calendar" for x in flags): score -= 50
    if any(x["type"] == "headline" for x in flags): score -= 25
    return max(0, score), flags


def build_ic_candidate_from_chain(symbol, spot, expiry, chain, target_delta=0.18, wing_width_pct=None,
                                   lots=1, force_symmetric=False, call_delta=None, put_delta=None):
    """Build an IC from already-quoted chain data. Optimises liquidity/economics while preserving
    the user's target delta; this is the core used by both the screener and strategy builder."""
    calls = sorted([o for o in chain if o["instrument_type"] == "CE" and o.get("delta") is not None], key=lambda x: x["strike"])
    puts = sorted([o for o in chain if o["instrument_type"] == "PE" and o.get("delta") is not None], key=lambda x: x["strike"])
    if not calls or not puts: return None
    ct = float(call_delta if call_delta is not None else target_delta)
    pt = float(put_delta if put_delta is not None else target_delta)
    # Keep the requested short-delta meaning intact.  In normal mode the two short
    # legs are searched within +/-0.03 delta of the requested target.  This prevents
    # a 0.25-delta call from replacing a 0.18-delta call simply because its premium is
    # larger.  Any permitted asymmetry is then explicitly scored below.
    delta_tolerance = 0.03
    call_candidates = [o for o in calls if max(0.10,ct-delta_tolerance) <= abs(o["delta"]) <= min(0.30,ct+delta_tolerance)]
    put_candidates = [o for o in puts if max(0.10,pt-delta_tolerance) <= abs(o["delta"]) <= min(0.30,pt+delta_tolerance)]
    if not call_candidates: call_candidates = [min(calls, key=lambda o: abs(abs(o["delta"]) - ct))]
    if not put_candidates: put_candidates = [min(puts, key=lambda o: abs(abs(o["delta"]) - pt))]
    best = None
    widths = [wing_width_pct] if wing_width_pct else list(IC_WING_WIDTHS_PCT)
    for sc in call_candidates:
        for sp in put_candidates:
            for w in widths:
                ca = [o for o in calls if o["strike"] > sc["strike"]]
                pb = [o for o in puts if o["strike"] < sp["strike"]]
                if not ca or not pb: continue
                lc = min(ca, key=lambda o: abs(o["strike"] - sc["strike"] * (1 + w)))
                lp = min(pb, key=lambda o: abs(o["strike"] - sp["strike"] * (1 - w)))
                if any(o.get("mid") is None for o in (sc, sp, lc, lp)): continue
                credit = (sc["mid"] + sp["mid"]) - (lc["mid"] + lp["mid"])
                cw, pw = lc["strike"] - sc["strike"], sp["strike"] - lp["strike"]
                max_loss = max(cw, pw) - credit
                if credit <= 0 or max_loss <= 0: continue
                dte = max((expiry - now_ist().date()).days, 1)
                atm_iv = np.mean([o.get("iv", 0) for o in chain if abs(o["strike"]-spot) == min(abs(x["strike"]-spot) for x in chain)])
                em = expected_move(spot, atm_iv, dte)
                if not em: continue
                ce_cushion = (sc["strike"] - spot) / em["expected_move"] if em["expected_move"] else 0
                pe_cushion = (spot - sp["strike"]) / em["expected_move"] if em["expected_move"] else 0
                be_low, be_high = sp["strike"] - credit, sc["strike"] + credit
                pop = model_expiry_probability_between(spot, be_low, be_high, atm_iv, dte)
                leg_spreads = [o.get("spread_pct") for o in (sc, sp, lc, lp) if o.get("spread_pct") is not None]
                liq_score = sum(min(100, max(0, 100 - x * 10)) for x in leg_spreads) / len(leg_spreads) if leg_spreads else 0
                oi_score = sum(min(100, math.log10(max(o.get("oi",0),1)) / 4 * 100) for o in (sc,sp,lc,lp)) / 4
                liquidity = 0.65 * liq_score + 0.35 * oi_score
                econ = min(100, max(0, (credit / max_loss) / 0.5 * 100))
                dte_score = 100 if IC_PREFERRED_DTE_LOW <= dte <= IC_PREFERRED_DTE_HIGH else max(40, 100 - abs(dte - 28) * 3)

                # Risk-adjust the premium advantage.  More premium is useful only if
                # it is accompanied by enough safety cushion.  A closer short strike
                # therefore has to earn its extra premium rather than winning on raw
                # credit alone.
                min_cushion = min(ce_cushion, pe_cushion)
                cushion_component = score_band(min_cushion, [(0.75,20),(0.90,35),(1.00,50),(1.15,70),(1.30,85),(1.50,95),(999,100)])
                delta_gap = abs(abs(sc.get("delta", 0)) - abs(sp.get("delta", 0)))
                delta_target_error = (abs(abs(sc.get("delta", 0)) - ct) + abs(abs(sp.get("delta", 0)) - pt)) / 2.0
                delta_symmetry_score = max(0, 100 - (delta_gap / 0.04) * 100)
                delta_target_score = max(0, 100 - (delta_target_error / 0.03) * 100)
                delta_score = 0.65 * delta_symmetry_score + 0.35 * delta_target_score
                # Effective economics: credit/max-loss is discounted when cushion is poor.
                risk_adjusted_econ = econ * (0.45 + 0.55 * cushion_component / 100.0)
                score = (0.25 * delta_score + 0.30 * cushion_component + 0.20 * risk_adjusted_econ +
                         0.15 * liquidity + 0.10 * dte_score)
                candidate = {"sell_call": sc, "buy_call": lc, "sell_put": sp, "buy_put": lp,
                             "credit": credit, "max_loss": max_loss, "call_wing": cw, "put_wing": pw,
                             "expected_move": em, "ce_cushion": ce_cushion, "pe_cushion": pe_cushion,
                             "probability_of_profit": pop, "liquidity_score": liquidity,
                             "economics_score": econ, "risk_adjusted_economics_score": risk_adjusted_econ,
                             "dte_score": dte_score, "delta_gap": delta_gap,
                             "delta_target_error": delta_target_error, "delta_symmetry_score": delta_score,
                             "selection_score": score, "atm_iv": atm_iv}
                if best is None or candidate["selection_score"] > best["selection_score"]: best = candidate
    return best


def fetch_chain_quotes_for_expiry(symbol, expiry, opts, spot_override=None, quotes_override=None):
    """Build one quoted option chain. If quotes_override is supplied, NO REST quote request is
    made here; this is what lets the IC screener batch thousands of option instruments into a
    small number of Kite /quote calls instead of making one request per stock/expiry."""
    keys = [f"NFO:{o['tradingsymbol']}" for o in opts]
    quotes = quotes_override if quotes_override is not None else kite_quote_bulk(keys, chunk_size=500, retries=1)
    spot, err = (spot_override, None) if spot_override else get_spot_price(symbol)
    if err: return None, err
    T = max((expiry-now_ist().date()).days, 0) / 365.0
    chain=[]
    for o in opts:
        q=quotes.get(f"NFO:{o['tradingsymbol']}")
        mid=quote_stats(q).get("mid")
        if mid is None: continue
        iv=implied_vol(mid, spot, o['strike'], T, o['instrument_type'])
        if iv is None or not math.isfinite(iv) or iv <= 0:
            continue
        chain.append(option_quality({**o, "iv": round(iv*100,1), "delta": round(bs_delta(spot,o['strike'],T,RISK_FREE_RATE,iv,o['instrument_type']),3)}, q))
    return {"spot":spot,"T":T,"chain":chain,"lot_size":opts[0]["lot_size"] if opts else None}, None



def extract_bid_ask(quote):
    """Return Zerodha best bid and best ask from live market depth."""
    if not quote:
        return None, None
    depth = quote.get("depth", {}) or {}
    buys = [b for b in (depth.get("buy") or []) if float(b.get("price") or 0) > 0]
    sells = [s for s in (depth.get("sell") or []) if float(s.get("price") or 0) > 0]
    bid = float(buys[0]["price"]) if buys else None
    ask = float(sells[0]["price"]) if sells else None
    return bid, ask


def recommended_limit_price(transaction_type, bid, ask, ltp=None):
    """Marketable LIMIT price: BUY uses best Ask; SELL uses best Bid."""
    if transaction_type == "BUY":
        return ask if ask is not None else ltp
    return bid if bid is not None else ltp


def refresh_execution_quotes(position):
    """Fetch fresh LTP/Bid/Ask and recommended LIMIT prices for all entry legs."""
    quantity = position.get("quantity", position["lot_size"])
    leg_keys = leg_keys_for(position)
    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite_quote_bulk(inst_keys, force_refresh=True)

    orders = []
    for k in leg_keys:
        leg = position["legs"][k]
        txn = "SELL" if k.startswith("sell") else "BUY"
        q = quotes.get(f"NFO:{leg['tradingsymbol']}") or {}
        ltp = q.get("last_price")
        bid, ask = extract_bid_ask(q)
        auto_price = recommended_limit_price(txn, bid, ask, ltp)
        orders.append({
            "leg": k,
            "tradingsymbol": leg["tradingsymbol"],
            "transaction_type": txn,
            "quantity": quantity,
            "ltp": ltp,
            "bid": bid,
            "ask": ask,
            "recommended_limit_price": auto_price,
            "reference_price": auto_price,
            "price_source": (
                "BID" if txn == "SELL" and bid is not None else
                "ASK" if txn == "BUY" and ask is not None else
                "LTP_FALLBACK"
            ),
        })
    return orders

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


def estimate_charges(orders):
    """orders: list of {'price': float, 'quantity': int, 'transaction_type': 'BUY'/'SELL'}.
    Returns an approximate total charges figure (brokerage + STT + exchange fee + SEBI fee +
    GST + stamp duty) for placing this exact basket as ONE side of a trade (i.e. call this once
    for entry orders, and again separately for exit orders, to get a full round-trip estimate).
    This is a planning estimate only — always verify against your actual Kite contract note."""
    total_brokerage = total_stt = total_exchange = total_sebi = total_stamp = 0.0
    for o in orders:
        turnover = float(o["price"]) * int(o["quantity"])
        if turnover <= 0:
            continue
        brokerage = min(CHARGES["brokerage_flat"], CHARGES["brokerage_pct"] * turnover)
        exchange_txn = CHARGES["exchange_txn_pct"] * turnover
        sebi = CHARGES["sebi_pct"] * turnover
        total_brokerage += brokerage
        total_exchange += exchange_txn
        total_sebi += sebi
        if o["transaction_type"] == "SELL":
            total_stt += CHARGES["stt_sell_pct"] * turnover
        else:
            total_stamp += CHARGES["stamp_duty_buy_pct"] * turnover

    gst = CHARGES["gst_pct"] * (total_brokerage + total_exchange + total_sebi)
    total = total_brokerage + total_stt + total_exchange + total_sebi + gst + total_stamp

    return {
        "brokerage": round(total_brokerage, 2), "stt": round(total_stt, 2),
        "exchange_txn_charges": round(total_exchange, 2), "sebi_fee": round(total_sebi, 2),
        "gst": round(gst, 2), "stamp_duty": round(total_stamp, 2), "total": round(total, 2),
    }


# ---------------------------------------------------------------------------
# Kite login flow
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Historical breakout diagnostics
# ---------------------------------------------------------------------------
def breakout_confirmation_diagnostics(c):
    """Six historical index-side confirmations; option momentum is excluded."""
    return {
        "breakout": bool(c.get("breakout")),
        "vwap": bool(c.get("vwap")),
        "ema": bool(c.get("ema")),
        "rsi": bool(c.get("rsi")),
        "volume": bool(c.get("volume")),
        "sr": bool(c.get("sr") or c.get("resistance_broken") or c.get("support_broken")),
        "strong_close": bool(c.get("strong_close", True)),
    }

def diagnostic_score(c):
    d = breakout_confirmation_diagnostics(c)
    return sum(d[k] for k in ("breakout","vwap","ema","rsi","volume","sr")), d

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
    SESSION["logged_in_at"] = now_ist().isoformat()
    kite.set_access_token(SESSION["access_token"])
    return redirect("/")


@app.route("/api/session-status")
def session_status():
    """Actively validates the token (not just checks it's present) so a stale/expired token
    can't keep showing 'Connected' after it's no longer valid — this is the fix for the tool
    showing 'connected' even when Zerodha has actually invalidated the session."""
    if not SESSION["access_token"]:
        return jsonify({"logged_in": False, "logged_in_at": None})
    try:
        kite.set_access_token(SESSION["access_token"])
        kite.profile()  # cheap call just to confirm the token still actually works
        return jsonify({"logged_in": True, "logged_in_at": SESSION["logged_in_at"]})
    except TokenException:
        SESSION["access_token"] = None
        SESSION["logged_in_at"] = None
        return jsonify({"logged_in": False, "logged_in_at": None, "session_expired": True})
    except Exception:
        # network hiccup or similar — don't log the user out for a transient error,
        # just report what we last knew
        return jsonify({"logged_in": True, "logged_in_at": SESSION["logged_in_at"],
                         "warning": "Could not verify token freshness right now (network issue?)."})


def require_session():
    if not SESSION["access_token"]:
        return False
    kite.set_access_token(SESSION["access_token"])
    return True


@app.route("/api/logout", methods=["POST"])
def logout():
    """Clears the stored session so the dashboard stops using this token — does NOT invalidate
    the token on Zerodha's side (Kite has no logout API), it just makes this app forget it."""
    SESSION["access_token"] = None
    SESSION["logged_in_at"] = None
    logger.info("User logged out — session cleared.")
    return jsonify({"ok": True})


@app.errorhandler(TokenException)
def handle_token_exception(e):
    """Catches an expired/invalid token from ANY route (whichever endpoint happened to hit
    Zerodha with a stale token), clears the stored session, and tells the frontend to show the
    login button again — instead of a generic 500 error or a UI that silently keeps showing
    'Connected' while every data call quietly fails."""
    SESSION["access_token"] = None
    SESSION["logged_in_at"] = None
    logger.warning("TokenException caught — clearing session and asking frontend to reconnect.")
    return jsonify({"error": "session_expired", "session_expired": True,
                     "message": "Your Zerodha session has expired. Please reconnect."}), 401


@app.errorhandler(Exception)
def handle_any_exception(e):
    """Safety net: ANY unhandled exception anywhere in the app returns valid JSON instead of an
    HTML error page. Without this, a bug in one route (e.g. a new feature touching old saved
    data) crashes with a raw 500 HTML page, which breaks every frontend '.json()' call with a
    confusing 'SyntaxError: string did not match expected pattern' instead of a clear message."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e  # let normal HTTP errors (404, 405, etc.) behave as Flask normally would
    logger.exception("Unhandled exception on %s %s", request.method, request.path)
    return jsonify({"error": "internal_error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Instrument cache
# ---------------------------------------------------------------------------
def get_bfo_instruments(force=False):
    now = now_ist()
    if force or INSTRUMENT_CACHE.get("bfo") is None or INSTRUMENT_CACHE.get("fetched_at") is None or now - INSTRUMENT_CACHE["fetched_at"] > timedelta(hours=6):
        INSTRUMENT_CACHE["bfo"] = kite.instruments("BFO")
    return INSTRUMENT_CACHE["bfo"]


def get_bse_instruments(force=False):
    now = now_ist()
    if force or INSTRUMENT_CACHE.get("bse") is None or INSTRUMENT_CACHE.get("fetched_at") is None or now - INSTRUMENT_CACHE["fetched_at"] > timedelta(hours=6):
        INSTRUMENT_CACHE["bse"] = kite.instruments("BSE")
    return INSTRUMENT_CACHE["bse"]


def option_exchange_for_symbol(symbol):
    return INDEX_OPTION_EXCHANGE.get(symbol.upper(), "NFO")


def option_segment_for_symbol(symbol):
    return OPTION_EXCHANGE_SEGMENT.get(option_exchange_for_symbol(symbol), "NFO-OPT")


def get_option_instruments_for_symbol(symbol):
    exchange = option_exchange_for_symbol(symbol)
    if exchange == "BFO":
        instruments = get_bfo_instruments()
    else:
        instruments, _ = get_instruments()
    return exchange, [i for i in instruments if i.get("name") == symbol.upper() and i.get("segment") == option_segment_for_symbol(symbol)]


def get_instruments(force=False):
    now = now_ist()
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
    IC_SCREENER_CACHE["results"] = None
    IC_SCREENER_CACHE["fetched_at"] = None
    return jsonify({"ok": True, "message": "Instrument cache cleared. Re-run the screener to refresh rankings."})


# ---------------------------------------------------------------------------
# Event calendar — avoid-new-entry warnings and existing-position event flags
# ---------------------------------------------------------------------------
@app.route("/api/event-calendar")
def event_calendar_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    days_ahead = int(request.args.get("days", 45))
    events = get_upcoming_events(days_ahead)
    return jsonify({"events": events,
                     "note": "Hand-maintained calendar (RBI/Fed dates from published sources — re-verify at "
                             "rbi.org.in and federalreserve.gov). Election result days and geopolitical events "
                             "are not predictable in advance and are not auto-tracked — add them yourself below "
                             "as they become known."})


@app.route("/api/event-calendar/add", methods=["POST"])
def event_calendar_add():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    date_str = body.get("date")
    if not date_str:
        return jsonify({"error": "date is required (YYYY-MM-DD)"}), 400
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format, expected YYYY-MM-DD"}), 400
    events = load_event_calendar()
    events.append({"date": date_str, "label": body.get("label", "Custom event"),
                    "type": body.get("type", "custom"), "source": "user-added"})
    save_event_calendar(events)
    return jsonify({"ok": True})


@app.route("/api/event-calendar/<int:index>", methods=["DELETE"])
def event_calendar_delete(index):
    events = load_event_calendar()
    if 0 <= index < len(events):
        events.pop(index)
        save_event_calendar(events)
        return jsonify({"ok": True})
    return jsonify({"error": "Invalid event index"}), 400


# ---------------------------------------------------------------------------
# Volatility screener
# ---------------------------------------------------------------------------
def historical_vol_and_atr(nse_token, days=60):
    to_date = now_ist()
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


@app.route("/api/fo-universe")
def fo_universe_route():
    """Indices + the real, current list of individual F&O stocks -- used by the Auto Trade tab's
    universe picker so you're selecting symbols that actually have tradable options, instead of
    typing a symbol blind and having it silently fail to scan."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        stocks = fo_stock_universe()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"indices": list(INDEX_SYMBOLS.keys()), "stocks": stocks})




@app.route("/api/screener")
def screener():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    limit = int(request.args.get("limit", 25))
    force = request.args.get("force", "false").lower() == "true"
    include_news = request.args.get("news", "true").lower() == "true"
    today = now_ist().date()
    today_str = str(today)

    universe = fo_stock_universe(force=force)
    nfo, nse = get_instruments(force=force)
    symbol_to_token = {i["tradingsymbol"]: i["instrument_token"] for i in nse if i["exchange"] == "NSE"}

    # --- Pass 1: calmness (HV/ATR from daily closes) — same as before ---
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

    # Seed shared spot cache from the already-computed historical close. This prevents
    # the later option-chain scan from making one NSE Quote request per stock.
    seed_spot_cache_from_prices(results)
    # --- Pass 2: ATM IV + liquidity, batched across all candidates at once ---
    candidates = [{"symbol": r["symbol"], "last_close": r["ltp"]} for r in results]
    try:
        iv_liquidity = get_atm_iv_and_liquidity_bulk(candidates, nfo)
    except Exception as e:
        iv_liquidity = {}
        logger.warning(f"ATM IV/liquidity batch fetch failed: {e}")

    iv_history = load_iv_history()
    for r in results:
        info = iv_liquidity.get(r["symbol"])
        if not info:
            r["atm_iv_pct"] = None
            r["atm_oi_total"] = None
            r["atm_spread_pct"] = None
            r["iv_rank_pct"] = None
            r["iv_rank_history_days"] = 0
            r["liquidity_ok"] = False
            continue
        r.update(info)
        rank_pct, hist_days = update_iv_history_and_get_rank(r["symbol"], info["atm_iv_pct"], iv_history, today_str)
        r["iv_rank_pct"] = rank_pct
        r["iv_rank_history_days"] = hist_days
        r["liquidity_ok"] = (info["atm_oi_total"] >= MIN_ATM_TOTAL_OI and
                              info["atm_spread_pct"] is not None and info["atm_spread_pct"] <= MAX_ATM_SPREAD_PCT)
        r["iv_hv"] = classify_iv_hv(r.get("atm_iv_pct"), r.get("hv_annualized_pct"))
    save_iv_history(iv_history)

    # --- Pass 3: best-effort F&O ban list, excludes banned symbols from top picks ---
    ban_symbols, ban_error = get_fo_ban_list()
    for r in results:
        r["fo_banned_today"] = (ban_symbols is not None and r["symbol"] in ban_symbols)

    # --- Composite score: rich IV (same-day cross-sectional percentile, since most stocks
    # won't have 20+ days of stored history yet) + calm underlying + tradeable liquidity ---
    all_iv = [r["atm_iv_pct"] for r in results]
    all_hv = [r["hv_annualized_pct"] for r in results]
    all_atr = [r["atr_pct_of_price"] for r in results]
    all_oi = [r["atm_oi_total"] for r in results if r["atm_oi_total"]]
    all_spread = [r["atm_spread_pct"] for r in results if r["atm_spread_pct"] is not None]

    eligible = []
    for r in results:
        iv_richness_pct = (r["iv_rank_pct"] if r["iv_rank_pct"] is not None
                            else _percentile_rank(r["atm_iv_pct"], all_iv))
        calm_hv_pct = 100 - _percentile_rank(r["hv_annualized_pct"], all_hv)
        calm_atr_pct = 100 - _percentile_rank(r["atr_pct_of_price"], all_atr)
        calmness_pct = (calm_hv_pct + calm_atr_pct) / 2
        oi_pct = _percentile_rank(r["atm_oi_total"], all_oi)
        spread_pct_rank = 100 - _percentile_rank(r["atm_spread_pct"], all_spread)
        liquidity_pct = (oi_pct + spread_pct_rank) / 2

        composite = (SCORE_WEIGHTS["iv_richness"] * iv_richness_pct +
                     SCORE_WEIGHTS["calmness"] * calmness_pct +
                     SCORE_WEIGHTS["liquidity"] * liquidity_pct)
        r["iv_richness_score"] = round(iv_richness_pct, 1)
        r["calmness_score"] = round(calmness_pct, 1)
        r["liquidity_score"] = round(liquidity_pct, 1)
        r["composite_score"] = round(composite, 1)
        if r["liquidity_ok"] and not r["fo_banned_today"] and r["atm_iv_pct"] is not None:
            eligible.append(r)

    eligible.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, r in enumerate(eligible):
        r["rank"] = i + 1
        r["total"] = len(eligible)

    # Everything else (illiquid, banned, or IV/liquidity data unavailable) still gets returned
    # further down the list so nothing silently disappears, just clearly marked as excluded.
    excluded = [r for r in results if r not in eligible]
    for r in excluded:
        r["rank"] = None
        r["total"] = len(eligible)

    top = eligible[:limit]

    # --- Pass 4: headlines, ONLY for the final top-N being shown, to keep this fast ---
    if include_news:
        for r in top[:NEWS_FOR_TOP_N]:
            r["headlines"], r["headlines_error"] = _get_headlines_best_effort(r["symbol"])
    for r in top[:NEWS_FOR_TOP_N]:
        r["iv_trend"] = get_iv_trend_from_history(r["symbol"], iv_history)

    SCREENER_CACHE["results"] = eligible + excluded
    SCREENER_CACHE["fetched_at"] = now_ist()

    return jsonify({
        "count": len(results), "eligible_count": len(eligible), "stocks": top,
        "excluded_sample": excluded[:10],
        "ban_list_note": ban_error if ban_error else "F&O ban list fetched OK — banned symbols excluded above.",
        "note": ("Ranked by a composite score for OPTION-SELLING: IV richness (real IV Rank once "
                 "20+ days of history accumulate in iv_history.json, cross-sectional IV percentile "
                 "until then) 40%, calmness (inverse HV+ATR) 35%, ATM liquidity (OI + spread) 25%. "
                 f"Stocks are excluded from ranking if ATM combined OI < {MIN_ATM_TOTAL_OI} lots, "
                 f"ATM spread > {MAX_ATM_SPREAD_PCT}%, on today's F&O ban list, or IV couldn't be "
                 "computed. Headlines are a best-effort keyword scan, NOT sentiment analysis or "
                 "verified news — read the actual articles, and still check earnings/corporate "
                 "action dates yourself before trading."),
    })


def get_stock_rank(symbol):
    if not SCREENER_CACHE["results"]:
        return None
    for r in SCREENER_CACHE["results"]:
        if r["symbol"] == symbol:
            return {"rank": r["rank"], "total": r["total"],
                     "hv_annualized_pct": r["hv_annualized_pct"], "atr_pct_of_price": r["atr_pct_of_price"],
                     "atm_iv_pct": r.get("atm_iv_pct"), "iv_rank_pct": r.get("iv_rank_pct"),
                     "composite_score": r.get("composite_score"), "liquidity_ok": r.get("liquidity_ok"),
                     "fo_banned_today": r.get("fo_banned_today"),
                     "screener_age_minutes": round((now_ist() - SCREENER_CACHE["fetched_at"]).total_seconds() / 60, 1)}
    return None

@app.route("/api/ic-screener")
def ic_screener():
    """Fast, production-safe IC screener.

    Screen 1 remains the full F&O stock ranking. Screen 2 deliberately uses Screen 1's ranked
    universe as its shortlist and then batches ALL option instruments needed for that shortlist
    into Kite's bulk /quote endpoint. This preserves actual four-leg Zerodha tradability without
    turning one browser click into hundreds of sequential REST requests and a 504 timeout.
    """
    if not require_session():
        return jsonify({"error":"not_logged_in"}), 401

    limit=max(1,min(int(request.args.get("limit",25)),50))
    force=request.args.get("force","false").lower()=="true"
    include_news=request.args.get("news","false").lower()=="true"
    target_delta=float(request.args.get("target_delta",DEFAULT_TARGET_DELTA))
    min_dte=int(request.args.get("min_dte",IC_MIN_DTE))
    max_dte=int(request.args.get("max_dte",IC_MAX_DTE))
    max_symbols=max(10,min(int(request.args.get("max_symbols",30)),50))
    max_expiries=max(1,min(int(request.args.get("max_expiries",2)),3))

    if min_dte < 1 or max_dte < min_dte:
        return jsonify({"error":"Invalid DTE range"}),400

    nfo,nse=get_instruments(force=force)
    today=now_ist().date()
    opts_by_sym={}
    for o in nfo:
        if o.get("segment")=="NFO-OPT":
            opts_by_sym.setdefault(o.get("name"),[]).append(o)

    # Prefer Screen 1's already-computed ranking. This is both faster and economically cleaner:
    # Screen 1 answers which stocks deserve attention; Screen 2 answers which actual IC structure
    # among those stocks is best.
    cached=SCREENER_CACHE.get("results") or []
    ranked=[r for r in cached if r.get("rank") is not None and not r.get("fo_banned_today")]
    ranked.sort(key=lambda x:x.get("composite_score",0), reverse=True)

    if ranked:
        shortlist=ranked[:max_symbols]
        shortlist_source="Screen 1 ranked universe"
    else:
        # Safe fallback if the user has not run Screen 1 yet: use a small deterministic F&O
        # shortlist rather than starting another 199-stock historical/IV scan inside this request.
        universe=fo_stock_universe(force=force)
        token_map={i["tradingsymbol"]:i["instrument_token"] for i in nse if i.get("exchange")=="NSE"}
        shortlist=[]
        for sym in universe:
            if sym in token_map and opts_by_sym.get(sym):
                shortlist.append({"symbol":sym,"ltp":None,"hv_annualized_pct":None,"atr_pct_of_price":None,
                                  "atm_iv_pct":None,"iv_rank_pct":None,"composite_score":0,"rank":None,
                                  "fo_banned_today":False})
            if len(shortlist)>=max_symbols:
                break
        shortlist_source="fallback F&O shortlist — run Screen 1 first for ranked candidates"

    seed=[]
    for r in shortlist:
        if r.get("ltp"):
            seed.append({"symbol":r["symbol"],"ltp":r["ltp"]})
    seed_spot_cache_from_prices(seed)

    # Choose up to max_expiries per stock, prioritising the user's preferred 21–35 DTE window
    # and then the closest remaining expiries to the midpoint. This gives meaningful expiry
    # diversity while keeping the REST quote workload bounded.
    selected=[]
    option_keys=[]
    selection_meta={}
    for r in shortlist:
        sym=r["symbol"]
        spot_ref=float(r.get("ltp") or 0)
        opts=opts_by_sym.get(sym,[])
        exps=sorted({o["expiry"] for o in opts if min_dte <= (o["expiry"]-today).days <= max_dte})
        if not exps:
            continue
        preferred=[e for e in exps if IC_PREFERRED_DTE_LOW <= (e-today).days <= IC_PREFERRED_DTE_HIGH]
        preferred.sort(key=lambda e: abs((e-today).days-28))
        remaining=[e for e in exps if e not in preferred]
        remaining.sort(key=lambda e: abs((e-today).days-28))
        chosen=(preferred+remaining)[:max_expiries]
        for exp in chosen:
            subset=[o for o in opts if o["expiry"]==exp and o.get("strike",0)>0 and
                    (not spot_ref or abs(float(o["strike"])-spot_ref)/spot_ref <= IC_CHAIN_STRIKE_RANGE_PCT)]
            if not subset:
                continue
            key=(sym,exp)
            selected.append((r,exp,subset))
            selection_meta[key]=r
            option_keys.extend(f"NFO:{o['tradingsymbol']}" for o in subset)

    # One/bounded set of bulk requests for the whole scan, instead of one request per expiry.
    # Kite supports up to 500 instruments per /quote request; kite_quote_bulk enforces the
    # account-wide 1 req/sec limit and caches results.
    all_quotes=kite_quote_bulk(option_keys,chunk_size=500,retries=1)

    evaluated=[]
    evaluation_errors=[]
    best_by_symbol={}
    for r,exp,subset in selected:
        sym=r["symbol"]
        try:
            spot=float(r.get("ltp") or 0)
            data,err=fetch_chain_quotes_for_expiry(sym,exp,subset,spot_override=spot,quotes_override=all_quotes)
            if err:
                evaluation_errors.append({"symbol":sym,"expiry":str(exp),"error":err})
                continue
            if not data or len(data.get("chain",[]))<4:
                evaluation_errors.append({"symbol":sym,"expiry":str(exp),"error":"Insufficient quoted option legs"})
                continue
            ic=build_ic_candidate_from_chain(sym,data["spot"],exp,data["chain"],target_delta=target_delta,lots=1)
            if not ic:
                continue
            dte=(exp-today).days
            candidate={**ic,"expiry":str(exp),"dte":dte,"lot_size":data["lot_size"]}
            prev=best_by_symbol.get(sym)
            if prev is None or candidate["selection_score"]>prev["selection_score"]:
                best_by_symbol[sym]=(r,candidate)
        except Exception as e:
            evaluation_errors.append({"symbol":sym,"expiry":str(exp),"error":str(e)})
            logger.exception("IC screener failed for %s %s",sym,exp)

    for sym,(r,best) in best_by_symbol.items():
        try:
            sc,lc,sp,lp=[best[k] for k in ("sell_call","buy_call","sell_put","buy_put")]
            short_legs=[sc,sp]; hedge_legs=[lc,lp]; all_legs=[sc,lc,sp,lp]
            spreads=[x.get("spread_pct") for x in all_legs if x.get("spread_pct") is not None]
            short_spreads=[x.get("spread_pct") for x in short_legs if x.get("spread_pct") is not None]
            hedge_spreads=[x.get("spread_pct") for x in hedge_legs if x.get("spread_pct") is not None]
            short_oi=[x.get("oi",0) for x in short_legs]
            hedge_oi=[x.get("oi",0) for x in hedge_legs]
            short_vol=[x.get("volume",0) for x in short_legs]
            total_vol=sum(x.get("volume",0) for x in all_legs)
            short_liq_ok=(min(short_oi)>=IC_MIN_SHORT_OI and
                          (not short_spreads or max(short_spreads)<=IC_MAX_SHORT_SPREAD_PCT) and
                          min(short_vol)>=IC_MIN_SHORT_VOLUME)
            hedge_liq_ok=(min(hedge_oi)>=IC_MIN_LEG_OI and
                          (not hedge_spreads or max(hedge_spreads)<=IC_MAX_LEG_SPREAD_PCT))
            leg_liq_ok=short_liq_ok and hedge_liq_ok and total_vol>=IC_MIN_TOTAL_VOLUME

            trend=get_trend_regime(sym)
            trend_ok=not trend.get("error")
            if trend_ok:
                reg=trend.get("regime","")
                range_score=95 if reg=="Range Bound" else 75 if reg in ("Transitioning","Volatile / Mixed") else 30
                if trend.get("avoid_premium_selling"): range_score=15
            else: range_score=50

            iv_score=(r.get("iv_rank_pct") if r.get("iv_rank_pct") is not None else r.get("composite_score",50))
            ivhv=(r.get("iv_hv") or {}).get("ratio")
            if ivhv is not None:
                iv_score=0.55*iv_score+0.45*min(100,max(0,(ivhv-0.8)/0.8*100))
            cushion=min(best["ce_cushion"],best["pe_cushion"])
            cushion_score=score_band(cushion,[(0.75,20),(0.90,35),(1.00,50),(1.15,70),(1.30,85),(1.50,95),(999,100)])
            # Delta symmetry is a first-class IC criterion.  This is what prevents a
            # richer 0.25-delta call from being presented as an ordinary 0.18-delta IC.
            short_delta_gap=abs(abs(sc.get("delta",0))-abs(sp.get("delta",0)))
            delta_score=float(best.get("delta_symmetry_score", max(0,100-(short_delta_gap/0.04)*100)))
            credit_ratio=best["credit"]/best["max_loss"] if best["max_loss"] else 0
            econ_score=score_band(credit_ratio,[(0.08,20),(0.10,30),(0.15,50),(0.20,65),(0.25,78),(0.35,92),(0.50,100),(999,100)])
            def spread_score(vals,scale):
                return sum(min(100,max(0,100-(v/scale)*100)) for v in vals)/len(vals) if vals else 60
            short_spread_score=spread_score(short_spreads,IC_MAX_SHORT_SPREAD_PCT)
            hedge_spread_score=spread_score(hedge_spreads,IC_MAX_LEG_SPREAD_PCT)
            short_oi_score=sum(min(100,math.log10(max(x,1))/4*100) for x in short_oi)/len(short_oi)
            hedge_oi_score=sum(min(100,math.log10(max(x,1))/4*100) for x in hedge_oi)/len(hedge_oi)
            liquidity_score=0.55*(0.65*short_spread_score+0.35*short_oi_score)+0.45*(0.65*hedge_spread_score+0.35*hedge_oi_score)
            if not short_liq_ok: liquidity_score=min(liquidity_score,45)
            elif not hedge_liq_ok: liquidity_score=min(liquidity_score,65)
            event_score,flags=ic_event_risk(sym,best["expiry"])
            final=(IC_SCORE_WEIGHTS["iv"]*iv_score+IC_SCORE_WEIGHTS["range"]*range_score+
                   IC_SCORE_WEIGHTS["cushion"]*cushion_score+IC_SCORE_WEIGHTS["liquidity"]*liquidity_score+
                   IC_SCORE_WEIGHTS["economics"]*econ_score+IC_SCORE_WEIGHTS["delta"]*delta_score+
                   IC_SCORE_WEIGHTS["event"]*event_score)
            hard_reasons=[]
            if not short_liq_ok: hard_reasons.append("short-leg liquidity failed")
            if not hedge_liq_ok: hard_reasons.append("hedge-leg liquidity weak")
            if total_vol<IC_MIN_TOTAL_VOLUME: hard_reasons.append("very low four-leg volume")
            if credit_ratio<0.08: hard_reasons.append("very low credit/max-loss")
            if cushion<0.75: hard_reasons.append("short strike inside 0.75x expected move")
            if short_delta_gap>0.04: hard_reasons.append("short-call/put delta asymmetry exceeds 0.04")
            if best.get("delta_target_error",0)>0.03: hard_reasons.append("short legs are too far from requested target delta")
            if best["credit"]<=0 or best["max_loss"]<=0: hard_reasons.append("invalid risk/reward")
            out=dict(r)
            out.update({"ic_score":round(final,1),"ic_label":"Excellent" if final>=80 else "Good" if final>=65 else "Average" if final>=50 else "Watch",
                        "ic_expiry":best["expiry"],"ic_dte":best["dte"],"ic_credit_per_share":round(best["credit"],2),
                        "ic_max_loss_per_share":round(best["max_loss"],2),"ic_credit_max_loss_pct":round(credit_ratio*100,1),
                        "ic_pop_pct":best["probability_of_profit"],"ic_ce_cushion_em":round(best["ce_cushion"],2),
                        "ic_pe_cushion_em":round(best["pe_cushion"],2),
                        "ic_sell_call_delta":round(float(sc.get("delta",0)),3),"ic_sell_put_delta":round(float(sp.get("delta",0)),3),
                        "ic_buy_call_delta":round(float(lc.get("delta",0)),3),"ic_buy_put_delta":round(float(lp.get("delta",0)),3),
                        "ic_delta_gap":round(short_delta_gap,3),"ic_delta_score":round(delta_score,1),
                        "ic_target_delta":round(target_delta,3),
                        "ic_risk_adjusted_economics_score":round(best.get("risk_adjusted_economics_score",econ_score),1),
                        "ic_liquidity_score":round(liquidity_score,1),
                        "ic_min_leg_oi":min(x.get("oi",0) for x in all_legs),"ic_min_short_oi":min(short_oi),
                        "ic_min_hedge_oi":min(hedge_oi),"ic_total_leg_volume":total_vol,
                        "ic_max_leg_spread_pct":round(max(spreads),2) if spreads else None,
                        "ic_max_short_spread_pct":round(max(short_spreads),2) if short_spreads else None,
                        "ic_max_hedge_spread_pct":round(max(hedge_spreads),2) if hedge_spreads else None,
                        "ic_short_liquidity_ok":short_liq_ok,"ic_hedge_liquidity_ok":hedge_liq_ok,
                        "trend_regime":trend.get("regime") if trend_ok else None,"trend_adx":trend.get("adx14") if trend_ok else None,
                        "event_score":event_score,"event_flags":flags,"hard_reasons":hard_reasons,
                        "ic_legs":{"sell_call":best["sell_call"]["strike"],"buy_call":best["buy_call"]["strike"],
                                   "sell_put":best["sell_put"]["strike"],"buy_put":best["buy_put"]["strike"]}})
            evaluated.append(out)
        except Exception as e:
            evaluation_errors.append({"symbol":sym,"error":str(e)})
            logger.exception("IC scoring failed for %s",sym)

    eligible=[r for r in evaluated if not r["hard_reasons"]]
    eligible.sort(key=lambda x:x["ic_score"],reverse=True)
    for i,r in enumerate(eligible,1):
        r["rank"]=i; r["total"]=len(eligible)
    excluded=[r for r in evaluated if r not in eligible]
    for r in excluded:
        r["rank"]=None; r["total"]=len(eligible)
    top=eligible[:limit]
    if include_news:
        for r in top[:NEWS_FOR_TOP_N]:
            r["headlines"],r["headlines_error"]=_get_headlines_best_effort(r["symbol"])
            r["event_score"],r["event_flags"]=ic_event_risk(r["symbol"],r["ic_expiry"],r.get("headlines"))
    for r in top:
        r["iv_trend"]=get_iv_trend_from_history(r["symbol"],load_iv_history())

    IC_SCREENER_CACHE["results"]=eligible+excluded
    IC_SCREENER_CACHE["fetched_at"]=now_ist()
    IC_SCREENER_CACHE["errors"]=evaluation_errors
    return jsonify({"count":len(shortlist),"eligible_count":len(eligible),"stocks":top,
                    "excluded_sample":excluded[:10],"deep_evaluated":len(shortlist),
                    "evaluation_error_count":len(evaluation_errors),"evaluation_errors_sample":evaluation_errors[:10],
                    "shortlist_source":shortlist_source,"option_quote_instruments":len(set(option_keys)),
                    "config":{"target_delta":target_delta,"min_dte":min_dte,"max_dte":max_dte,
                              "preferred_dte":[IC_PREFERRED_DTE_LOW,IC_PREFERRED_DTE_HIGH],"max_symbols":max_symbols,
                              "max_expiries_per_symbol":max_expiries,"weights":IC_SCORE_WEIGHTS,
                              "delta_tolerance":0.03,"max_short_delta_gap":0.04,
                              "short_leg_min_oi":IC_MIN_SHORT_OI,"short_leg_max_spread_pct":IC_MAX_SHORT_SPREAD_PCT,
                              "hedge_min_oi":IC_MIN_LEG_OI,"hedge_max_spread_pct":IC_MAX_LEG_SPREAD_PCT},
                    "note":"Screen 1 ranks the F&O universe; this screen uses the top ranked stocks, then batches the required option-chain instruments into Kite quote requests. It evaluates up to two preferred expiries per stock by default, prioritising 21–35 DTE, to remain within Zerodha REST limits and avoid browser 504 timeouts. The result is an actual four-leg Zerodha-tradable IC heuristic. Normal mode keeps both short legs within +/-0.03 of the requested delta and rejects >0.04 CE/PE delta asymmetry; expected-move cushion risk-adjusts the premium/economics score. It is not a backtest or guarantee."})


@app.route("/api/screener-health")
def screener_health():
    if not require_session():
        return jsonify({"error":"not_logged_in"}), 401
    now = time.monotonic()
    return jsonify({"stock_screener_cache":bool(SCREENER_CACHE.get("results")),
                    "ic_screener_cache":bool(IC_SCREENER_CACHE.get("results")),
                    "ic_error_count":len(IC_SCREENER_CACHE.get("errors",[])),
                    "quote_cache_size":len(_QUOTE_CACHE),
                    "quote_cooldown_seconds":round(max(0.0, _QUOTE_COOLDOWN_UNTIL-now),2),
                    "quote_min_interval":_QUOTE_MIN_INTERVAL,
                    "quote_cache_ttl":_QUOTE_CACHE_TTL})


# ---------------------------------------------------------------------------
# Expiry list + option chain (supports stocks AND indices, any expiry you pick)
# ---------------------------------------------------------------------------
def get_spot_price(symbol):
    """Returns (spot_price, error_dict_or_None). Uses the shared quote cache first and
    only falls back to a rate-limited single Quote call when no cached value exists."""
    symbol = symbol.upper()
    key = INDEX_SYMBOLS.get(symbol) if symbol in INDEX_SYMBOLS else f"NSE:{symbol}"
    cached = _quote_cache_get([key]).get(key)
    if cached and cached.get("last_price") is not None:
        return cached["last_price"], None
    try:
        quote = kite_quote_bulk([key], chunk_size=500, retries=1).get(key)
        if quote and quote.get("last_price") is not None:
            return quote["last_price"], None
        return None, {"error": f"No live quote returned for {symbol}"}
    except Exception as e:
        return None, {"error": f"Could not fetch {symbol} quote: {e}"}


@app.route("/api/expiries/<symbol>")
def expiries(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    symbol = symbol.upper()
    exchange, opts = get_option_instruments_for_symbol(symbol)
    if not opts:
        return jsonify({"error": f"No options found for {symbol}"}), 404
    today = now_ist().date()
    exp_list = sorted({o["expiry"] for o in opts if (o["expiry"] - today).days >= 1})
    return jsonify({"symbol": symbol, "expiries": [str(e) for e in exp_list]})


def get_chain_for_symbol(symbol, expiry_str=None):
    """Returns (data_dict, None) or (None, error_dict). Supports NFO indices and BFO SENSEX."""
    symbol = symbol.upper()
    spot, err = get_spot_price(symbol)
    if err:
        return None, err

    exchange, opts = get_option_instruments_for_symbol(symbol)
    if not opts:
        return None, {"error": f"No options found for {symbol}"}

    today = now_ist().date()
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
    quotes = kite_quote_bulk(inst_keys)

    enriched = []
    for o in chain:
        key = f"NFO:{o['tradingsymbol']}"
        q = quotes.get(key)
        st = quote_stats(q)
        ltp = st["mid"] if st["mid"] is not None else extract_price(q)
        if ltp is None:
            continue
        iv = implied_vol(ltp, spot, o["strike"], T, o["instrument_type"])
        delta = bs_delta(spot, o["strike"], T, RISK_FREE_RATE, iv, o["instrument_type"])
        enriched.append({**o, "ltp": ltp, "bid": st["bid"], "ask": st["ask"], "mid": st["mid"],
                         "spread_pct": round(st["spread_pct"],2) if st["spread_pct"] is not None else None,
                         "volume": st["volume"], "oi": st["oi"], "iv": round(iv * 100, 1), "delta": round(delta, 3)})

    return {"symbol": symbol, "exchange": exchange, "spot": spot, "expiry": expiry, "T": T, "lot_size": lot_size, "chain": enriched,
            "all_expiries": [str(e) for e in all_expiries]}, None


def compute_pcr_and_max_pain(chain):
    """Put/Call Ratio (by OI) and Max Pain strike, computed across the FULL fetched chain
    (not just the strikes shown in the UI's +/-25% window) so both are based on complete OI."""
    calls = [o for o in chain if o["instrument_type"] == "CE"]
    puts = [o for o in chain if o["instrument_type"] == "PE"]
    total_call_oi = sum(o["oi"] for o in calls)
    total_put_oi = sum(o["oi"] for o in puts)
    pcr = round(total_put_oi / total_call_oi, 2) if total_call_oi else None

    strikes = sorted({o["strike"] for o in chain})
    max_pain_strike, min_pain = None, None
    for k in strikes:
        pain = 0.0
        for o in calls:
            if k > o["strike"]:
                pain += (k - o["strike"]) * o["oi"]
        for o in puts:
            if k < o["strike"]:
                pain += (o["strike"] - k) * o["oi"]
        if min_pain is None or pain < min_pain:
            min_pain, max_pain_strike = pain, k

    return {"pcr": pcr, "total_call_oi": int(total_call_oi), "total_put_oi": int(total_put_oi),
            "max_pain_strike": max_pain_strike,
            "note": "PCR > 1 is traditionally read as bullish/support-building, < 1 as bearish — "
                    "a rough sentiment gauge, not a price target. Max Pain is the strike where option "
                    "writers' aggregate payout is lowest at expiry; a commonly-watched but unreliable-alone "
                    "expiry-pinning heuristic."}


@app.route("/api/optionchain/<symbol>")
def option_chain(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    expiry_str = request.args.get("expiry")
    data, err = get_chain_for_symbol(symbol, expiry_str)
    if err:
        return jsonify(err), 404

    oi_summary = compute_pcr_and_max_pain(data["chain"])
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
        "all_expiries": data["all_expiries"], "oi_summary": oi_summary,
        "calls": [slim(o) for o in calls], "puts": [slim(o) for o in puts]
    })


# ---------------------------------------------------------------------------
# Strategy builder — Iron Condor or Naked Strangle, adjustable delta/wing/lots/expiry
# ---------------------------------------------------------------------------
def build_strategy(symbol, target_delta=DEFAULT_TARGET_DELTA, wing_width_pct=DEFAULT_WING_WIDTH_PCT,
                    strategy_type="iron_condor", expiry_str=None, lots=1, put_delta=None, call_delta=None,
                    wing_mode="auto"):
    symbol=symbol.upper(); data,err=get_chain_for_symbol(symbol,expiry_str)
    if err: return err
    spot,expiry,T,lot_size,chain=data["spot"],data["expiry"],data["T"],data["lot_size"],data["chain"]
    today=now_ist().date(); quantity=lot_size*max(1,int(lots))
    calls=sorted([e for e in chain if e["instrument_type"]=="CE"],key=lambda x:x["strike"])
    puts=sorted([e for e in chain if e["instrument_type"]=="PE"],key=lambda x:x["strike"])
    def closest(options,target,sign): return min(options,key=lambda o:abs(o["delta"]-sign*target)) if options else None
    cd=float(call_delta if call_delta is not None else target_delta); pd=float(put_delta if put_delta is not None else target_delta)
    short_call=closest(calls,cd,1); short_put=closest(puts,pd,-1)
    if not short_call or not short_put: return {"error":"Could not find suitable short strikes"}
    def leg(o): return {"strike":o["strike"],"ltp":o["ltp"],"mid":o.get("mid",o["ltp"]),"bid":o.get("bid"),"ask":o.get("ask"),"delta":o["delta"],"iv":o.get("iv"),"oi":o.get("oi"),"volume":o.get("volume"),"spread_pct":o.get("spread_pct"),"tradingsymbol":o["tradingsymbol"]}
    if strategy_type=="naked_strangle":
        net_credit=short_call["mid"]+short_put["mid"]
        result={"symbol":symbol,"spot":spot,"expiry":str(expiry),"days_to_expiry":(expiry-today).days,"lot_size":lot_size,"lots":lots,"quantity":quantity,
                "strategy_type":"naked_strangle","legs":{"sell_call":leg(short_call),"sell_put":leg(short_put)},"net_credit_per_share":round(net_credit,2),
                "max_profit":round(net_credit*quantity,2),"max_loss":None,"breakeven_upper":round(short_call["strike"]+net_credit,2),"breakeven_lower":round(short_put["strike"]-net_credit,2)}
    else:
        # Use the actual quoted chain and optimise wing width unless the user explicitly selected a fixed percentage.
        width=float(wing_width_pct) if wing_mode=="fixed" else None
        ic=build_ic_candidate_from_chain(symbol,spot,expiry,chain,target_delta=target_delta,wing_width_pct=width,lots=lots,call_delta=cd,put_delta=pd)
        if not ic: return {"error":"Could not construct a positive-credit, liquid Iron Condor from the current chain"}
        # Respect independently requested call/put deltas when supplied by selecting nearest valid strikes, then re-optimise wings around them.
        sc,lc,sp,lp=ic["sell_call"],ic["buy_call"],ic["sell_put"],ic["buy_put"]
        net_credit=ic["credit"]; max_loss=ic["max_loss"]
        result={"symbol":symbol,"spot":spot,"expiry":str(expiry),"days_to_expiry":(expiry-today).days,"lot_size":lot_size,"lots":lots,"quantity":quantity,
                "strategy_type":"iron_condor","legs":{"sell_call":leg(sc),"buy_call":leg(lc),"sell_put":leg(sp),"buy_put":leg(lp)},
                "net_credit_per_share":round(net_credit,2),"max_profit":round(net_credit*quantity,2),"max_loss":round(max_loss*quantity,2),
                "breakeven_upper":round(sc["strike"]+net_credit,2),"breakeven_lower":round(sp["strike"]-net_credit,2),
                "call_wing":round(ic["call_wing"],2),"put_wing":round(ic["put_wing"],2),"wing_width_pct_used":wing_width_pct if width else None,
                "ic_selection_score":round(ic["selection_score"],1)}
    result["target_delta_used"]=target_delta; result["call_delta_used"]=cd; result["put_delta_used"]=pd; result["wing_mode"]=wing_mode
    result["rank_info"]=get_stock_rank(symbol); result["all_expiries"]=data["all_expiries"]
    legs_for_margin=[{"tradingsymbol":lg["tradingsymbol"],"transaction_type":"SELL" if k.startswith("sell") else "BUY"} for k,lg in result["legs"].items()]
    result["margin_required"],result["margin_error"]=compute_margin(legs_for_margin,quantity)
    result["entry_event_warning"]=get_entry_warning(); result["event_before_expiry"]=get_event_before_expiry(result["expiry"])
    entry=[{"price":lg.get("mid",lg["ltp"]),"quantity":quantity,"transaction_type":"SELL" if k.startswith("sell") else "BUY"} for k,lg in result["legs"].items()]
    result["estimated_entry_charges"]=estimate_charges(entry); result["net_profit_after_entry_charges"]=round(result["max_profit"]-result["estimated_entry_charges"]["total"],2) if result.get("max_profit") is not None else None
    rank=result.get("rank_info") or {}; iv_hv=classify_iv_hv(rank.get("atm_iv_pct"),rank.get("hv_annualized_pct")); result["iv_hv"]=iv_hv
    em=expected_move(spot,rank.get("atm_iv_pct"),result["days_to_expiry"]); result["expected_move"]=em
    if em and strategy_type=="iron_condor":
        result["ce_cushion_em"]=round((result["legs"]["sell_call"]["strike"]-spot)/em["expected_move"],2)
        result["pe_cushion_em"]=round((spot-result["legs"]["sell_put"]["strike"])/em["expected_move"],2)
        result["probability_of_profit_pct"]=model_expiry_probability_between(spot,result["breakeven_lower"],result["breakeven_upper"],rank.get("atm_iv_pct"),result["days_to_expiry"])
    else:
        result["probability_of_profit_pct"]=round(max(0,(1-abs(cd)-abs(pd))*100),1)
    for k in ("sell_call","sell_put"):
        if k in result["legs"]: result["legs"][k]["probability_of_touch_pct"]=probability_of_touch(result["legs"][k]["delta"])
    trend=get_trend_regime(symbol); result["trend"]=None if trend.get("error") else trend
    vix,vix_err=get_india_vix(); result["volatility_regime"]=classify_volatility_regime(vix,rank.get("iv_rank_pct"))
    if strategy_type=="iron_condor":
        liq=min([result["legs"][k].get("oi",0) for k in result["legs"]]) if result.get("legs") else 0
        spreads=[result["legs"][k].get("spread_pct") for k in result["legs"] if result["legs"][k].get("spread_pct") is not None]
        result["ic_liquidity_ok"]=liq>=IC_MIN_LEG_OI and (not spreads or max(spreads)<=IC_MAX_LEG_SPREAD_PCT)
        result["ic_credit_max_loss_pct"]=round(result["net_credit_per_share"]/max(result["max_loss"]/quantity,1e-9)*100,1)
        result["ic_event_score"],result["ic_event_flags"]=ic_event_risk(symbol,result["expiry"])
        result["trade_quality_score"]=round((0.25*(rank.get("ic_score") or 50)+0.25*(iv_hv and min(100,max(0,(iv_hv["ratio"]-0.8)/0.8*100)) or 50)+
                                             0.25*(min(result.get("ce_cushion_em",0),result.get("pe_cushion_em",0))/1.5*100)+0.25*(80 if result["ic_liquidity_ok"] else 30)),1)
    else: result["trade_quality_score"]=rank.get("composite_score")
    result["trade_quality_label"]="Excellent" if result["trade_quality_score"]>=80 else "Good" if result["trade_quality_score"]>=60 else "Average" if result["trade_quality_score"]>=40 else "Avoid"
    result["suggested_strategy"]=suggest_strategy_family(rank.get("iv_rank_pct"),trend)
    result["trade_quality_note"]="IC score is a transparent heuristic combining volatility richness, range behaviour, expected-move cushion, four-leg liquidity and trade economics. It is not backtested and is not a probability guarantee."
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
    put_delta = request.args.get("put_delta")
    call_delta = request.args.get("call_delta")
    wing_mode = request.args.get("wing_mode", "auto")
    result = build_strategy(symbol, target_delta, wing_width_pct, strategy_type, expiry_str, lots,
                            put_delta=float(put_delta) if put_delta not in (None, "") else None,
                            call_delta=float(call_delta) if call_delta not in (None, "") else None,
                            wing_mode=wing_mode)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


# ---------------------------------------------------------------------------
# Strategy builder — Double Calendar Spread (sell near-term call+put, buy far-term
# call+put at the same two strikes). Net debit, defined risk, long vega / positive theta.
# ---------------------------------------------------------------------------
def pick_calendar_expiries(all_expiries_str, near_expiry_str_override=None, far_expiry_str_override=None):
    """Auto-picks a sensible near/far expiry pair from the full expiry list (strings 'YYYY-MM-DD').
    Near = nearest expiry beyond MIN_DAYS_TO_EXPIRY (same rule as the other strategy builders).
    Far = the available expiry whose gap from Near is closest to CALENDAR_TARGET_GAP_DAYS (and
    strictly after Near) — this is what "automatically pick which expiries" means in practice:
    typically the current/next-week expiry paired with the next monthly one or two out.
    Returns (near_str, far_str) or (None, None, error_dict)."""
    today = now_ist().date()
    all_expiries = sorted(datetime.strptime(e, "%Y-%m-%d").date() for e in all_expiries_str)

    if near_expiry_str_override:
        try:
            near = datetime.strptime(near_expiry_str_override, "%Y-%m-%d").date()
        except ValueError:
            return None, None, {"error": f"Invalid near_expiry format '{near_expiry_str_override}'"}
        if near not in all_expiries:
            return None, None, {"error": f"{near_expiry_str_override} is not a valid expiry for this symbol"}
    else:
        valid = [e for e in all_expiries if (e - today).days >= MIN_DAYS_TO_EXPIRY]
        if not valid:
            return None, None, {"error": "No near expiry beyond minimum days-to-expiry filter"}
        near = valid[0]

    later = [e for e in all_expiries if e > near]
    if not later:
        return None, None, {"error": f"No later expiry available beyond near expiry {near} to use as the far leg"}

    if far_expiry_str_override:
        try:
            far = datetime.strptime(far_expiry_str_override, "%Y-%m-%d").date()
        except ValueError:
            return None, None, {"error": f"Invalid far_expiry format '{far_expiry_str_override}'"}
        if far not in later:
            return None, None, {"error": f"{far_expiry_str_override} must be a valid expiry strictly after {near}"}
    else:
        far = min(later, key=lambda e: abs((e - near).days - CALENDAR_TARGET_GAP_DAYS))

    return str(near), str(far), None


def build_double_calendar_strategy(symbol, strike_mode="otm_pct", otm_pct=DEFAULT_CALENDAR_OTM_PCT,
                                    target_delta=DEFAULT_CALENDAR_TARGET_DELTA,
                                    near_expiry_str=None, far_expiry_str=None, lots=1):
    symbol = symbol.upper()
    today = now_ist().date()

    # Resolve which two expiries to use (auto-picked unless the user overrode one/both).
    nfo, _ = get_instruments()
    opts = [i for i in nfo if i["name"] == symbol and i["segment"] == "NFO-OPT"]
    if not opts:
        return {"error": f"No options found for {symbol}"}
    all_expiries_str = [str(e) for e in sorted({o["expiry"] for o in opts})]
    near_expiry_str, far_expiry_str, err = pick_calendar_expiries(all_expiries_str, near_expiry_str, far_expiry_str)
    if err:
        return err

    near_data, err = get_chain_for_symbol(symbol, near_expiry_str)
    if err:
        return err
    far_data, err = get_chain_for_symbol(symbol, far_expiry_str)
    if err:
        return err

    spot = near_data["spot"]
    lot_size = near_data["lot_size"]
    quantity = lot_size * max(1, int(lots))
    near_expiry = near_data["expiry"]
    far_expiry = far_data["expiry"]
    days_to_near = (near_expiry - today).days
    days_between = (far_expiry - near_expiry).days
    if days_between <= 0:
        return {"error": "Far expiry must be strictly after near expiry"}

    near_calls = sorted([o for o in near_data["chain"] if o["instrument_type"] == "CE"], key=lambda x: x["strike"])
    near_puts = sorted([o for o in near_data["chain"] if o["instrument_type"] == "PE"], key=lambda x: x["strike"])
    far_calls = sorted([o for o in far_data["chain"] if o["instrument_type"] == "CE"], key=lambda x: x["strike"])
    far_puts = sorted([o for o in far_data["chain"] if o["instrument_type"] == "PE"], key=lambda x: x["strike"])
    if not near_calls or not near_puts or not far_calls or not far_puts:
        return {"error": "Could not load a complete call/put chain for both expiries"}

    def closest_strike(options, target):
        return min(options, key=lambda o: abs(o["strike"] - target))

    def closest_by_delta(options, target, sign):
        return min(options, key=lambda o: abs(o["delta"] - sign * target))

    if strike_mode == "atm":
        call_strike_target = put_strike_target = spot
        near_call = closest_strike(near_calls, call_strike_target)
        near_put = closest_strike(near_puts, put_strike_target)
    elif strike_mode == "delta":
        near_call = closest_by_delta(near_calls, target_delta, +1)
        near_put = closest_by_delta(near_puts, target_delta, -1)
    else:  # "otm_pct" (default)
        near_call = closest_strike(near_calls, spot * (1 + otm_pct))
        near_put = closest_strike(near_puts, spot * (1 - otm_pct))

    # Match the SAME strikes on the far expiry (nearest available if strikes differ slightly).
    far_call = closest_strike(far_calls, near_call["strike"])
    far_put = closest_strike(far_puts, near_put["strike"])

    def leg(o):
        return {"strike": o["strike"], "ltp": o["ltp"], "delta": o["delta"], "iv_pct": o["iv"],
                "tradingsymbol": o["tradingsymbol"]}

    legs = {"sell_call_near": leg(near_call), "sell_put_near": leg(near_put),
            "buy_call_far": leg(far_call), "buy_put_far": leg(far_put)}

    net_debit_per_share = ((far_call["ltp"] + far_put["ltp"]) - (near_call["ltp"] + near_put["ltp"]))
    max_loss_per_share = max(net_debit_per_share, 0.0)  # defined risk: worst case both near legs expire
    # worthless and you simply own the far legs, having overpaid the debit — you lose at most the debit.

    # --- Model-based P&L curve at NEAR expiry, across a range of assumed spot outcomes ---
    # At near expiry: the short near-leg is worth its intrinsic value (you owe that to close it);
    # the long far-leg still has (far_expiry - near_expiry) days left, valued via Black-Scholes at
    # today's implied vol for that leg (assumes IV holds roughly steady — the standard simplifying
    # assumption for calendar-spread payoff diagrams; real IV can/does change).
    T_far_remaining = days_between / 365.0
    call_iv = far_call["iv"] / 100.0
    put_iv = far_put["iv"] / 100.0
    call_strike, put_strike = near_call["strike"], near_put["strike"]

    def pnl_at_spot(s_t):
        near_call_intrinsic = max(s_t - call_strike, 0.0)
        near_put_intrinsic = max(put_strike - s_t, 0.0)
        far_call_value = bs_price(s_t, call_strike, T_far_remaining, RISK_FREE_RATE, call_iv, "CE")
        far_put_value = bs_price(s_t, put_strike, T_far_remaining, RISK_FREE_RATE, put_iv, "PE")
        position_value = (far_call_value - near_call_intrinsic) + (far_put_value - near_put_intrinsic)
        return position_value - net_debit_per_share

    lo = spot * (1 - CALENDAR_CURVE_RANGE_PCT)
    hi = spot * (1 + CALENDAR_CURVE_RANGE_PCT)
    step = (hi - lo) / (CALENDAR_CURVE_POINTS - 1)
    curve = []
    for i in range(CALENDAR_CURVE_POINTS):
        s_t = lo + i * step
        pnl_per_share = pnl_at_spot(s_t)
        curve.append({"spot": round(s_t, 2), "pnl": round(pnl_per_share * quantity, 2)})

    max_profit_point = max(curve, key=lambda pt: pt["pnl"])
    max_profit_estimated = max_profit_point["pnl"]

    # Breakevens: spot values where the curve crosses zero (linear interpolation between samples).
    breakevens = []
    for i in range(len(curve) - 1):
        p1, p2 = curve[i], curve[i + 1]
        if (p1["pnl"] <= 0 <= p2["pnl"]) or (p1["pnl"] >= 0 >= p2["pnl"]):
            if p2["pnl"] != p1["pnl"]:
                frac = -p1["pnl"] / (p2["pnl"] - p1["pnl"])
                be_spot = p1["spot"] + frac * (p2["spot"] - p1["spot"])
                breakevens.append(round(be_spot, 2))
    # de-dupe near-identical crossings
    dedup_breakevens = []
    for b in breakevens:
        if not any(abs(b - x) < 0.5 for x in dedup_breakevens):
            dedup_breakevens.append(b)

    result = {
        "symbol": symbol, "spot": spot, "lot_size": lot_size, "lots": lots, "quantity": quantity,
        "strategy_type": "double_calendar", "strike_mode": strike_mode,
        "near_expiry": str(near_expiry), "far_expiry": str(far_expiry),
        "days_to_near_expiry": days_to_near, "days_between_expiries": days_between,
        "all_expiries": all_expiries_str,
        "legs": legs,
        "net_debit_per_share": round(net_debit_per_share, 2),
        "max_loss": round(max_loss_per_share * quantity, 2),
        "max_profit_estimated": round(max_profit_estimated, 2),
        "breakevens": dedup_breakevens,
        "sweet_spot_range": [near_put["strike"], near_call["strike"]],
        "curve": curve,
        "note": ("DOUBLE CALENDAR SPREAD: net-debit, defined-risk trade. Max loss is capped at the debit "
                 "paid; max profit is a MODEL ESTIMATE (Black-Scholes value of the far leg at near expiry, "
                 "assuming today's IV holds) — not guaranteed, since realized IV and the exact time of exit "
                 "both move the actual P&L. Profit is maximized if spot sits between the two short strikes "
                 "at near expiry; sharp moves in either direction erode it. Educational calculation only — "
                 "not a trade recommendation. Verify prices, margin, and lot size on your broker terminal.")
    }

    legs_for_margin = [
        {"tradingsymbol": legs["sell_call_near"]["tradingsymbol"], "transaction_type": "SELL"},
        {"tradingsymbol": legs["sell_put_near"]["tradingsymbol"], "transaction_type": "SELL"},
        {"tradingsymbol": legs["buy_call_far"]["tradingsymbol"], "transaction_type": "BUY"},
        {"tradingsymbol": legs["buy_put_far"]["tradingsymbol"], "transaction_type": "BUY"},
    ]
    margin_required, margin_error = compute_margin(legs_for_margin, quantity)
    result["margin_required"] = margin_required
    result["margin_error"] = margin_error
    result["entry_event_warning"] = get_entry_warning()
    result["event_before_expiry"] = get_event_before_expiry(result["near_expiry"])

    entry_orders_for_charges = [
        {"price": legs["sell_call_near"]["ltp"], "quantity": quantity, "transaction_type": "SELL"},
        {"price": legs["sell_put_near"]["ltp"], "quantity": quantity, "transaction_type": "SELL"},
        {"price": legs["buy_call_far"]["ltp"], "quantity": quantity, "transaction_type": "BUY"},
        {"price": legs["buy_put_far"]["ltp"], "quantity": quantity, "transaction_type": "BUY"},
    ]
    entry_charges = estimate_charges(entry_orders_for_charges)
    result["estimated_entry_charges"] = entry_charges
    result["charges_note"] = ("Entry-side charges only. If you square off before near expiry, exit-side "
                               "charges apply too — see the Track Positions section for the running "
                               "round-trip estimate once tracked.")

    # --- Reuse the same trading-logic layer as the Iron Condor/Strangle builder ---
    rank_info = get_stock_rank(symbol)
    result["rank_info"] = rank_info
    iv_hv = classify_iv_hv(rank_info.get("atm_iv_pct") if rank_info else None,
                            rank_info.get("hv_annualized_pct") if rank_info else None)
    result["iv_hv"] = iv_hv
    if iv_hv is None:
        result["iv_hv_note"] = "Run the Screener (section 1) first so IV/HV data is cached for this symbol."

    em = expected_move(spot, rank_info.get("atm_iv_pct") if rank_info else None, days_to_near)
    result["expected_move"] = em
    if em:
        outside_sweet_spot = em["upper"] > near_call["strike"] or em["lower"] < near_put["strike"]
        result["expected_move_vs_sweet_spot"] = (
            f"{days_to_near}-day expected move (±₹{em['expected_move']}, range {em['lower']}–{em['upper']}) "
            + ("extends BEYOND the short strikes (" + f"{near_put['strike']}–{near_call['strike']}"
               + ") — a normal move could already erode profit before near expiry."
               if outside_sweet_spot else
               "comfortably stays WITHIN the short strikes (" + f"{near_put['strike']}–{near_call['strike']}"
               + ") — favorable for this trade."))

    trend = get_trend_regime(symbol)
    result["trend"] = None if trend.get("error") else trend
    if trend.get("error"):
        result["trend_note"] = trend["error"]
    if trend and not trend.get("error") and trend.get("avoid_premium_selling"):
        result["trend_warning"] = (f"{trend['regime']} detected — calendars do best in range-bound/low-trend "
                                    f"conditions; a strong trend risks pushing spot outside the sweet spot.")

    vix, vix_err = get_india_vix()
    iv_rank_for_regime = rank_info.get("iv_rank_pct") if rank_info else None
    result["volatility_regime"] = classify_volatility_regime(vix, iv_rank_for_regime)
    if vix_err:
        result["volatility_regime"]["note"] = f"India VIX fetch failed ({vix_err}); classification unavailable."
    # Calendars are LONG vega (unlike condors/strangles which are short vega) — a rising-IV regime
    # after entry helps this trade, so flip the usual "avoid high vol" framing into a note here.
    result["vega_note"] = ("This trade is net LONG vega (benefits if IV rises after entry) and net SHORT "
                            "gamma near-term — opposite of the Iron Condor/Strangle builder's exposure. "
                            "A low-IV entry (cheap far-month vega) with room for IV to expand is typically "
                            "more favorable than entering when IV is already elevated.")

    score_components = []
    if iv_hv:
        # For a long-vega trade, a LOW iv/hv ratio (calm now, room to expand) scores better — inverse
        # of the condor/strangle scoring, which wants rich IV to sell.
        inv_label = {"avoid": 90, "fair": 70, "good": 55, "excellent": 35}.get(iv_hv["label"].lower(), 50)
        score_components.append(inv_label)
    if trend and not trend.get("error"):
        score_components.append(25 if trend.get("avoid_premium_selling") else 80)
    if result["volatility_regime"]["label"] != "Unknown":
        vr_score = {"Low Volatility": 80, "Normal": 65, "High Volatility": 35, "Extreme": 15}.get(
            result["volatility_regime"]["label"], 50)
        score_components.append(vr_score)
    if rank_info and rank_info.get("fo_banned_today"):
        score_components.append(0)
    trade_quality_score = round(sum(score_components) / len(score_components), 1) if score_components else None
    result["trade_quality_score"] = trade_quality_score
    if trade_quality_score is not None:
        result["trade_quality_label"] = ("Excellent" if trade_quality_score >= 80 else
                                          "Good" if trade_quality_score >= 60 else
                                          "Average" if trade_quality_score >= 40 else "Avoid")
    result["trade_quality_note"] = ("Heuristic score for a LONG-VEGA/theta trade: rewards calm current IV "
                                     "with room to rise, range-bound trend, and low-to-normal volatility "
                                     "regime — the inverse of the premium-selling score elsewhere in this "
                                     "dashboard. Not a probability, not backtested — a rough triage aid only.")

    return result


@app.route("/api/calendar-strategy/<symbol>")
def calendar_strategy(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    strike_mode = request.args.get("strike_mode", "otm_pct")
    otm_pct = float(request.args.get("otm_pct", DEFAULT_CALENDAR_OTM_PCT))
    target_delta = float(request.args.get("target_delta", DEFAULT_CALENDAR_TARGET_DELTA))
    near_expiry = request.args.get("near_expiry")
    far_expiry = request.args.get("far_expiry")
    lots = int(request.args.get("lots", 1))
    result = build_double_calendar_strategy(symbol, strike_mode, otm_pct, target_delta,
                                             near_expiry, far_expiry, lots)
    if "error" in result:
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/trend/<symbol>")
def trend(symbol):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    result = get_trend_regime(symbol)
    if result.get("error"):
        return jsonify(result), 404
    return jsonify(result)


@app.route("/api/position-sizing")
def position_sizing():
    """Dynamic position sizing (fixed-fractional): given total capital, risk-per-trade %, and
    the max loss of ONE lot of the trade you're considering, returns how many lots keep you
    within that risk budget."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        capital = float(request.args.get("capital"))
        risk_pct = float(request.args.get("risk_pct"))
        max_loss_per_lot = float(request.args.get("max_loss_per_lot"))
    except (TypeError, ValueError):
        return jsonify({"error": "capital, risk_pct, and max_loss_per_lot are all required numeric params"}), 400
    result = recommended_position_size(capital, risk_pct, max_loss_per_lot)
    if result is None:
        return jsonify({"error": "Invalid inputs — all values must be positive numbers"}), 400
    result["note"] = ("recommended_lots = floor((capital x risk_pct%) / max_loss_per_lot). This caps your RISK "
                       "budget only — it does not check margin availability. Always confirm actual margin "
                       "required (shown in the Strategy Builder) is within your free cash too.")
    return jsonify(result)


def position_greeks(position):
    """Per-position net Greeks via Black-Scholes at current quotes (Kite doesn't publish Greeks
    itself). Gamma/Vega/Theta are estimated by bump-and-reprice off the same bs_price/bs_delta
    helpers used everywhere else in this file. Handles double_calendar's two different expiries
    (near legs use position['expiry'], far legs use position['far_expiry'])."""
    strategy_type = position.get("strategy_type", "iron_condor")
    leg_keys = leg_keys_for(position)
    quantity = position.get("quantity", position["lot_size"])
    spot, err = get_spot_price(position["symbol"])
    if err:
        return {"error": err["error"]}

    today = now_ist().date()
    near_expiry_date = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    days_left_near = max((near_expiry_date - today).days, 0)
    far_expiry_date = None
    days_left_far = None
    if strategy_type == "double_calendar":
        far_expiry_date = datetime.strptime(position["far_expiry"], "%Y-%m-%d").date()
        days_left_far = max((far_expiry_date - today).days, 0)
        if days_left_near <= 0 and days_left_far <= 0:
            return {"error": "Position has expired"}
    else:
        if days_left_near <= 0:
            return {"error": "Position has expired"}

    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    try:
        quotes = kite_quote_bulk(inst_keys)
    except Exception as e:
        return {"error": str(e)}

    net_delta = net_theta = net_vega = net_gamma = 0.0
    for k in leg_keys:
        strike = position["legs"][k]["strike"]
        opt_type = "CE" if "call" in k else "PE"
        # calendars: near legs (sell_*_near) decay against the near expiry; far legs (buy_*_far)
        # against the far expiry. Everything else (iron_condor/strangle) has a single shared expiry.
        T = (days_left_far if (strategy_type == "double_calendar" and k.endswith("_far"))
             else days_left_near) / 365.0
        if T <= 0:
            continue
        ltp = extract_price(quotes.get(f"NFO:{position['legs'][k]['tradingsymbol']}"))
        if ltp is None:
            return {"error": f"No usable price for {k}"}
        iv = implied_vol(ltp, spot, strike, T, opt_type)
        delta = bs_delta(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
        bump_s = spot * 0.01
        delta_up = bs_delta(spot + bump_s, strike, T, RISK_FREE_RATE, iv, opt_type)
        gamma = (delta_up - delta) / bump_s if bump_s else 0.0
        vega = (bs_price(spot, strike, T, RISK_FREE_RATE, iv + 0.01, opt_type)
                - bs_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type))
        theta = -(bs_price(spot, strike, max(T - 1 / 365, 0), RISK_FREE_RATE, iv, opt_type)
                  - bs_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type))
        sign = -1 if k.startswith("sell") else 1
        net_delta += sign * delta * quantity
        net_gamma += sign * gamma * quantity
        net_vega += sign * vega * quantity
        net_theta += sign * theta * quantity

    return {"net_delta": round(net_delta, 2), "net_gamma": round(net_gamma, 4),
            "net_vega": round(net_vega, 2), "net_theta": round(net_theta, 2)}



@app.route("/api/portfolio-greeks")
def portfolio_greeks():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_positions()
    total_delta = total_theta = total_vega = total_gamma = 0.0
    details, errors = [], []
    for p in positions:
        g = position_greeks(p)
        if g.get("error"):
            errors.append({"id": p["id"], "symbol": p["symbol"], "error": g["error"]})
            continue
        total_delta += g["net_delta"]; total_gamma += g["net_gamma"]
        total_vega += g["net_vega"]; total_theta += g["net_theta"]
        details.append({"id": p["id"], "symbol": p["symbol"], **g})
    return jsonify({
        "net_delta": round(total_delta, 2), "net_gamma": round(total_gamma, 4),
        "net_vega": round(total_vega, 2), "net_theta": round(total_theta, 2),
        "positions": details, "errors": errors,
        "note": "Estimated via Black-Scholes at current quotes/implied vol — an approximation, not "
                "Kite's own Greeks (Kite doesn't publish them). Theta is per-day time decay; Vega is "
                "per 1-point (1%) change in IV.",
    })


@app.route("/api/best-trade")
def best_trade():
    """Rule-based 'Today's Best Trade' — combines the current Screener ranking with the Strategy
    Builder's enhanced output (IV/HV, expected move, trend, volatility regime) into one summary.
    This is NOT a machine-learning prediction and is NOT validated by backtesting — it's a
    transparent aggregation of the same signals shown elsewhere in this dashboard."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    if not SCREENER_CACHE["results"]:
        return jsonify({"error": "Run the Screener (section 1) first."}), 400
    eligible = [r for r in SCREENER_CACHE["results"] if r.get("rank")]
    if not eligible:
        return jsonify({"error": "No eligible stocks in the last screener run."}), 400
    eligible.sort(key=lambda r: r["rank"])
    top = eligible[0]
    strategy_type = request.args.get("strategy_type", "iron_condor")

    built = build_strategy(top["symbol"], strategy_type=strategy_type)
    if "error" in built:
        return jsonify({"error": f"Could not build a strategy for top pick {top['symbol']}: {built['error']}"}), 400

    reasons = []
    if built.get("iv_hv"):
        reasons.append(f"IV/HV ratio {built['iv_hv']['ratio']} ({built['iv_hv']['label']}).")
    if built.get("trend"):
        reasons.append(f"Trend regime: {built['trend']['regime']}.")
    if built.get("volatility_regime"):
        reasons.append(f"Volatility regime: {built['volatility_regime']['label']} "
                        f"(recommendation: {built['volatility_regime']['recommendation']}).")
    if built.get("suggested_strategy", {}).get("reason"):
        reasons.append(built["suggested_strategy"]["reason"])
    if built.get("expected_move_warning"):
        reasons.append(built["expected_move_warning"])

    risks = []
    if built.get("entry_event_warning"):
        risks.append(built["entry_event_warning"])
    if built.get("event_before_expiry"):
        risks.append(f"{built['event_before_expiry']['label']} on {built['event_before_expiry']['date']} "
                      f"falls before this expiry.")
    if built["strategy_type"] == "naked_strangle":
        risks.append("Naked strangle: unlimited risk on the call side.")

    max_profit = built.get("max_profit")
    return jsonify({
        "symbol": built["symbol"], "screener_rank": top["rank"], "strategy": built,
        "why_this_trade": reasons, "risks": risks,
        "expected_return": built.get("net_profit_after_entry_charges"),
        "probability_of_profit_pct": built.get("probability_of_profit_pct"),
        "max_risk": built.get("max_loss"),
        "suggested_exit_plan": [
            f"Profit target: exit at 50% of max profit"
            + (f" (₹{round(max_profit * 0.5, 2)})." if max_profit else "."),
            f"Time exit: close 3 days before expiry ({built['expiry']}) if still open.",
            f"Delta exit: exit a short leg if its delta rises to ≥{STOP_LOSS_DELTA_THRESHOLD}.",
        ],
        "note": "Rule-based triage using the current Screener + Strategy Builder output — NOT a "
                "machine-learning prediction, NOT investment advice, and NOT validated by backtesting. "
                "Verify everything before trading real money.",
    })


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

    today_str = now_ist().date().isoformat()
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
        "entry_estimated_charges": built.get("estimated_entry_charges", {}).get("total"),
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


@app.route("/api/calendar-watchlist/add", methods=["POST"])
def calendar_watchlist_add():
    """Track-a-position counterpart of /api/watchlist/add, for Double Calendar Spreads. Kept as its
    own endpoint (rather than overloading /api/watchlist/add) since the position shape is different
    enough (two expiries, four legs with different leg-key names, debit instead of credit) to be
    clearer as a separate, explicit flow — mirrors how this dashboard keeps the Calendar tab separate."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol = body.get("symbol", "").upper()
    strike_mode = body.get("strike_mode", "otm_pct")
    otm_pct = float(body.get("otm_pct", DEFAULT_CALENDAR_OTM_PCT))
    target_delta = float(body.get("target_delta", DEFAULT_CALENDAR_TARGET_DELTA))
    near_expiry = body.get("near_expiry")
    far_expiry = body.get("far_expiry")
    lots = int(body.get("lots", 1))

    built = build_double_calendar_strategy(symbol, strike_mode, otm_pct, target_delta,
                                            near_expiry, far_expiry, lots)
    if "error" in built:
        return jsonify(built), 404

    today_str = now_ist().date().isoformat()
    position = {
        "id": f"{symbol}_CAL_{int(time.time())}",
        "symbol": symbol,
        "added_on": today_str,
        "entry_spot": built["spot"],
        "strategy_type": "double_calendar",
        "strike_mode": built["strike_mode"],
        "expiry": built["near_expiry"],          # "expiry" = the near/critical management date
        "far_expiry": built["far_expiry"],
        "lot_size": built["lot_size"],
        "lots": built["lots"],
        "quantity": built["quantity"],
        "legs": built["legs"],
        "entry_net_debit_per_share": built["net_debit_per_share"],
        "entry_max_loss": built["max_loss"],
        "entry_max_profit_estimated": built["max_profit_estimated"],
        "entry_margin_required": built.get("margin_required"),
        "entry_margin_error": built.get("margin_error"),
        "entry_estimated_charges": built.get("estimated_entry_charges", {}).get("total"),
        "breakevens": built["breakevens"],
        "sweet_spot_range": built["sweet_spot_range"],
        "broker_orders": [],
        "history": [{"date": today_str, "spot": built["spot"],
                     "pnl": 0.0, "current_debit_per_share": built["net_debit_per_share"]}],
    }
    positions = load_positions()
    positions.append(position)
    save_positions(positions)
    return jsonify({"ok": True, "position": position})


@app.route("/api/calendar-watchlist/<pos_id>/curve")
def calendar_watchlist_curve(pos_id):
    """Regenerates a LIVE payoff curve for an already-tracked calendar position, using current spot
    and current far-leg IV (rather than the IV at entry time) — lets the Track Positions tab show how
    the expected max-profit/max-loss shape has shifted since entry, not just the frozen entry-day curve."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position or position.get("strategy_type") != "double_calendar":
        return jsonify({"error": "Calendar position not found"}), 404

    spot, err = get_spot_price(position["symbol"])
    if err:
        return jsonify(err), 404

    today = now_ist().date()
    near_expiry = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    far_expiry = datetime.strptime(position["far_expiry"], "%Y-%m-%d").date()
    days_to_near = max((near_expiry - today).days, 0)
    days_between = max((far_expiry - near_expiry).days, 1)

    call_strike = position["legs"]["sell_call_near"]["strike"]
    put_strike = position["legs"]["sell_put_near"]["strike"]
    quantity = position.get("quantity", position["lot_size"])

    inst_keys = [f"NFO:{position['legs']['buy_call_far']['tradingsymbol']}",
                 f"NFO:{position['legs']['buy_put_far']['tradingsymbol']}"]
    try:
        quotes = kite_quote_bulk(inst_keys)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    T_near_remaining = days_to_near / 365.0
    far_call_ltp = extract_price(quotes.get(inst_keys[0]))
    far_put_ltp = extract_price(quotes.get(inst_keys[1]))
    if far_call_ltp is None or far_put_ltp is None:
        return jsonify({"error": "No usable live price for one or both far legs"}), 400
    call_iv = implied_vol(far_call_ltp, spot, call_strike, T_near_remaining + days_between / 365.0, "CE")
    put_iv = implied_vol(far_put_ltp, spot, put_strike, T_near_remaining + days_between / 365.0, "PE")

    entry_debit = position["entry_net_debit_per_share"]
    T_far_remaining = days_between / 365.0

    def pnl_at_spot(s_t):
        near_call_intrinsic = max(s_t - call_strike, 0.0)
        near_put_intrinsic = max(put_strike - s_t, 0.0)
        far_call_value = bs_price(s_t, call_strike, T_far_remaining, RISK_FREE_RATE, call_iv, "CE")
        far_put_value = bs_price(s_t, put_strike, T_far_remaining, RISK_FREE_RATE, put_iv, "PE")
        position_value = (far_call_value - near_call_intrinsic) + (far_put_value - near_put_intrinsic)
        return position_value - entry_debit

    lo = spot * (1 - CALENDAR_CURVE_RANGE_PCT)
    hi = spot * (1 + CALENDAR_CURVE_RANGE_PCT)
    step = (hi - lo) / (CALENDAR_CURVE_POINTS - 1)
    curve = []
    for i in range(CALENDAR_CURVE_POINTS):
        s_t = lo + i * step
        curve.append({"spot": round(s_t, 2), "pnl": round(pnl_at_spot(s_t) * quantity, 2)})

    return jsonify({
        "position_id": pos_id, "spot": spot, "days_to_near_expiry": days_to_near,
        "curve": curve, "sweet_spot_range": [put_strike, call_strike],
        "note": "Live re-estimate using current spot and today's implied vol on the far legs — the "
                "curve shape will keep shifting daily as time passes and IV moves; treat it as a "
                "current best-guess snapshot, not a fixed prediction."
    })




@app.route("/api/watchlist/<pos_id>", methods=["DELETE"])
def watchlist_remove(pos_id):
    positions = load_positions()
    positions = [p for p in positions if p["id"] != pos_id]
    save_positions(positions)
    return jsonify({"ok": True})


def mark_to_market_calendar(position):
    """Double Calendar equivalent of mark_to_market() below — kept separate because the P&L math,
    zone logic, and exit rules are genuinely different for a debit calendar vs a credit condor/strangle
    (two expiries, model-based re-valuation of the far leg instead of a simple credit/debit diff)."""
    quantity = position.get("quantity", position["lot_size"])
    leg_keys = ["sell_call_near", "sell_put_near", "buy_call_far", "buy_put_far"]

    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite_quote_bulk(inst_keys)

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

    # Current cost to CLOSE this spread: sell the far longs at their ltp, buy back the near shorts
    # at their ltp. Position value rising above the entry debit is what "profit" means here.
    current_value_per_share = ((prices["buy_call_far"] - prices["sell_call_near"]) +
                                (prices["buy_put_far"] - prices["sell_put_near"]))
    entry_debit = position["entry_net_debit_per_share"]
    pnl_per_share = current_value_per_share - entry_debit
    pnl = round(pnl_per_share * quantity, 2)
    current_position_value = round(current_value_per_share * quantity, 2)

    spot, err = get_spot_price(position["symbol"])
    if err:
        return {"__error__": err["error"]}

    today = now_ist().date()
    near_expiry_date = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    far_expiry_date = datetime.strptime(position["far_expiry"], "%Y-%m-%d").date()
    days_left = (near_expiry_date - today).days
    T_near_remaining = max(days_left, 0) / 365.0

    call_strike = position["legs"]["sell_call_near"]["strike"]
    put_strike = position["legs"]["sell_put_near"]["strike"]
    sweet_spot_lo, sweet_spot_hi = put_strike, call_strike

    zone = "safe"
    if spot > sweet_spot_hi or spot < sweet_spot_lo:
        zone = "outside_sweet_spot"
    if days_left <= CALENDAR_NEAR_EXPIRY_DAYS_WARNING:
        zone = "near_expiry"

    delta_call = delta_put = None
    if days_left > 0:
        iv_call = implied_vol(prices["sell_call_near"], spot, call_strike, T_near_remaining, "CE")
        iv_put = implied_vol(prices["sell_put_near"], spot, put_strike, T_near_remaining, "PE")
        delta_call = bs_delta(spot, call_strike, T_near_remaining, RISK_FREE_RATE, iv_call, "CE")
        delta_put = bs_delta(spot, put_strike, T_near_remaining, RISK_FREE_RATE, iv_put, "PE")

    # Probability spot is still WITHIN the sweet spot (between the two short strikes) at near expiry —
    # lognormal approx using the same expected-move machinery used elsewhere in this file.
    probability_in_sweet_spot = None
    rank_info = get_stock_rank(position["symbol"])
    atm_iv_pct = rank_info.get("atm_iv_pct") if rank_info else None
    if atm_iv_pct and days_left > 0 and spot:
        sigma = atm_iv_pct / 100.0
        T = days_left / 365.0
        if sigma > 0 and T > 0:
            d_hi = (math.log(sweet_spot_hi / spot)) / (sigma * math.sqrt(T))
            d_lo = (math.log(sweet_spot_lo / spot)) / (sigma * math.sqrt(T))
            probability_in_sweet_spot = round((norm_cdf(d_hi) - norm_cdf(d_lo)) * 100, 1)
    elif days_left <= 0:
        probability_in_sweet_spot = 100.0 if zone == "safe" else 0.0

    # --- Exit suggestion (informational only) ---
    exit_suggested, exit_reasons = False, []
    entry_debit_total = abs(entry_debit * quantity)
    if entry_debit_total and pnl <= -CALENDAR_STOP_LOSS_DEBIT_MULTIPLE * entry_debit_total:
        exit_suggested = True
        exit_reasons.append(f"Loss (₹{abs(pnl)}) has reached {CALENDAR_STOP_LOSS_DEBIT_MULTIPLE}x the debit "
                             f"paid (₹{round(entry_debit_total,2)}).")
    if zone == "outside_sweet_spot":
        exit_suggested = True
        exit_reasons.append(f"Spot (₹{spot}) has moved outside the sweet spot range "
                             f"({sweet_spot_lo}–{sweet_spot_hi}) — the near leg is losing its edge.")
    if days_left <= CALENDAR_NEAR_EXPIRY_DAYS_WARNING and days_left >= 0:
        exit_suggested = True
        exit_reasons.append(f"Only {days_left} day(s) to near-leg expiry — consider closing or rolling "
                             f"the near leg to manage gamma/assignment risk.")

    event_flag = get_event_before_expiry(position["expiry"])

    exit_orders_for_charges = [
        {"price": prices["sell_call_near"], "quantity": quantity, "transaction_type": "BUY"},
        {"price": prices["sell_put_near"], "quantity": quantity, "transaction_type": "BUY"},
        {"price": prices["buy_call_far"], "quantity": quantity, "transaction_type": "SELL"},
        {"price": prices["buy_put_far"], "quantity": quantity, "transaction_type": "SELL"},
    ]
    exit_charges = estimate_charges(exit_orders_for_charges)
    entry_charges_total = position.get("entry_estimated_charges") or 0
    round_trip_charges = round(entry_charges_total + exit_charges["total"], 2)
    net_pnl_after_charges = round(pnl - round_trip_charges, 2)

    leg_details = {}
    for k in leg_keys:
        entry_price = position["legs"][k]["ltp"]
        current_price = prices[k]
        is_sell = k.startswith("sell")
        per_share = (entry_price - current_price) if is_sell else (current_price - entry_price)
        leg_details[k] = {
            "tradingsymbol": position["legs"][k]["tradingsymbol"],
            "strike": position["legs"][k]["strike"],
            "entry_price": entry_price, "current_price": round(current_price, 2),
            "pnl": round(per_share * quantity, 2),
        }
        if k == "sell_call_near" and delta_call is not None:
            leg_details[k]["current_delta"] = round(delta_call, 3)
        if k == "sell_put_near" and delta_put is not None:
            leg_details[k]["current_delta"] = round(delta_put, 3)

    entry_max_profit = position.get("entry_max_profit_estimated")
    return {
        "spot": spot, "pnl": pnl, "current_debit_per_share": round(current_value_per_share, 2),
        "current_position_value": current_position_value, "legs_current": leg_details,
        "days_left": days_left, "zone": zone,
        "probability_in_sweet_spot_pct": probability_in_sweet_spot,
        "pct_of_max_profit": round((pnl / entry_max_profit * 100), 1) if entry_max_profit else None,
        "exit_suggested": exit_suggested, "exit_reasons": exit_reasons,
        "event_before_expiry": event_flag,
        "entry_charges": entry_charges_total, "estimated_exit_charges": exit_charges["total"],
        "estimated_round_trip_charges": round_trip_charges, "net_pnl_after_charges": net_pnl_after_charges,
        "sweet_spot_range": [sweet_spot_lo, sweet_spot_hi],
    }


def mark_to_market(position):
    if position.get("strategy_type") == "double_calendar":
        return mark_to_market_calendar(position)

    strategy_type = position.get("strategy_type", "iron_condor")
    leg_keys = ["sell_call", "buy_call", "sell_put", "buy_put"] if strategy_type == "iron_condor" \
        else ["sell_call", "sell_put"]
    quantity = position.get("quantity", position["lot_size"])

    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite_quote_bulk(inst_keys)

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

    today = now_ist().date()
    expiry_date = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    days_left = (expiry_date - today).days
    T_remaining = max(days_left, 0) / 365.0

    zone = "safe"
    if spot > position["breakeven_upper"] or spot < position["breakeven_lower"]:
        zone = "breached"
    elif days_left <= 2:
        zone = "near_expiry"

    probability_of_success = None
    delta_call = delta_put = None
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

    # --- Stop-loss / exit suggestion (informational only — never auto-exits) ---
    # Trigger on whichever occurs first: total loss reaches N x premium received, or
    # either short leg's delta magnitude has risen to the threshold. Checking delta rather
    # than only waiting for the theoretical max loss catches a position going wrong earlier.
    exit_suggested, exit_reasons = False, []
    entry_premium_total = abs(position["entry_net_credit_per_share"] * quantity)
    if entry_premium_total and pnl <= -STOP_LOSS_PREMIUM_MULTIPLE * entry_premium_total:
        exit_suggested = True
        exit_reasons.append(f"Loss (₹{abs(pnl)}) has reached {STOP_LOSS_PREMIUM_MULTIPLE}x the premium "
                             f"received (₹{entry_premium_total}).")
    if delta_call is not None and abs(delta_call) >= STOP_LOSS_DELTA_THRESHOLD:
        exit_suggested = True
        exit_reasons.append(f"Short call delta has risen to {round(delta_call, 3)} "
                             f"(≥{STOP_LOSS_DELTA_THRESHOLD} threshold) — that side is losing its 'safety margin'.")
    if delta_put is not None and abs(delta_put) >= STOP_LOSS_DELTA_THRESHOLD:
        exit_suggested = True
        exit_reasons.append(f"Short put delta has risen to {round(delta_put, 3)} "
                             f"(≥{STOP_LOSS_DELTA_THRESHOLD} threshold) — that side is losing its 'safety margin'.")

    event_flag = get_event_before_expiry(position["expiry"])

    # --- Charges: entry (stored at tracking time) + a live exit-side estimate, giving a running
    # round-trip net P&L. This is what actually answers "what would I really pocket if I closed now."
    exit_orders_for_charges = []
    for k in leg_keys:
        original_txn = "SELL" if k.startswith("sell") else "BUY"
        close_txn = "BUY" if original_txn == "SELL" else "SELL"
        exit_orders_for_charges.append({"price": prices[k], "quantity": quantity, "transaction_type": close_txn})
    exit_charges = estimate_charges(exit_orders_for_charges)
    entry_charges_total = position.get("entry_estimated_charges") or 0
    round_trip_charges = round(entry_charges_total + exit_charges["total"], 2)
    net_pnl_after_charges = round(pnl - round_trip_charges, 2)

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
        if k == "sell_call" and delta_call is not None:
            leg_details[k]["current_delta"] = round(delta_call, 3)
            leg_details[k]["probability_of_touch_pct"] = probability_of_touch(delta_call)
        if k == "sell_put" and delta_put is not None:
            leg_details[k]["current_delta"] = round(delta_put, 3)
            leg_details[k]["probability_of_touch_pct"] = probability_of_touch(delta_put)

    return {
        "spot": spot, "pnl": pnl, "current_debit_per_share": round(current_debit_per_share, 2),
        "current_position_value": current_position_value, "legs_current": leg_details,
        "days_left": days_left, "zone": zone, "probability_of_success_pct": probability_of_success,
        "pct_of_max_profit": round((pnl / position["entry_max_profit"] * 100), 1) if position["entry_max_profit"] else None,
        "exit_suggested": exit_suggested, "exit_reasons": exit_reasons,
        "event_before_expiry": event_flag,
        "entry_charges": entry_charges_total, "estimated_exit_charges": exit_charges["total"],
        "estimated_round_trip_charges": round_trip_charges, "net_pnl_after_charges": net_pnl_after_charges,
    }


@app.route("/api/watchlist")
def watchlist():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_positions()
    today_str = now_ist().date().isoformat()
    out, changed = [], False
    for p in positions:
        try:
            mtm = mark_to_market(p)
        except Exception as e:
            logger.exception("mark_to_market failed for position %s (%s)", p.get("id"), p.get("symbol"))
            out.append({**p, "mtm_error": f"Internal error while pricing this position: {e}"})
            continue
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
    st = position.get("strategy_type")
    if st == "iron_condor":
        return ["sell_call", "buy_call", "sell_put", "buy_put"]
    if st == "double_calendar":
        return ["sell_call_near", "sell_put_near", "buy_call_far", "buy_put_far"]
    return ["sell_call", "sell_put"]


ORDER_TERMINAL_STATUSES = ("COMPLETE", "REJECTED", "CANCELLED")


def wait_for_order_terminal(order_id, timeout_seconds=8, poll_interval=0.5):
    """Polls Kite's order book for a specific order_id until it reaches a terminal state
    (COMPLETE / REJECTED / CANCELLED) or the timeout elapses. Returns the last status seen, or
    'TIMEOUT' if it was still open/pending when we stopped waiting (Kite market orders on NFO
    normally resolve in well under a second, so the timeout is just a safety net against a hung
    poll — it does not cancel the order)."""
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        try:
            for o in kite.orders():
                if o.get("order_id") == order_id:
                    last_status = o.get("status")
                    break
        except Exception:
            pass
        if last_status in ORDER_TERMINAL_STATUSES:
            return last_status
        time.sleep(poll_interval)
    return last_status or "TIMEOUT"


def place_basket_orders(legs_to_place, product, order_type, sequence_for_margin=True):
    """Places each leg as a separate real order. Stops immediately on the first failure rather
    than continuing — continuing could leave a partial, unintentionally unhedged position.

    sequence_for_margin=True (the default for basket entry/close) sends every BUY leg first and
    — for MARKET orders — waits for each BUY to actually reach a terminal state before sending any
    SELL leg. Zerodha checks margin against your live positions at the moment each order hits the
    exchange, so a SELL leg fired before its offsetting BUY leg has filled can get REJECTED for
    insufficient margin even though the combo is fully hedged once both legs are in. Waiting for
    the BUY fill first lets the freed-up/hedged margin actually register before the SELL leg goes.
    Pass sequence_for_margin=False for one-off, independent leg placements/exits where there's no
    basket-level margin ordering to respect (e.g. the per-leg 'Execute this leg' button, or exiting
    an arbitrary set of live positions picked by the user)."""
    ordered = legs_to_place
    if sequence_for_margin:
        buys = [item for item in legs_to_place if item["transaction_type"] == "BUY"]
        sells = [item for item in legs_to_place if item["transaction_type"] != "BUY"]
        ordered = buys + sells

    results = []
    for item in ordered:
        txn_type = kite.TRANSACTION_TYPE_SELL if item["transaction_type"] == "SELL" else kite.TRANSACTION_TYPE_BUY
        quantity = int(item.get("quantity") or 1)
        reference_price = item.get("price")
        try:
            kwargs = dict(
                variety=kite.VARIETY_REGULAR, exchange=item.get("exchange", kite.EXCHANGE_NFO),
                tradingsymbol=item["tradingsymbol"], transaction_type=txn_type,
                quantity=quantity, product=getattr(kite, f"PRODUCT_{product}"),
                order_type=getattr(kite, f"ORDER_TYPE_{order_type}"),
                validity=kite.VALIDITY_DAY,
            )
            if item.get("tag"):
                kwargs["tag"] = str(item["tag"])[:20]
            if item.get("autoslice") is not None:
                kwargs["autoslice"] = bool(item["autoslice"])
            if item.get("market_protection") is not None and order_type in ("MARKET", "SL-M"):
                kwargs["market_protection"] = item["market_protection"]
            if order_type == "LIMIT" and reference_price:
                kwargs["price"] = float(reference_price)
            if order_type in ("MARKET", "SL-M") and "market_protection" not in kwargs:
                # -1 lets Zerodha apply its automatic market-protection band.
                kwargs["market_protection"] = -1
            try:
                order_id = kite.place_order(**kwargs)
            except TypeError as te:
                if "market_protection" in str(te):
                    # Installed kiteconnect SDK predates the market_protection parameter (a known
                    # PyPI packaging gap -- see zerodha/pykiteconnect issue #225). Retry without it;
                    # this will still work UNLESS your broker has already started enforcing the
                    # exchange's market-protection requirement, in which case upgrade the SDK:
                    #   pip install --upgrade kiteconnect
                    kwargs.pop("market_protection", None)
                    order_id = kite.place_order(**kwargs)
                else:
                    raise

            fill_status = None
            if sequence_for_margin and order_type == "MARKET":
                fill_status = wait_for_order_terminal(order_id)
                if fill_status == "REJECTED":
                    results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                                     "transaction_type": item["transaction_type"], "quantity": quantity,
                                     "status": "failed", "order_id": order_id, "fill_status": fill_status,
                                     "error": "Order was REJECTED by the exchange/broker.",
                                     "reference_price": reference_price})
                    break

            results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                             "transaction_type": item["transaction_type"], "quantity": quantity,
                             "status": "placed", "order_id": order_id, "fill_status": fill_status,
                             "estimated_realized_pnl": 0,  # filled in by caller if this is a closing trade
                             "reference_price": reference_price})
        except Exception as e:
            results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                             "transaction_type": item["transaction_type"], "quantity": quantity,
                             "status": "failed", "error": str(e)})
            break
    return results


@app.route("/api/execute/<pos_id>/preview")
def execute_preview(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404

    try:
        orders = refresh_execution_quotes(position)
    except Exception as e:
        return jsonify({"error": f"Could not fetch live Bid/Ask: {e}"}), 502

    return jsonify({
        "position_id": pos_id,
        "symbol": position["symbol"],
        "orders": orders,
        "default_product": "NRML",
        "default_order_type": "LIMIT",
        "warning": "LIMIT prices use best Ask for BUY legs and best Bid for SELL legs. "
                   "Quotes can change before the order reaches the exchange."
    })


@app.route("/api/execute/<pos_id>/refresh-quotes")
def execute_refresh_quotes(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404

    try:
        orders = refresh_execution_quotes(position)
        return jsonify({
            "position_id": pos_id,
            "symbol": position["symbol"],
            "orders": orders,
            "refreshed_at": now_ist().strftime("%H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": f"Could not refresh live Bid/Ask: {e}"}), 502


@app.route("/api/execute/<pos_id>/leg", methods=["POST"])
def execute_single_leg(pos_id):
    """Place exactly one entry leg chosen by the user from the execution review."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    if not body.get("confirmed"):
        return jsonify({"error": "Confirmation flag not set — nothing was placed."}), 400
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404
    order = body.get("order")
    if not order or not order.get("tradingsymbol") or not order.get("transaction_type"):
        return jsonify({"error": "No valid entry leg order provided."}), 400
    product = body.get("product", "NRML")
    order_type = body.get("order_type", "LIMIT")
    order = dict(order)
    if order_type == "LIMIT" and order.get("price_source") == "AUTO":
        try:
            fresh = {o["leg"]: o for o in refresh_execution_quotes(position)}
            fq = fresh.get(order.get("leg"))
            if fq and fq.get("recommended_limit_price") is not None:
                order["price"] = fq["recommended_limit_price"]
                order["ltp"] = fq.get("ltp")
                order["bid"] = fq.get("bid")
                order["ask"] = fq.get("ask")
        except Exception as e:
            return jsonify({"error": f"Could not refresh live Bid/Ask before placement: {e}"}), 502
    if order_type == "LIMIT" and not order.get("price"):
        return jsonify({"error": "A LIMIT price is required. Refresh prices or enter a price manually."}), 400
    results = place_basket_orders([order], product, order_type, sequence_for_margin=False)
    positions = load_positions()
    for p in positions:
        if p["id"] == pos_id:
            p["broker_orders"] = p.get("broker_orders", []) + results
    save_positions(positions)
    return jsonify({"results": results, "position_id": pos_id, "order": order})


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

    if order_type == "LIMIT":
        # Last-second server refresh for rows still marked AUTO.
        try:
            fresh = {o["leg"]: o for o in refresh_execution_quotes(position)}
            refreshed = []
            for item in legs_to_place:
                item = dict(item)
                if item.get("price_source") == "AUTO":
                    fq = fresh.get(item.get("leg"))
                    if fq and fq.get("recommended_limit_price") is not None:
                        item["price"] = fq["recommended_limit_price"]
                refreshed.append(item)
            legs_to_place = refreshed
        except Exception as e:
            return jsonify({"error": f"Could not refresh live Bid/Ask before placement: {e}"}), 502

    results = place_basket_orders(legs_to_place, product, order_type)

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


# ---------------------------------------------------------------------------
# Close / square-off a position — reverses each leg (buy back what you sold,
# sell what you bought) to flatten it before expiry.
# ---------------------------------------------------------------------------
def build_close_orders(position):
    """Reverse of the entry orders, with a fresh reference price per leg from live quotes."""
    quantity = position.get("quantity", position["lot_size"])
    leg_keys = leg_keys_for(position)
    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite_quote_bulk(inst_keys)

    orders = []
    for k in leg_keys:
        leg = position["legs"][k]
        original_txn = "SELL" if k.startswith("sell") else "BUY"
        close_txn = "BUY" if original_txn == "SELL" else "SELL"
        ref_price = extract_price(quotes.get(f"NFO:{leg['tradingsymbol']}"))
        orders.append({
            "leg": k, "tradingsymbol": leg["tradingsymbol"], "transaction_type": close_txn,
            "quantity": quantity, "price": ref_price, "reference_price": ref_price,
            "entry_price": leg["ltp"], "original_transaction_type": original_txn,
        })
    return orders


@app.route("/api/execute/<pos_id>/close/preview")
def close_preview(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404
    orders = build_close_orders(position)
    return jsonify({
        "position_id": pos_id, "symbol": position["symbol"], "orders": orders,
        "default_product": "NRML", "default_order_type": "MARKET",
        "warning": "This will CLOSE/SQUARE OFF this position — buying back what you sold and selling what "
                   "you bought, at current market prices. Review carefully, then confirm to send these real "
                   "orders to your Zerodha account."
    })


@app.route("/api/execute/<pos_id>/close/leg", methods=["POST"])
def close_single_leg(pos_id):
    """Places exactly ONE closing leg right now — the close-flow counterpart of
    /api/execute/<pos_id>/leg, for manually sequencing a square-off leg by leg."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    if not body.get("confirmed"):
        return jsonify({"error": "Confirmation flag not set — nothing was placed."}), 400
    order = body.get("order")
    if not order or not order.get("tradingsymbol"):
        return jsonify({"error": "No leg order provided."}), 400

    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404

    product = body.get("product", "NRML")
    order_type = body.get("order_type", "MARKET")
    results = place_basket_orders([order], product, order_type, sequence_for_margin=False)

    positions = load_positions()
    for p in positions:
        if p["id"] == pos_id:
            p["broker_orders"] = p.get("broker_orders", []) + results
    save_positions(positions)

    return jsonify({"results": results})


@app.route("/api/execute/<pos_id>/close/confirm", methods=["POST"])
def close_confirm(pos_id):
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
    custom_orders = body.get("orders")
    legs_to_place = custom_orders if custom_orders else build_close_orders(position)

    if not legs_to_place:
        return jsonify({"error": "No legs left to place — every leg was removed in the review screen."}), 400

    results = place_basket_orders(legs_to_place, product, order_type)

    # Estimate realized P&L per leg using the reference price captured at preview/placement time
    # (NOT a confirmed fill price — market orders execute asynchronously). Purely informational.
    entry_by_leg = {o["leg"]: o.get("entry_price") for o in legs_to_place}
    for r in results:
        if r["status"] != "placed":
            continue
        entry_price = entry_by_leg.get(r["leg"])
        close_price = next((o.get("reference_price") for o in legs_to_place if o["leg"] == r["leg"]), None)
        if entry_price is not None and close_price is not None:
            is_sell_originally = r["transaction_type"] == "BUY"  # closing a BUY means original leg was a SELL
            per_share = (entry_price - close_price) if is_sell_originally else (close_price - entry_price)
            r["estimated_realized_pnl"] = round(per_share * r["quantity"], 2)

    positions = load_positions()
    still_present = None
    for p in positions:
        if p["id"] == pos_id:
            p["broker_orders"] = p.get("broker_orders", []) + results
            still_present = p

    any_failed = any(r["status"] == "failed" for r in results)
    placed_count = sum(1 for r in results if r["status"] == "placed")
    total_legs = len(legs_to_place)
    fully_closed = placed_count == total_legs and not any_failed

    if fully_closed and still_present:
        archive_closed_position(still_present, results)
        positions = [p for p in positions if p["id"] != pos_id]
        note = (f"Position fully closed and archived to trade_history.json. "
                f"Estimated realized P&L: ₹{round(sum(r.get('estimated_realized_pnl', 0) for r in results), 2)} "
                f"(based on quoted prices at close, not confirmed fills — check your contract note).")
    elif any_failed and placed_count > 0:
        note = ("PARTIAL CLOSE: some legs closed, one failed. You may now hold a mismatched position. "
                "Open your Zerodha app / Kite web IMMEDIATELY to check and manually complete the close.")
    elif any_failed:
        note = "All legs failed — nothing was closed."
    else:
        note = f"All {placed_count}/{total_legs} legs placed to close this position. Verify fills in your Zerodha app."

    save_positions(positions)
    return jsonify({"results": results, "fully_closed": fully_closed, "note": note})



# ---------------------------------------------------------------------------
# Existing Position Intelligence — live Zerodha positions
# ---------------------------------------------------------------------------
def _position_instrument_map():
    """Map live option tradingsymbols to exchange/instrument metadata."""
    mp = {}
    try:
        nfo, _ = get_instruments()
        for i in nfo:
            if i.get("segment") == "NFO-OPT":
                mp[i.get("tradingsymbol")] = i
    except Exception:
        pass
    try:
        for i in get_bse_instruments():
            if i.get("segment") == "BFO-OPT":
                mp[i.get("tradingsymbol")] = i
    except Exception:
        pass
    return mp


def _option_quote_key(inst):
    ex = inst.get("exchange") or ("BFO" if inst.get("segment") == "BFO-OPT" else "NFO")
    return f"{ex}:{inst['tradingsymbol']}"


def _classify_position_group(legs):
    """Classify a same-underlying/same-expiry basket."""
    calls = [x for x in legs if x["type"] == "CE"]
    puts = [x for x in legs if x["type"] == "PE"]
    shorts = [x for x in legs if x["side"] == "SHORT"]
    longs = [x for x in legs if x["side"] == "LONG"]

    if len(legs) == 4 and len(calls) == 2 and len(puts) == 2 and len(shorts) == 2 and len(longs) == 2:
        return "IRON CONDOR"
    if len(legs) == 2 and len(calls) == 1 and len(puts) == 1 and len(shorts) == 2:
        return "SHORT STRANGLE"
    if len(legs) == 2 and len(calls) == 1 and len(puts) == 1 and len(longs) == 2:
        return "LONG STRADDLE"
    if len(legs) == 2 and ((len(calls) == 2) or (len(puts) == 2)):
        return "VERTICAL SPREAD"
    if len(legs) == 4:
        return "4-LEG / REVIEW"
    return "UNCLASSIFIED"


def _intraday_position_momentum(symbol):
    """5-minute live momentum snapshot used only for management context."""
    token, err = resolve_token_for_symbol(symbol)
    if err:
        return {"error": err}
    try:
        end = now_ist()
        start = end - timedelta(days=4)
        candles = kite.historical_data(token, start, end, "5minute")
    except Exception as e:
        return {"error": str(e)}
    if len(candles) < 25:
        return {"error": "Not enough 5-minute candles"}

    c = candles[-120:]
    closes = np.array([float(x["close"]) for x in c])
    highs = np.array([float(x["high"]) for x in c])
    lows = np.array([float(x["low"]) for x in c])
    vols = np.array([float(x.get("volume") or 0) for x in c])

    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    rsi = _rsi(closes, 14)
    adx = _adx(highs, lows, closes, 14)
    vwap = None
    if np.sum(vols) > 0:
        typical = (highs + lows + closes) / 3.0
        vwap = float(np.sum(typical * vols) / np.sum(vols))

    spot = float(closes[-1])
    ret5 = (spot / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
    ret30 = (spot / closes[-7] - 1) * 100 if len(closes) >= 7 else None
    ret60 = (spot / closes[-13] - 1) * 100 if len(closes) >= 13 else None
    avg_vol = float(np.mean(vols[-21:-1])) if len(vols) >= 22 and np.mean(vols[-21:-1]) > 0 else None
    rv = float(vols[-1] / avg_vol) if avg_vol else None

    direction = "BULLISH" if ((ema20 is not None and ema50 is not None and ema20 > ema50)
                               and (rsi is None or rsi >= 55)) else \
                "BEARISH" if ((ema20 is not None and ema50 is not None and ema20 < ema50)
                              and (rsi is None or rsi <= 45)) else "MIXED"

    strength = "STRONG" if adx is not None and adx >= 25 else "MODERATE" if adx is not None and adx >= 20 else "WEAK/RANGE"

    return {
        "spot": round(spot, 2), "ema20": round(ema20, 2) if ema20 is not None else None,
        "ema50": round(ema50, 2) if ema50 is not None else None,
        "rsi": round(rsi, 1) if rsi is not None else None,
        "adx": round(adx, 1) if adx is not None else None,
        "vwap": round(vwap, 2) if vwap is not None else None,
        "return_5m_pct": round(ret5, 3), "return_30m_pct": round(ret30, 3) if ret30 is not None else None,
        "return_60m_pct": round(ret60, 3) if ret60 is not None else None,
        "relative_volume": round(rv, 2) if rv is not None else None,
        "direction": direction, "strength": strength,
        "above_vwap": bool(vwap is not None and spot > vwap),
    }


def _position_management_action(group, momentum):
    """Transparent rule-based triage. It recommends a management action; it never places orders."""
    dte = group["dte"]
    pnl = group["pnl"]
    captured = group.get("profit_captured_pct")
    call_risk = group.get("call_risk_score", 0)
    put_risk = group.get("put_risk_score", 0)
    call_delta = abs(group.get("short_call_delta") or 0)
    put_delta = abs(group.get("short_put_delta") or 0)
    direction = momentum.get("direction") if momentum and not momentum.get("error") else "MIXED"
    strength = momentum.get("strength") if momentum and not momentum.get("error") else "UNKNOWN"

    reasons, actions = [], []
    threatened = None
    if group["strategy"] == "IRON CONDOR":
        if direction == "BULLISH" and call_risk >= put_risk:
            threatened = "CALL"
        elif direction == "BEARISH" and put_risk >= call_risk:
            threatened = "PUT"
        elif call_risk > put_risk * 1.25:
            threatened = "CALL"
        elif put_risk > call_risk * 1.25:
            threatened = "PUT"

    if captured is not None and captured >= 70 and dte <= 10:
        action, level = "CLOSE ENTIRE POSITION / TAKE PROFIT", "HIGH"
        reasons.append(f"{captured:.0f}% of estimated entry credit has been captured with only {dte} DTE.")
    elif group["strategy"] == "IRON CONDOR" and threatened and (
        (threatened == "CALL" and (call_delta >= 0.30 or call_risk >= 75)) or
        (threatened == "PUT" and (put_delta >= 0.30 or put_risk >= 75))
    ):
        action, level = f"ADJUST {threatened} SIDE — DO NOT WAIT FOR EXPIRY", "HIGH"
        reasons.append(f"{threatened} side is the threatened side based on spot distance and short-leg delta.")
        reasons.append(f"Short {threatened} delta is {call_delta if threatened=='CALL' else put_delta:.2f}.")
        if strength == "STRONG":
            reasons.append("Underlying momentum is strong, increasing breakout/gamma risk.")
    elif group["strategy"] == "IRON CONDOR" and dte <= 3 and (call_risk >= 55 or put_risk >= 55):
        action, level = "REDUCE RISK / CONSIDER FULL EXIT", "HIGH"
        reasons.append(f"{dte} DTE with a short strike under meaningful pressure.")
    elif group["strategy"] == "IRON CONDOR" and pnl < 0 and (call_risk >= 55 or put_risk >= 55):
        action, level = f"ADJUST THREATENED {threatened or 'SIDE'}", "MEDIUM"
        reasons.append("Position is losing while one side is becoming materially closer to the underlying.")
    elif group["strategy"] == "IRON CONDOR" and captured is not None and captured >= 50:
        action, level = "HOLD / PROTECT PROFIT", "LOW"
        reasons.append(f"{captured:.0f}% of estimated entry credit is captured and no severe side breach is detected.")
    else:
        action, level = "HOLD AND MONITOR", "LOW"
        reasons.append("No current rule-based trigger for adjustment or full exit.")

    if direction in ("BULLISH", "BEARISH"):
        reasons.append(f"Underlying is currently {direction.lower()} with {strength.lower()} momentum.")
    if momentum.get("above_vwap") is True:
        reasons.append("Price is above VWAP.")
    elif momentum.get("above_vwap") is False:
        reasons.append("Price is below VWAP.")

    return {"action": action, "level": level, "threatened_side": threatened,
            "reasons": reasons,
            "execution_note": "Recommendation only. Review live option-chain prices and margin before placing any adjustment."}


@app.route("/api/existing-position-analysis")
def existing_position_analysis():
    """Analyse every currently open Zerodha NFO/BFO option basket as a whole."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401

    try:
        raw = kite.positions().get("net", [])
        instruments = _position_instrument_map()
        rows = []
        for p in raw:
            if p.get("exchange") not in ("NFO", "BFO"):
                continue
            qty = int(p.get("quantity") or 0)
            ts = p.get("tradingsymbol")
            if qty == 0 or not ts or ts not in instruments:
                continue
            ins = instruments[ts]
            rows.append({
                "tradingsymbol": ts, "exchange": p.get("exchange"),
                "quantity": abs(qty), "side": "LONG" if qty > 0 else "SHORT",
                "entry_price": float(p.get("average_price") or 0),
                "ltp": float(p.get("last_price") or 0),
                "pnl": float(p.get("pnl") or 0),
                "type": ins.get("instrument_type"), "strike": float(ins.get("strike") or 0),
                "expiry": str(ins.get("expiry")), "underlying": ins.get("name"),
                "lot_size": int(ins.get("lot_size") or 1),
                "instrument": ins,
            })

        # Group by underlying + expiry. This intentionally groups the user's actual
        # open legs rather than relying on positions.json.
        grouped = {}
        for r in rows:
            key = (r["underlying"], r["expiry"])
            grouped.setdefault(key, []).append(r)

        analyses = []
        for (underlying, expiry), legs in grouped.items():
            strategy = _classify_position_group(legs)
            spot, spot_err = get_spot_price(underlying)
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = max((exp_date - now_ist().date()).days, 0)

            # Live quotes for executable exit prices and current leg Greeks.
            keys = [_option_quote_key(x["instrument"]) for x in legs]
            quotes = kite_quote_bulk(keys, force_refresh=True)
            short_call_delta = short_put_delta = None
            call_risk = put_risk = 0.0
            initial_credit_total = 0.0
            current_close_debit = 0.0

            for r in legs:
                q = quotes.get(_option_quote_key(r["instrument"])) or {}
                bid, ask = extract_bid_ask(q)
                r["bid"], r["ask"] = bid, ask
                exit_px = bid if r["side"] == "LONG" else ask
                r["exit_price"] = exit_px
                # Short entry contributes positive credit; long entry is debit.
                sign_credit = 1 if r["side"] == "SHORT" else -1
                initial_credit_total += sign_credit * r["entry_price"] * r["quantity"]
                if exit_px is not None:
                    # Cost to close a short = buy at ask; proceeds from closing long = sell at bid.
                    current_close_debit += (exit_px * r["quantity"]) * (1 if r["side"] == "SHORT" else -1)

            if initial_credit_total > 0:
                current_value_profit = initial_credit_total - current_close_debit
                captured = max(0.0, min(100.0, current_value_profit / initial_credit_total * 100))
            else:
                current_value_profit = sum(r["pnl"] for r in legs)
                captured = None

            if spot is not None:
                for r in legs:
                    T = max((exp_date - now_ist().date()).days, 0) / 365.0
                    ltp = r["ltp"]
                    iv = implied_vol(ltp, spot, r["strike"], T, r["type"]) if ltp and T > 0 else 0.25
                    delta = bs_delta(spot, r["strike"], T, RISK_FREE_RATE, iv, r["type"]) if T > 0 else (1.0 if ((r["type"]=="CE" and spot>r["strike"]) or (r["type"]=="PE" and spot<r["strike"])) else 0.0)
                    r["delta"] = round(float(delta), 3)
                    r["iv_pct"] = round(float(iv*100), 1)
                    if r["side"] == "SHORT" and r["type"] == "CE":
                        short_call_delta = r["delta"]
                    if r["side"] == "SHORT" and r["type"] == "PE":
                        short_put_delta = r["delta"]

            short_calls = [r for r in legs if r["side"]=="SHORT" and r["type"]=="CE"]
            short_puts = [r for r in legs if r["side"]=="SHORT" and r["type"]=="PE"]
            if spot is not None:
                if short_calls:
                    sc = short_calls[0]
                    call_risk = min(100.0, abs(spot-sc["strike"])/max(spot,1)*1000 + abs(short_call_delta or 0)*150)
                if short_puts:
                    sp = short_puts[0]
                    put_risk = min(100.0, abs(spot-sp["strike"])/max(spot,1)*1000 + abs(short_put_delta or 0)*150)
                # Distance is the better directional measure: closer strike = higher risk.
                if short_calls:
                    dist = (short_calls[0]["strike"]-spot)/spot*100
                    call_risk = min(100.0, max(0.0, 100.0 - dist*40) + abs(short_call_delta or 0)*50)
                if short_puts:
                    dist = (spot-short_puts[0]["strike"])/spot*100
                    put_risk = min(100.0, max(0.0, 100.0 - dist*40) + abs(short_put_delta or 0)*50)

            momentum = _intraday_position_momentum(underlying)
            health = 100.0
            if pnl := sum(r["pnl"] for r in legs):
                if pnl < 0: health -= min(30, abs(pnl)/max(abs(initial_credit_total), 1)*30)
            health -= min(30, max(call_risk, put_risk)*0.30)
            if dte <= 3: health -= 15
            elif dte <= 7: health -= 8
            if momentum.get("strength") == "STRONG": health -= 8
            health = round(max(0, min(100, health)), 0)

            group = {
                "underlying": underlying, "expiry": expiry, "dte": dte, "strategy": strategy,
                "legs": [{k:v for k,v in r.items() if k != "instrument"} for r in legs],
                "pnl": round(sum(r["pnl"] for r in legs), 2),
                "entry_credit_total": round(initial_credit_total, 2),
                "current_close_debit": round(current_close_debit, 2),
                "profit_captured_pct": round(captured, 1) if captured is not None else None,
                "spot": round(spot, 2) if spot is not None else None,
                "short_call_delta": short_call_delta, "short_put_delta": short_put_delta,
                "call_risk_score": round(call_risk, 1), "put_risk_score": round(put_risk, 1),
                "health_score": int(health), "momentum": momentum,
            }
            group["recommendation"] = _position_management_action(group, momentum)
            # Scenario distances to the short strikes.
            group["scenario"] = {}
            if spot is not None:
                for pct in (-2,-1,-0.5,0.5,1,2):
                    s = spot*(1+pct/100)
                    scenario_pnl = 0.0
                    T = max(dte,0)/365.0
                    for r in legs:
                        iv = (r.get("iv_pct") or 20)/100
                        theo = bs_price(s,r["strike"],T,RISK_FREE_RATE,iv,r["type"]) if T>0 else max((s-r["strike"]) if r["type"]=="CE" else (r["strike"]-s),0)
                        # mark position to theoretical option value
                        scenario_pnl += (theo-r["entry_price"]) * r["quantity"] * (1 if r["side"]=="LONG" else -1)
                    group["scenario"][f"{pct:+g}%"] = round(scenario_pnl,2)
            analyses.append(group)

        analyses.sort(key=lambda x: x["health_score"])
        return jsonify({
            "positions": analyses,
            "count": len(analyses),
            "refreshed_at": now_ist().strftime("%H:%M:%S"),
            "note": "Rule-based live position management. Recommendations are decision support, not automatic orders or guarantees. Greeks/IV are model estimates."
        })
    except Exception as e:
        logger.exception("Existing position analysis failed")
        return jsonify({"error": str(e)}), 400


@app.route("/api/broker-positions")
def broker_positions():
    """Live F&O positions straight from your Zerodha account (Kite's net positions() call) —
    independent of this tool's own tracked Iron Condor / Strangle baskets in positions.json, and
    independent of which strategy or basket a leg originally came from. For each open NFO leg this
    returns the entry (average) price, live LTP, and running P&L reported by Kite itself, so the
    Order Management tab can show exactly what your account currently holds and let you price and
    fire an exit — for one leg or several at once — straight from here."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        pos = kite.positions()
        net = pos.get("net", [])
        rows = []
        open_rows = []
        for p in net:
            if p.get("exchange") != "NFO":
                continue
            qty = int(p.get("quantity") or 0)
            if qty == 0:
                continue  # already flat — nothing open on this tradingsymbol
            row = {
                "tradingsymbol": p.get("tradingsymbol"),
                "product": p.get("product"),
                "quantity": qty,
                "side": "LONG" if qty > 0 else "SHORT",
                "average_price": p.get("average_price"),
                "last_price": p.get("last_price"),
                "pnl": p.get("pnl"),
                "close_price": p.get("close_price"),
                "bid": None,
                "ask": None,
                "exit_price": None,
                "exit_price_basis": None,
            }
            rows.append(row)
            if p.get("tradingsymbol"):
                open_rows.append(row)

        # Fetch LIVE market depth for every open NFO leg.  Do not use the position
        # response's close_price as Bid/Ask: it is not the current executable quote.
        # A LONG position is closed with SELL at Bid; a SHORT position is closed
        # with BUY at Ask.  This is also what the frontend uses for the displayed
        # immediately-executable P&L.
        if open_rows:
            quote_keys = [f"NFO:{r['tradingsymbol']}" for r in open_rows]
            quotes = kite_quote_bulk(quote_keys, force_refresh=True)
            for r in open_rows:
                q = quotes.get(f"NFO:{r['tradingsymbol']}") or {}
                bid, ask = extract_bid_ask(q)
                r["bid"] = bid
                r["ask"] = ask
                if r["side"] == "LONG":
                    r["exit_price"] = bid
                    r["exit_price_basis"] = "BID"
                else:
                    r["exit_price"] = ask
                    r["exit_price_basis"] = "ASK"

        return jsonify({"positions": rows, "refreshed_at": now_ist().strftime("%H:%M:%S")})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/broker-positions/exit", methods=["POST"])
def broker_positions_exit():
    """Squares off one or more live Zerodha F&O positions directly by tradingsymbol — backs the
    Order Management tab's per-row 'Exit' button and the 'Exit Selected' multi-select action.
    Independent of this tool's own tracked baskets; works on whatever legs you pick, in whatever
    combination. A LONG position is squared off with a SELL, a SHORT position with a BUY. Each leg
    can optionally carry its own exit price (LIMIT) — legs left blank use MARKET. These legs are NOT
    run through the BUY-before-SELL basket sequencing (see place_basket_orders) since they're
    independent square-offs you chose yourself, not a hedged multi-leg entry."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    if not body.get("confirmed"):
        return jsonify({"error": "Confirmation flag not set — nothing was placed."}), 400
    legs = body.get("legs")
    if not legs:
        return jsonify({"error": "No legs provided."}), 400

    product = body.get("product", "NRML")

    legs_to_place = []
    for lg in legs:
        if not lg.get("tradingsymbol"):
            continue
        qty = abs(int(lg.get("quantity") or 0))
        if qty <= 0:
            continue
        side = str(lg.get("side", "LONG")).upper()
        close_txn = "SELL" if side == "LONG" else "BUY"
        price = lg.get("price")
        legs_to_place.append({
            "leg": lg["tradingsymbol"], "tradingsymbol": lg["tradingsymbol"],
            "transaction_type": close_txn, "quantity": qty,
            "price": float(price) if price not in (None, "") else None,
            "_original_side": side,
        })

    if not legs_to_place:
        return jsonify({"error": "No valid legs to place."}), 400

    # Zerodha market orders are not used by this exit path.  For every leg whose
    # custom price is blank, fetch a FRESH quote immediately before placement:
    #   LONG -> SELL at best Bid
    #   SHORT -> BUY at best Ask
    # These are marketable LIMIT orders and are intended to execute immediately
    # at the current executable side, subject to the quote still being available
    # when the order reaches the exchange.
    inst_keys = [f"NFO:{lg['tradingsymbol']}" for lg in legs_to_place]
    try:
        quotes = kite_quote_bulk(inst_keys, force_refresh=True)
    except Exception as e:
        return jsonify({"error": f"Could not fetch live Bid/Ask for exit: {e}"}), 502

    missing = []
    for lg in legs_to_place:
        if lg["price"] is not None:
            continue  # user explicitly supplied a custom LIMIT price
        q = quotes.get(f"NFO:{lg['tradingsymbol']}") or {}
        bid, ask = extract_bid_ask(q)
        auto_price = bid if lg["_original_side"] == "LONG" else ask
        if auto_price is None:
            missing.append(lg["tradingsymbol"])
        else:
            lg["price"] = auto_price

    if missing:
        return jsonify({
            "error": "Live Bid/Ask unavailable for: " + ", ".join(missing) +
                     ". No exit orders were placed. Refresh positions and try again."
        }), 502

    for lg in legs_to_place:
        lg.pop("_original_side", None)

    # Always LIMIT here.  Blank custom prices are automatically converted to the
    # correct marketable Bid/Ask price above.
    order_type = "LIMIT"
    results = place_basket_orders(legs_to_place, product, order_type, sequence_for_margin=False)
    return jsonify({
        "results": results,
        "order_type": order_type,
        "note": "Auto-priced exits use fresh Bid for LONG positions and fresh Ask for SHORT positions."
    })


@app.route("/api/broker/account")
def broker_account():
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    out={}
    try: out["profile"]=kite.profile()
    except Exception as e: out["profile_error"]=str(e)
    try: out["margins"]=kite.margins()
    except Exception as e: out["margins_error"]=str(e)
    try: out["holdings"]=kite.holdings()
    except Exception as e: out["holdings_error"]=str(e)
    try: out["positions"]=kite.positions()
    except Exception as e: out["positions_error"]=str(e)
    return jsonify(out)

@app.route("/api/quote/<path:instrument>")
def broker_quote(instrument):
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    try:
        key=instrument if ":" in instrument else f"NSE:{instrument.upper()}"
        return jsonify({"quote":kite_quote_bulk([key]).get(key)})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/trades")
def broker_trades():
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    try: return jsonify({"trades":kite.trades()})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/charges/orders", methods=["POST"])
def broker_charges_orders():
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    body=request.json or {}; orders=body.get("orders") or []
    if not orders: return jsonify({"error":"orders required"}),400
    try:
        if hasattr(kite,"order_charges"):
            return jsonify({"charges":kite.order_charges(orders)})
        return jsonify({"charges":None,"note":"Installed kiteconnect SDK does not expose order_charges; use estimate_charges locally."})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/gtt")
def gtt_list():
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    try: return jsonify({"triggers":kite.get_gtts()})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/gtt", methods=["POST"])
def gtt_create():
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    body=request.json or {}
    try: return jsonify({"trigger_id":kite.place_gtt(body["trigger_type"],body["condition"],body["orders"])})
    except Exception as e: return jsonify({"error":str(e)}),400

@app.route("/api/gtt/<int:trigger_id>", methods=["PUT","DELETE"])
def gtt_manage(trigger_id):
    if not require_session(): return jsonify({"error":"not_logged_in"}),401
    try:
        if request.method=="DELETE": return jsonify({"ok":kite.delete_gtt(trigger_id)})
        body=request.json or {}; return jsonify({"ok":kite.modify_gtt(trigger_id,body.get("condition"),body.get("orders"))})
    except Exception as e: return jsonify({"error":str(e)}),400

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

    to_date = now_ist()
    from_date = to_date - timedelta(days=days + (15 if interval == "day" else 5))
    candles = kite.historical_data(token, from_date, to_date, interval)

    if interval != "day":
        # The window above is padded (weekends/holidays could otherwise leave fewer trading
        # sessions than requested), so it can return MORE trading days than `days` asked for.
        # Trim down to exactly the most recent `days` trading days so a short lookback (e.g. 1
        # day) doesn't silently keep showing extra earlier sessions -- which is what made the
        # chart look like it "wasn't updating" when you changed the lookback control.
        by_day = {}
        for c in candles:
            d = c["date"]
            key = d.date() if hasattr(d, "date") else str(d)[:10]
            by_day.setdefault(key, []).append(c)
        wanted_days = sorted(by_day.keys())[-days:]
        candles = [c for day_key in wanted_days for c in by_day[day_key]]

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
def _get_headlines_best_effort(symbol, max_items=3):
    """Shared by the screener (top-N picks) and /api/news/<symbol>. Best-effort keyword scan
    of public RSS feeds — NOT sentiment analysis, NOT a verified event-risk signal. Returns
    (list_of_headline_dicts, error_string_or_None)."""
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
                return headlines[:max_items], None
            errors.append(f"{name} returned no items")
        except requests.exceptions.SSLError:
            ssl_issue_seen = True
            errors.append(f"{name}: SSL certificate verification failed")
            if ALLOW_INSECURE_NEWS:
                try:
                    headlines = _fetch_rss(url, headers, verify=False)
                    if headlines:
                        return headlines[:max_items], None
                except Exception as e2:
                    errors.append(f"{name} (insecure retry) also failed: {e2}")
        except Exception as e:
            errors.append(f"{name} failed: {e}")

    guidance = ""
    if ssl_issue_seen:
        guidance = (" This looks like a network TLS-interception issue (corporate/government firewall) "
                     "rather than a real absence of news — see /api/news/<symbol> for the full explanation.")
    return [], "Could not fetch headlines. " + " | ".join(errors) + guidance


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
    headlines, error = _get_headlines_best_effort(symbol, max_items=5)
    if headlines:
        return jsonify({"symbol": symbol, "headlines": headlines,
                         "note": "Best-effort headline scan, not verified analysis. "
                                 "Read the actual articles before treating this as an event-risk signal."})

    guidance = ""
    if error and "SSL certificate verification failed" in error:
        guidance = (" This looks like your network (office/government firewall, antivirus, or a proxy) is "
                     "intercepting HTTPS traffic with its own certificate — common on corporate/government "
                     "networks. Kite API calls aren't affected since those go through Kite's own SDK. To fix "
                     "properly: ask your IT team for the organization's root CA certificate and set it via the "
                     "REQUESTS_CA_BUNDLE environment variable. As a quick workaround for this headlines feature "
                     "only (not recommended on untrusted networks), you can set ALLOW_INSECURE_NEWS=true as an "
                     "environment variable before running backend.py.")
    return jsonify({"symbol": symbol, "headlines": [],
                     "error": "Could not fetch news from any source. " + (error or "") + guidance})


# ---------------------------------------------------------------------------
# AUTO TRADE MODE — rule-based intraday breakout scanner + option buyer
# ---------------------------------------------------------------------------
# What this actually is, in plain terms:
#   - It is NOT an AI making trading judgements. It is a simple, fully-visible technical rule:
#     price breaks above/below its recent N-candle range (a Donchian-channel breakout), with a
#     minimum move size in ATR units and (best-effort) volume confirmation.
#   - On a qualifying breakout it buys the ATM option in the breakout direction (CE for an
#     upside break, PE for a downside break) using MIS (intraday) product, sized off a rupee
#     budget you set, then manages that ONE position with a premium-based stop-loss, a
#     premium-based target, and a trailing stop once it's in profit — until SL/target/trailing
#     stop hits, or the end-of-day square-off time, whichever comes first.
#   - MANUAL mode: it scans continuously and shows you the ranked signal — nothing is ever sent
#     to your broker until you click "Execute this signal".
#   - AUTO mode: after you click OK on the confirmation dialog, it places that entry order the
#     moment a qualifying signal appears, with no per-trade click. This is real money, unattended.
#     Hard safety rails below (daily trade cap, daily loss cap, single-position-at-a-time, kill
#     switch) exist specifically because of that — they are not optional and cannot be disabled
#     from the UI.
#   - Universe: either a hand-picked list, OR "scan_all_fo" mode, which rotates through the ENTIRE
#     live F&O stock list (indices always included) a batch at a time, staying under Kite's
#     historical-data rate limit -- see _effective_scan_universe().
#   - Signal model: Donchian-channel breakout scored with THREE confirmation layers -- EMA9/21
#     trend agreement, RSI(14) momentum, and continuous relative volume -- combined into one
#     composite score. Still fully rules-based and transparent (see _detect_breakout()); it is
#     NOT a proven or guaranteed-profitable strategy, and false breakouts remain one of the most
#     common ways to lose money in intraday trading. Past behavior of this logic on any symbol
#     says nothing about future results. Only ever risk capital you can afford to lose, and watch
#     your Zerodha app while Auto mode is armed.
# ---------------------------------------------------------------------------
AUTOTRADE_FILE = os.path.join(os.path.dirname(__file__), "autotrade_state.json")
AUTOTRADE_TRADES_FILE = os.path.join(os.path.dirname(__file__), "autotrade_trades.json")
_autotrade_lock = threading.Lock()

AUTOTRADE_MARKET_OPEN = "09:20"   # a few minutes after the 9:15 open, letting the opening range settle

AUTOTRADE_DEFAULTS = {
    "enabled": False,                 # is the scan/execute loop armed at all
    "mode": "manual",                 # "manual" (show signal, wait for click) or "auto" (self-execute)
    # --- Order execution: Track (paper) vs Live (real orders) ---
    # Orthogonal to `mode` above -- applies to BOTH Manual and Auto scanning.
    #   "track" (DEFAULT): NO real orders are ever sent to your broker. Entries and exits are
    #             simulated fills at the same real, live LTP a live trade would have used, so P&L,
    #             SL/target/trailing, and the auto-detected exit logic all run exactly as they would
    #             live -- just with paper money. Safe to leave scanning/armed indefinitely.
    #   "live"  : real orders are placed on your live Zerodha account, same as before this setting
    #             existed. Can only be turned on via /api/autotrade/set-execution-mode with an
    #             explicit ack -- NOT changeable through the bulk /api/autotrade/config save, so a
    #             stray "Save settings" click can never silently flip you into real-money trading.
    # Each trade snapshots which mode it was opened under (trade["execution_mode"]), so switching
    # this setting never changes how an already-open position behaves.
    "execution_mode": "track",
    "universe": ["NIFTY", "BANKNIFTY", "FINNIFTY"],   # fallback list used only if scan_all_fo is OFF
    "scan_all_fo": True,              # DEFAULT: scan the ENTIRE live F&O stock universe (every
                                       # NFO-OPT name, indices included) instead of a hand-picked list
                                       # -- see _effective_scan_universe() for how this is rate-limited.
                                       # Untick in Settings to fall back to the manual `universe` list.
    "_fo_scan_cursor": 0,             # internal: rotation position through the full F&O list
    "candle_interval": "5minute",
    "breakout_lookback": 20,          # candles in the Donchian channel
    "poll_seconds": 20,
    "max_trades_per_day": 3,
    "max_concurrent_positions": 1,    # how many auto-trade positions can be open AT ONCE
    "capital_per_trade": 15000,       # approx premium budget (Rs) used to size lots
    "max_daily_loss": 5000,           # Rs; auto-disarms Auto mode the instant realized loss hits this
    # --- Exit logic: choose how open auto-trades get closed ---
    # "auto"      (DEFAULT): system auto-detected exit -- the position is closed the moment the
    #             underlying's own trend/momentum invalidates the breakout thesis that opened it
    #             (EMA9/21 flips + RSI disagrees), OR a trailing-stop gives back too much of the
    #             peak profit reached, OR the hard safety-stop floor is hit, OR EOD square-off.
    #             There is no fixed profit cap in this mode -- winners are allowed to run as long as
    #             the trend holds. A hard percentage safety floor (hard_stop_pct) is ALWAYS active
    #             underneath this, regardless of what the reversal/trailing logic is doing.
    # "target_sl" : fixed, user-defined Target % / Stop-loss % (+ trailing) exactly as configured
    #             below -- the position exits purely on those numbers, no trend re-evaluation.
    "exit_mode": "auto",
    "hard_stop_pct": 50,               # Auto exit mode ONLY: catastrophic-loss floor, % of premium
                                        # paid -- exits immediately no matter what the reversal/
                                        # trailing logic says. This is a safety net, not the primary
                                        # exit trigger in Auto mode.
    "sl_mode": "pct",                 # "pct" (of entry premium) or "points" (flat rupees off premium)
    "sl_pct_of_premium": 30,          # stop-loss = premium paid minus this %   } used when
    "sl_points": 5.0,                 # stop-loss = premium paid minus this many rupees (sl_mode=points)  } exit_mode
    "target_mode": "pct",             # "pct" (of entry premium) or "points" (flat rupees over premium)  } ==
    "target_pct_of_premium": 60,      # target = premium paid plus this %      } "target_sl"
    "target_points": 10.0,            # target = premium paid plus this many rupees (if target_mode=points)
    "trail_after_pct": 30,            # once profit reaches this %, switch to trailing-stop mode
                                       # (used by BOTH exit modes -- peak-profit tracking is shared)
    "trail_giveback_pct": 15,         # trailing stop = peak-profit% minus this many percentage points
    "min_breakout_score": 0.5,        # minimum breakout size, in ATR multiples, to qualify as a signal
    "strict_breakout_filters": True,  # DEFAULT ON: extra false-breakout confirmation layers -- see
                                       # _detect_breakout() (strong-close filter, consolidation/
                                       # tightness check, minimum ATR floor, whipsaw/failed-breakout
                                       # penalty, higher volume bar). Untick to fall back to the
                                       # looser original breakout+3-confirmation model.
    "square_off_time": "15:15",       # force-exit any open auto-trade by this time regardless of SL/target
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "day": None,
    "last_scan_at": None,
    "last_scan_candidates": [],
    "last_error": None,
    "disarm_reason": None,
}

CONFIGURABLE_AUTOTRADE_KEYS = (
    "universe", "scan_all_fo", "candle_interval", "breakout_lookback", "poll_seconds",
    "max_trades_per_day", "max_concurrent_positions", "capital_per_trade", "max_daily_loss",
    "exit_mode", "hard_stop_pct",
    "sl_mode", "sl_pct_of_premium", "sl_points", "target_mode", "target_pct_of_premium", "target_points",
    "trail_after_pct", "trail_giveback_pct", "min_breakout_score", "strict_breakout_filters",
    "square_off_time",
)

# --- "Scan all F&O stocks" mode ---
# Kite's historical-data endpoint is rate-limited to roughly 3 requests/second. The live F&O stock
# list is typically ~180-220 names, so scanning all of them in one go takes well over a minute and
# would blow through that limit if it were retried every `poll_seconds`. Instead of scanning
# everything every cycle, the loop rotates through the full list a chunk at a time (covering it all
# every few minutes) and calls to Kite are staggered. The background loop also has a floor on how
# often it's allowed to run while this mode is on.
FO_SCAN_CHUNK_SIZE = 40            # symbols scanned per background poll cycle when scan_all_fo is on
FO_SCAN_MIN_POLL_SECONDS = 45      # floor on poll_seconds while scan_all_fo is on
HISTORICAL_CALL_STAGGER_SECONDS = 0.35   # ~2.8 req/sec between historical-data calls, under Kite's cap


def _effective_scan_universe(state):
    """Returns the symbols to scan THIS cycle. If scan_all_fo is off, that's just the configured
    `universe`. If it's on, rotates through the full live F&O stock list (indices always included)
    in FO_SCAN_CHUNK_SIZE-sized slices, advancing the cursor stored in state each call, so the
    entire universe gets covered progressively across consecutive cycles rather than all at once."""
    if not state.get("scan_all_fo"):
        return state.get("universe") or list(AUTOTRADE_DEFAULTS["universe"])
    try:
        full = list(INDEX_SYMBOLS.keys()) + fo_stock_universe()
    except Exception:
        return state.get("universe") or list(AUTOTRADE_DEFAULTS["universe"])
    if not full:
        return state.get("universe") or list(AUTOTRADE_DEFAULTS["universe"])
    cursor = state.get("_fo_scan_cursor", 0) % len(full)
    rotated = full[cursor:] + full[:cursor]
    chunk = rotated[:FO_SCAN_CHUNK_SIZE]
    state["_fo_scan_cursor"] = (cursor + FO_SCAN_CHUNK_SIZE) % len(full)
    return chunk


def load_autotrade_state():
    state = dict(AUTOTRADE_DEFAULTS)
    if os.path.exists(AUTOTRADE_FILE):
        try:
            with open(AUTOTRADE_FILE, "r") as f:
                state.update(json.load(f))
        except Exception:
            pass
    return state


def save_autotrade_state(state):
    with _autotrade_lock:
        with open(AUTOTRADE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)


def load_autotrade_trades():
    if not os.path.exists(AUTOTRADE_TRADES_FILE):
        return []
    try:
        with open(AUTOTRADE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_autotrade_trades(trades):
    with _autotrade_lock:
        with open(AUTOTRADE_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)


def _autotrade_roll_day_if_needed(state):
    """Resets the daily trade counter / P&L exactly once per calendar day. Does NOT touch any
    currently-open trade -- that's handled by the end-of-day square-off check in the monitor loop."""
    today_str = now_ist().strftime("%Y-%m-%d")
    if state.get("day") != today_str:
        state["day"] = today_str
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["disarm_reason"] = None
    return state


def _fetch_recent_intraday(symbol, interval, lookback):
    token, err = resolve_token_for_symbol(symbol)
    if err:
        return None, err
    to_date = now_ist()
    from_date = to_date - timedelta(days=7)  # a bit more history so EMA21/RSI14 have enough bars
    try:
        candles = kite.historical_data(token, from_date, to_date, interval)
    except Exception as e:
        return None, str(e)
    min_needed = max(lookback + 5, 30)
    if len(candles) < min_needed:
        return None, "Not enough intraday candles yet today for a reliable channel"
    return candles, None


def _ema(values, period):
    """Simple exponential moving average over `values` (oldest-first), seeded with the first value."""
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema


def _rsi(closes, period=14):
    """Wilder-style RSI (simple-average variant) over the trailing `period` bars."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain, avg_loss = float(np.mean(gains)), float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1 + rs))


# ---------------------------------------------------------------------------
# NIFTY Bullish Breakout Quality Engine
# ---------------------------------------------------------------------------
NIFTY_BREAKOUT_DEFAULT_LOOKBACK = 20
NIFTY_BREAKOUT_VOLUME_MULTIPLE = 1.5


def _latest_session_candles(candles):
    if not candles:
        return []
    last_dt = candles[-1].get("date")
    last_day = last_dt.date() if hasattr(last_dt, "date") else str(last_dt)[:10]
    return [c for c in candles
            if (c.get("date").date() if hasattr(c.get("date"), "date") else str(c.get("date"))[:10]) == last_day]


def _session_vwap(candles):
    session = _latest_session_candles(candles)
    if not session:
        return None
    pv = sum(((float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0) * float(c.get("volume", 0) or 0) for c in session)
    vol = sum(float(c.get("volume", 0) or 0) for c in session)
    return pv / vol if vol > 0 else None


def _nifty_bullish_breakout(candles, option_chain_data=None, lookback=20):
    """Six-condition, read-only NIFTY bullish breakout classifier."""
    if len(candles) < max(lookback + 2, 55):
        return None, "Not enough NIFTY candles for EMA20/EMA50 and breakout analysis."
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    volumes = np.array([float(c.get("volume", 0) or 0) for c in candles], dtype=float)
    spot = float(closes[-1])
    vwap = _session_vwap(candles)
    ema20 = _ema(closes[-min(len(closes), 120):], 20)
    ema50 = _ema(closes[-min(len(closes), 120):], 50)
    rsi = _rsi(closes, 14)
    resistance = float(np.max(highs[-(lookback + 1):-1]))
    avg_vol = float(np.mean(volumes[-(lookback + 1):-1]))
    rel_volume = float(volumes[-1] / avg_vol) if avg_vol > 0 else None
    true_ranges = np.maximum(highs[1:] - np.array([float(c["low"]) for c in candles[1:]]),
                              np.maximum(np.abs(highs[1:] - closes[:-1]),
                                         np.abs(np.array([float(c["low"]) for c in candles[1:]]) - closes[:-1])))
    atr = float(np.mean(true_ranges[-lookback:])) if len(true_ranges) >= lookback else float(np.mean(true_ranges))
    breakout_distance = spot - resistance
    conditions = {
        "bullish_breakout": breakout_distance >= max(0.0, 0.15 * atr),
        "above_vwap": vwap is not None and spot > vwap,
        "ema20_above_ema50": ema20 is not None and ema50 is not None and ema20 > ema50,
        "rsi_above_55": rsi is not None and rsi > 55,
        "volume_expansion": rel_volume is not None and rel_volume >= NIFTY_BREAKOUT_VOLUME_MULTIPLE,
        "resistance_broken": spot > resistance,
        "call_side_momentum": False,
    }
    ce_momentum = pe_momentum = None
    ce_volume = pe_volume = None
    atm_strike = option_expiry = None
    if option_chain_data:
        chain = option_chain_data.get("chain") or []
        chain_spot = float(option_chain_data.get("spot") or spot)
        ce_opts = [o for o in chain if o.get("instrument_type") == "CE" and o.get("instrument_token")]
        pe_opts = [o for o in chain if o.get("instrument_type") == "PE" and o.get("instrument_token")]
        ce_closes, pe_closes = [], []
        if ce_opts:
            ce_atm = min(ce_opts, key=lambda o: abs(float(o["strike"]) - chain_spot))
            atm_strike, option_expiry = ce_atm.get("strike"), option_chain_data.get("expiry")
            ce_volume = float(ce_atm.get("volume") or 0)
            try:
                ce_hist = kite.historical_data(int(ce_atm["instrument_token"]), now_ist() - timedelta(days=3), now_ist(), "5minute")
                ce_closes = [float(c["close"]) for c in ce_hist if c.get("close") is not None]
                if len(ce_closes) >= 4:
                    ce_momentum = ce_closes[-1] > ce_closes[-2] and ce_closes[-1] > ce_closes[-4]
            except Exception as e:
                logger.warning("NIFTY CE momentum fetch failed: %s", e)
            if pe_opts:
                pe_atm = min(pe_opts, key=lambda o: abs(float(o["strike"]) - chain_spot))
                pe_volume = float(pe_atm.get("volume") or 0)
                try:
                    pe_hist = kite.historical_data(int(pe_atm["instrument_token"]), now_ist() - timedelta(days=3), now_ist(), "5minute")
                    pe_closes = [float(c["close"]) for c in pe_hist if c.get("close") is not None]
                    if len(pe_closes) >= 4:
                        pe_momentum = pe_closes[-1] > pe_closes[-2] and pe_closes[-1] > pe_closes[-4]
                except Exception as e:
                    logger.warning("NIFTY PE momentum fetch failed: %s", e)
            ce_change = (ce_closes[-1] / ce_closes[-2] - 1) if len(ce_closes) >= 2 else None
            pe_change = (pe_closes[-1] / pe_closes[-2] - 1) if len(pe_closes) >= 2 else None
            conditions["call_side_momentum"] = bool(ce_momentum and
                (pe_change is None or ce_change is None or ce_change > pe_change) and
                (ce_volume == 0 or pe_volume is None or ce_volume >= pe_volume))

    score = sum(1 for v in conditions.values() if v)
    quality = "HIGH QUALITY CE SETUP" if score == 7 else ("WATCH — 5/7 or 6/7" if score >= 5 else "NO CE SETUP")
    return {
        "symbol": "NIFTY", "direction": "CE", "score": score, "max_score": 6, "quality": quality,
        "spot": round(spot, 2), "vwap": round(vwap, 2) if vwap is not None else None,
        "ema20": round(ema20, 2) if ema20 is not None else None,
        "ema50": round(ema50, 2) if ema50 is not None else None,
        "rsi": round(rsi, 1) if rsi is not None else None, "resistance": round(resistance, 2),
        "atr": round(atr, 2), "breakout_distance": round(breakout_distance, 2),
        "rel_volume": round(rel_volume, 2) if rel_volume is not None else None,
        "conditions": conditions, "ce_momentum": ce_momentum, "pe_momentum": pe_momentum,
        "ce_volume": ce_volume, "pe_volume": pe_volume, "atm_strike": atm_strike,
        "option_expiry": str(option_expiry) if option_expiry else None, "checked_at": now_ist().isoformat(),
        "note": "Rule-based setup score only; not a probability of profit or trading recommendation.",
    }, None


# --- False-breakout guards (active whenever strict=True, i.e. state["strict_breakout_filters"]) ---
# Intraday breakouts fail (whipsaw back into the range) very often; these thresholds exist
# specifically to filter out the weakest, least-reliable-looking ones before they ever become a
# candidate -- on top of the existing trend/momentum/volume confirmation layers.
MIN_BREAKOUT_ATR_FLOOR = 0.15        # raw move beyond the level, in ATR, below which it's just noise
CHANNEL_TIGHTNESS_MAX_RATIO = 6.0    # if the pre-breakout range is already wider than this many ATRs,
                                      # there's no real consolidation to break OUT of -- more likely
                                      # an already-choppy/trending stock than a clean range break
STRONG_CLOSE_MIN_RATIO = 0.6         # the breakout candle must close in the outer 40% of its own
                                      # high-low range, in the breakout direction -- a close near the
                                      # middle (or worse, back toward the opposite end) is the classic
                                      # tell of a wick-and-reject false breakout
VOLUME_CONFIRM_MULTIPLE = 1.5        # raised from a looser 1.2x -- demands real participation behind
                                      # the move, not just "any" above-average print
WHIPSAW_LOOKBACK_EXTRA_BARS = 10     # how many extra bars (beyond the channel itself) to scan for
                                      # recent failed pokes at this same level
WHIPSAW_PENALTY_PER_FAILURE = 0.25   # composite score is knocked down this fraction per recent
                                      # failed breakout found at the same level (capped, see below)


def _detect_breakout(candles, lookback, strict=True):
    """Donchian-channel breakout PLUS confirmation layers, combined into one transparent composite
    score (still fully rules-based -- every input is echoed in `reasoning`, no black box):

      1. Breakout size    -- how far the close is outside the prior `lookback`-candle range, in ATR.
      2. Trend alignment  -- EMA9 vs EMA21: does the breakout agree with the prevailing short-term
                             trend, or is it a counter-trend poke that's more likely to fail?
      3. Momentum         -- RSI(14): is momentum actually pushing the same direction as the break?
      4. Relative volume  -- today's bar's volume vs the recent average, scaled continuously instead
                             of a blunt above/below-average flag.

    When `strict` is True (the default -- state["strict_breakout_filters"]), four extra false-
    breakout guards are applied BEFORE anything is even scored:
      5. Minimum breakout size floor (MIN_BREAKOUT_ATR_FLOOR) -- rejects marginal, noise-level pokes.
      6. Consolidation/tightness check (CHANNEL_TIGHTNESS_MAX_RATIO) -- rejects "breakouts" out of a
         range that was never actually tight to begin with (already trending/choppy).
      7. Strong-close filter (STRONG_CLOSE_MIN_RATIO) -- rejects a breakout candle that closed back
         near the middle of its own range (weak, indecisive, wick-heavy -- a classic failed-breakout
         candle shape).
      8. Whipsaw/failed-breakout penalty -- discounts the score if this same level has already been
         poked through and rejected recently today.
    Volume confirmation is also raised to a stricter multiple (VOLUME_CONFIRM_MULTIPLE) under strict
    mode. Returns None if there's no qualifying breakout at all.
    """
    highs = np.array([c["high"] for c in candles], dtype=float)
    lows = np.array([c["low"] for c in candles], dtype=float)
    closes = np.array([c["close"] for c in candles], dtype=float)
    opens = np.array([c.get("open", c["close"]) for c in candles], dtype=float)
    volumes = np.array([c.get("volume", 0) or 0 for c in candles], dtype=float)

    window_highs = highs[-(lookback + 1):-1]
    window_lows = lows[-(lookback + 1):-1]
    channel_high = float(np.max(window_highs))
    channel_low = float(np.min(window_lows))
    last_close = float(closes[-1])
    last_high = float(highs[-1])
    last_low = float(lows[-1])

    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = float(np.mean(tr[-lookback:])) if len(tr) >= lookback else (float(np.mean(tr)) if len(tr) else 0.0)
    if atr <= 0:
        return None

    # Consolidation/tightness guard -- skip breakouts out of a range that was already too wide to be
    # a real base (this is checked before we even know direction, since it's a property of the range).
    channel_width = channel_high - channel_low
    if strict and channel_width / atr > CHANNEL_TIGHTNESS_MAX_RATIO:
        return None

    prior_vol = volumes[-(lookback + 1):-1]
    avg_vol = float(np.mean(prior_vol)) if prior_vol.size else 0.0
    last_vol = float(volumes[-1])
    rel_volume = round(last_vol / avg_vol, 2) if avg_vol > 0 else None
    vol_multiple = VOLUME_CONFIRM_MULTIPLE if strict else 1.2
    vol_confirmed = (avg_vol == 0) or (last_vol >= avg_vol * vol_multiple)

    direction, breakout_level = None, None
    if last_close > channel_high:
        direction, breakout_level = "CE", channel_high
    elif last_close < channel_low:
        direction, breakout_level = "PE", channel_low
    if not direction:
        return None

    breakout_size_score = abs(last_close - breakout_level) / atr
    if strict and breakout_size_score < MIN_BREAKOUT_ATR_FLOOR:
        return None  # the move beyond the level is still within noise -- not a real breakout yet

    # Strong-close filter: reject a breakout candle that closed back toward the middle/wrong end of
    # its own high-low range -- a common false-breakout / stop-hunt wick signature.
    candle_range = max(last_high - last_low, 1e-9)
    if direction == "CE":
        close_position = (last_close - last_low) / candle_range
    else:
        close_position = (last_high - last_close) / candle_range
    strong_close = close_position >= STRONG_CLOSE_MIN_RATIO
    if strict and not strong_close:
        return None

    # Whipsaw guard: count recent bars (just before this one) that already poked through this same
    # level and then closed back inside it -- a level that's failed repeatedly today is less trustworthy.
    scan_from = -(lookback + 1 + WHIPSAW_LOOKBACK_EXTRA_BARS)
    recent_highs = highs[scan_from:-1]
    recent_lows = lows[scan_from:-1]
    recent_closes = closes[scan_from:-1]
    failed_breakouts = 0
    for h, l, c in zip(recent_highs, recent_lows, recent_closes):
        if direction == "CE" and h > channel_high and c <= channel_high:
            failed_breakouts += 1
        elif direction == "PE" and l < channel_low and c >= channel_low:
            failed_breakouts += 1

    ema_fast = _ema(closes[-min(len(closes), 60):], 9)
    ema_slow = _ema(closes[-min(len(closes), 60):], 21)
    trend_aligned = None
    if ema_fast is not None and ema_slow is not None:
        trend_aligned = (ema_fast > ema_slow) if direction == "CE" else (ema_fast < ema_slow)

    rsi = _rsi(closes, 14)
    momentum_aligned = None
    if rsi is not None:
        momentum_aligned = (rsi > 55) if direction == "CE" else (rsi < 45)

    # Composite: breakout size is the base signal; trend agreement and momentum each scale it up,
    # a counter-trend breakout gets heavily discounted (those fail far more often intraday), relative
    # volume scales it continuously, a strong close gets a small bonus, and any recent failed
    # breakouts at this same level knock the score down (capped so it never goes negative).
    composite = breakout_size_score
    if trend_aligned is True:
        composite *= 1.25
    elif trend_aligned is False:
        composite *= 0.6
    if momentum_aligned is True:
        composite *= 1.15
    if rel_volume is not None:
        composite *= min(max(rel_volume, 0.5), 2.5) / 1.2
    if strict and strong_close:
        composite *= 1.1
    if strict and failed_breakouts:
        composite *= max(0.25, 1 - WHIPSAW_PENALTY_PER_FAILURE * failed_breakouts)

    score = round(composite, 2)

    return {
        "direction": direction, "score": score, "breakout_size_atr": round(breakout_size_score, 2),
        "breakout_level": round(breakout_level, 2), "last_close": round(last_close, 2),
        "atr": round(atr, 2), "volume_confirmed": bool(vol_confirmed), "rel_volume": rel_volume,
        "trend_aligned": trend_aligned, "momentum_aligned": momentum_aligned,
        "rsi": round(rsi, 1) if rsi is not None else None,
        "channel_high": round(channel_high, 2), "channel_low": round(channel_low, 2),
        "strong_close": bool(strong_close), "close_position": round(close_position, 2),
        "failed_breakouts_recent": failed_breakouts,
    }


def _annotate_candidate(result, symbol, lookback):
    """Fills in symbol, timestamp, confidence bucket, and the plain-English reasoning string for a
    raw _detect_breakout() result. Shared by scan_breakouts (full scan) and the single-symbol
    rescan-one route (checking one candidate's CURRENT score on demand)."""
    result["symbol"] = symbol
    result["detected_at"] = now_ist().isoformat()
    # Plain-English confidence bucket from the composite score -- NOT a probability-of-profit
    # guarantee, just a rough ranking of "how many confirmations lined up" (used to decide what
    # Auto mode is allowed to touch -- see the auto-mode qualification in the loop below).
    result["confidence"] = "High" if result["score"] >= 3 else "Medium" if result["score"] >= 1.5 else "Low"
    direction_word = "broke above" if result["direction"] == "CE" else "broke below"
    trend_note = ("EMA9/21 trend agrees" if result["trend_aligned"] is True else
                   "EMA9/21 trend disagrees (counter-trend, discounted)" if result["trend_aligned"] is False
                   else "trend unavailable")
    momentum_note = (f"RSI {result['rsi']} agrees" if result["momentum_aligned"] is True else
                      f"RSI {result['rsi']} disagrees" if result["momentum_aligned"] is False
                      else "RSI unavailable")
    vol_note = (f"relative volume {result['rel_volume']}x average" if result["rel_volume"] is not None
                else "no average-volume baseline yet")
    close_note = (f"strong close ({int(result['close_position']*100)}% of candle range)"
                  if result.get("strong_close") else "weak/indecisive close")
    whipsaw = result.get("failed_breakouts_recent", 0)
    whipsaw_note = f"; {whipsaw} failed breakout(s) at this level recently (discounted)" if whipsaw else ""
    result["reasoning"] = (
        f"{symbol}: price {direction_word} its {lookback}-candle range ({result['breakout_level']}), "
        f"now at {result['last_close']} -- {result['breakout_size_atr']}x ATR raw move, {close_note}; "
        f"{trend_note}; {momentum_note}; {vol_note}{whipsaw_note}. Composite score {result['score']}."
    )
    return result


def scan_breakouts(universe, interval, lookback, strict=True):
    """Ranked, fully-transparent list of breakout candidates across the given universe. Every
    candidate carries a plain-English `reasoning` string -- this IS the "why" shown in the UI, there
    is no hidden model behind it. Calls are staggered to stay under Kite's historical-data rate
    limit when scanning long lists (e.g. the full F&O universe). `strict` toggles the extra false-
    breakout guards in _detect_breakout() -- see state["strict_breakout_filters"]."""
    candidates, errors = [], []
    for idx, symbol in enumerate(universe):
        if idx > 0:
            time.sleep(HISTORICAL_CALL_STAGGER_SECONDS)
        candles, err = _fetch_recent_intraday(symbol, interval, lookback)
        if err:
            errors.append(f"{symbol}: {err}")
            continue
        result = _detect_breakout(candles, lookback, strict=strict)
        if not result:
            continue
        candidates.append(_annotate_candidate(result, symbol, lookback))
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, errors


def _breakout_still_valid(direction, breakout_level, current_spot):
    """Has price stayed beyond the breakout level since it was detected, or has it already
    reverted back inside the range? Returns None if either input is missing (can't tell)."""
    if breakout_level is None or current_spot is None:
        return None
    return (current_spot > breakout_level) if direction == "CE" else (current_spot < breakout_level)


def _build_autotrade_order(candidate_symbol, direction, capital_per_trade):
    return _build_autotrade_order_impl(candidate_symbol, direction, capital_per_trade)


def _build_autotrade_order_impl(symbol, direction, capital_per_trade):
    data, err = get_chain_for_symbol(symbol)
    if err:
        return None, err.get("error", str(err))
    chain, lot_size, spot = data["chain"], data["lot_size"], data["spot"]
    opts = [o for o in chain if o["instrument_type"] == direction]
    if not opts:
        return None, f"No {direction} contracts found for {symbol}"
    atm = min(opts, key=lambda o: abs(o["strike"] - spot))
    if not atm.get("ltp") or atm["ltp"] <= 0:
        return None, "Could not get a valid live price for the ATM option"
    premium = atm["ltp"]
    lots = max(1, int(capital_per_trade // (premium * lot_size)))
    return {
        "symbol": symbol, "direction": direction, "tradingsymbol": atm["tradingsymbol"],
        "exchange": data.get("exchange", "NFO"), "strike": atm["strike"], "expiry": str(data["expiry"]), "lot_size": lot_size,
        "lots": lots, "quantity": lots * lot_size, "premium": premium, "spot": spot,
    }, None


def _execute_autotrade_entry(candidate, state):
    order_info, err = _build_autotrade_order(candidate["symbol"], candidate["direction"], state["capital_per_trade"])
    if err:
        return None, err

    execution_mode = state.get("execution_mode", "track")
    if execution_mode == "live":
        leg = {"leg": "auto_entry", "tradingsymbol": order_info["tradingsymbol"],
               "exchange": order_info.get("exchange", "NFO"), "transaction_type": "BUY", "quantity": order_info["quantity"]}
        # MIS (intraday) product on purpose for auto-trades: it carries the broker's OWN automatic
        # end-of-day square-off as a second, independent safety net on top of ours.
        results = place_basket_orders([leg], product="MIS", order_type="MARKET", sequence_for_margin=False)
        result = results[0] if results else {"status": "failed", "error": "No result returned"}
        if result["status"] != "placed":
            return None, result.get("error", "Order failed")
        order_id = result.get("order_id")
    else:
        # TRACK MODE: nothing is sent to the broker. order_info["premium"] already came from a real,
        # live quote (the same one a live entry would have used to size and price itself), so the
        # simulated fill below tracks real market movement exactly -- only the order placement itself
        # is skipped.
        order_id = f"TRACK-{int(time.time() * 1000)}"

    premium = order_info["premium"]
    if state.get("sl_mode") == "points":
        sl_price = round(max(premium - state.get("sl_points", 5.0), 0.05), 2)
    else:
        sl_price = round(premium * (1 - state["sl_pct_of_premium"] / 100.0), 2)
    if state.get("target_mode") == "points":
        target_price = round(premium + state.get("target_points", 10.0), 2)
    else:
        target_price = round(premium * (1 + state["target_pct_of_premium"] / 100.0), 2)
    hard_stop_price = round(premium * (1 - abs(state.get("hard_stop_pct", 50)) / 100.0), 2)
    trade = {
        "id": f"AT{int(time.time() * 1000)}", "symbol": order_info["symbol"], "direction": order_info["direction"],
        "tradingsymbol": order_info["tradingsymbol"], "strike": order_info["strike"], "expiry": order_info["expiry"],
        "quantity": order_info["quantity"], "lot_size": order_info["lot_size"], "entry_price": premium,
        "sl_price": sl_price, "target_price": target_price, "trail_active": False,
        # Exit-mode settings are snapshotted at ENTRY time -- if you change Settings while this trade
        # is open, this specific position keeps behaving the way it was opened under, rather than
        # switching exit logic underneath you mid-trade.
        "exit_mode": state.get("exit_mode", "auto"), "hard_stop_price": hard_stop_price,
        "peak_pnl_pct": 0.0, "_last_reversal_check_ts": 0,
        "execution_mode": execution_mode,
        "breakout_level": candidate["breakout_level"], "reasoning": candidate["reasoning"],
        "confidence": candidate.get("confidence"), "detected_at": candidate.get("detected_at"),
        "order_id": order_id, "status": "open", "opened_at": now_ist().isoformat(),
        "closed_at": None, "exit_reason": None, "exit_price": None, "realized_pnl": None,
        "last_ltp": premium, "mode": state["mode"],
    }
    return trade, None


def _execute_autotrade_exit(trade, reason):
    # Legacy trades opened before execution_mode existed have no "execution_mode" key -- treat those
    # as "live" (that was the only behavior that existed then), never silently as "track".
    execution_mode = trade.get("execution_mode", "live")
    if execution_mode == "live":
        leg = {"leg": "auto_exit", "tradingsymbol": trade["tradingsymbol"],
               "exchange": trade.get("exchange", "NFO"), "transaction_type": "SELL", "quantity": trade["quantity"]}
        results = place_basket_orders([leg], product="MIS", order_type="MARKET", sequence_for_margin=False)
        result = results[0] if results else {"status": "failed", "error": "No result returned"}
        exit_status = result["status"]
        trade["status"] = "closed"
        trade["closed_at"] = now_ist().isoformat()
        trade["exit_order_status"] = exit_status
        if exit_status != "placed":
            trade["exit_reason"] = reason + " -- EXIT ORDER FAILED, check your Zerodha app IMMEDIATELY"
            trade["exit_price"], trade["realized_pnl"] = None, None
            return trade
    else:
        # TRACK MODE: no order sent -- simulate an immediate fill at the last known live LTP, same
        # price a real market exit order would have been chasing.
        trade["status"] = "closed"
        trade["closed_at"] = now_ist().isoformat()
        trade["exit_order_status"] = "placed"

    exit_price = trade.get("last_ltp", trade["entry_price"])
    trade["exit_reason"] = reason
    trade["exit_price"] = exit_price
    trade["realized_pnl"] = round((exit_price - trade["entry_price"]) * trade["quantity"], 2)
    return trade


def _reconcile_broker_positions(trades):
    """If a position was closed manually in Kite (or anywhere outside this tool), the local trade
    record would otherwise sit "open" forever and this tool would try to square it off again at
    EOD. Cross-checks every locally-open LIVE trade against the broker's ACTUAL net position for
    that tradingsymbol and marks it closed here (no order placed, nothing re-sent) if the broker
    already shows it flat. TRACK-mode trades are skipped entirely -- they never had a real broker
    position to reconcile against."""
    open_trades = [t for t in trades if t["status"] == "open" and t.get("execution_mode", "live") == "live"]
    if not open_trades:
        return False
    try:
        positions = kite.positions().get("net", [])
    except Exception as e:
        logger.warning(f"Could not fetch broker positions for reconciliation: {e}")
        return False
    net_qty = {}
    for p in positions:
        if p.get("exchange") == "NFO":
            net_qty[p["tradingsymbol"]] = net_qty.get(p["tradingsymbol"], 0) + p.get("quantity", 0)
    changed = False
    for trade in open_trades:
        if net_qty.get(trade["tradingsymbol"], 0) == 0:
            trade["status"] = "closed"
            trade["closed_at"] = now_ist().isoformat()
            trade["exit_reason"] = "Closed outside this tool (e.g. manually in Kite) -- detected by checking your broker positions"
            trade["exit_price"] = trade.get("last_ltp")
            trade["realized_pnl"] = None  # actual fill price wasn't ours to see -- can't compute this reliably
            changed = True
    return changed


AUTO_EXIT_REVERSAL_MIN_INTERVAL_SECONDS = 60  # don't re-fetch underlying candles more than once a
                                               # minute per open trade -- keeps this well under Kite's
                                               # historical-data rate limit alongside the scanner


def _check_auto_exit_signal(trade, state):
    """System auto-detected exit signal, used when exit_mode == 'auto'. Re-checks the UNDERLYING's
    own trend/momentum (not the option premium) using the SAME EMA9/21 + RSI(14) confirmations that
    were used to take the trade -- the moment they flip against the position, the original breakout
    thesis is considered invalidated and the position is closed. This is what lets Auto exit mode let
    a winner run while the trend holds, instead of capping every trade at a fixed target. Rate-
    limited per trade via AUTO_EXIT_REVERSAL_MIN_INTERVAL_SECONDS. Returns a reason string, or None."""
    last_check = trade.get("_last_reversal_check_ts", 0)
    now_ts = time.time()
    if now_ts - last_check < AUTO_EXIT_REVERSAL_MIN_INTERVAL_SECONDS:
        return None
    trade["_last_reversal_check_ts"] = now_ts
    candles, err = _fetch_recent_intraday(trade["symbol"], state.get("candle_interval", "5minute"),
                                           state.get("breakout_lookback", 20))
    if err or not candles:
        return None
    closes = np.array([c["close"] for c in candles], dtype=float)
    ema_fast = _ema(closes[-min(len(closes), 60):], 9)
    ema_slow = _ema(closes[-min(len(closes), 60):], 21)
    rsi = _rsi(closes, 14)
    if ema_fast is None or ema_slow is None or rsi is None:
        return None
    if trade["direction"] == "CE":
        trend_flipped, momentum_flipped = ema_fast < ema_slow, rsi < 45
    else:
        trend_flipped, momentum_flipped = ema_fast > ema_slow, rsi > 55
    if trend_flipped and momentum_flipped:
        return f"Auto-exit: underlying trend reversed (EMA9/21 flipped, RSI {round(rsi, 1)}) -- breakout thesis invalidated"
    return None


def _monitor_autotrade_positions(state, trades):
    """Checks every open auto-trade and exits it immediately (real market order) the instant its
    exit condition trips. Two exit modes, chosen per-trade at entry (trade["exit_mode"]):

      "auto"      -- system auto-detected exit: trend-reversal check (_check_auto_exit_signal),
                     trailing-stop off the peak profit reached, and a hard safety-loss floor, in
                     that priority. No fixed target cap -- winners run as long as the trend holds.
      "target_sl" -- fixed Target % / Stop-loss % (+ optional trailing once trail_after_pct is
                     reached), exactly as configured in Settings.

    EOD square-off and the hard safety floor apply either way."""
    changed = _reconcile_broker_positions(trades)
    for trade in trades:
        if trade["status"] != "open":
            continue
        try:
            q = kite_quote_bulk([f"NFO:{trade['tradingsymbol']}"]).get(f"NFO:{trade['tradingsymbol']}")
            ltp = extract_price(q)
        except Exception as e:
            state["last_error"] = f"Quote fetch failed for {trade['tradingsymbol']}: {e}"
            continue
        if not ltp:
            continue
        trade["last_ltp"] = ltp
        pnl_pct = (ltp / trade["entry_price"] - 1) * 100.0
        trade["peak_pnl_pct"] = max(trade.get("peak_pnl_pct", 0.0), pnl_pct)
        now_str = now_ist().strftime("%H:%M")
        exit_mode = trade.get("exit_mode", "auto")
        reason = None

        if exit_mode == "auto":
            hard_stop_price = trade.get("hard_stop_price") or round(
                trade["entry_price"] * (1 - abs(state.get("hard_stop_pct", 50)) / 100.0), 2)
            peak = trade["peak_pnl_pct"]
            if ltp <= hard_stop_price:
                reason = "Hard safety stop hit"
            elif peak >= state.get("trail_after_pct", 30) and pnl_pct <= peak - state.get("trail_giveback_pct", 15):
                reason = "Auto-exit: trailing stop from peak profit"
                trade["trail_active"] = True
            if not reason:
                reason = _check_auto_exit_signal(trade, state)
            if not reason and now_str >= state.get("square_off_time", "15:15"):
                reason = "End-of-day square-off"
        else:  # "target_sl" -- fixed target/stop-loss (+ trailing), as configured
            if pnl_pct >= state["trail_after_pct"]:
                new_sl = trade["entry_price"] * (1 + (pnl_pct - state["trail_giveback_pct"]) / 100.0)
                if new_sl > trade["sl_price"]:
                    trade["sl_price"] = round(new_sl, 2)
                    trade["trail_active"] = True
            if ltp <= trade["sl_price"]:
                reason = "Trailing stop hit" if trade["trail_active"] else "Stop loss hit"
            elif ltp >= trade["target_price"] and not trade["trail_active"]:
                reason = "Target hit"
            elif now_str >= state.get("square_off_time", "15:15"):
                reason = "End-of-day square-off"

        if reason:
            _execute_autotrade_exit(trade, reason)
            state["realized_pnl_today"] = round(state.get("realized_pnl_today", 0.0) + (trade["realized_pnl"] or 0.0), 2)
            changed = True
    return changed


def _autotrade_loop():
    """Background daemon thread -- always running as long as you're logged in. Stop-loss/target/
    trailing-stop monitoring and auto-close (_monitor_autotrade_positions below) apply to EVERY open
    position regardless of how it was opened -- a manual "Execute" click from the breakout list gets
    exactly the same automatic SL/target management as an Auto-mode entry, and it keeps running even
    if you click "Stop scanning" (that only stops NEW entries; it does not touch a position already
    open). Only scanning for new signals and self-executing new entries require state['enabled'].
    Sleeps state['poll_seconds'] between iterations."""
    while True:
        sleep_for = AUTOTRADE_DEFAULTS["poll_seconds"]
        try:
            if SESSION.get("access_token"):
                state = load_autotrade_state()
                state = _autotrade_roll_day_if_needed(state)
                trades = load_autotrade_trades()
                changed = _monitor_autotrade_positions(state, trades)

                if state.get("realized_pnl_today", 0.0) <= -abs(state.get("max_daily_loss", 5000)) and state["enabled"]:
                    state["enabled"] = False
                    state["disarm_reason"] = (f"Daily loss limit of Rs {state['max_daily_loss']} reached "
                                               f"(realized P&L today: Rs {state['realized_pnl_today']}). Auto mode disarmed.")
                    changed = True

                open_count = sum(1 for t in trades if t["status"] == "open")
                max_positions = max(1, int(state.get("max_concurrent_positions", 1)))
                now_str = now_ist().strftime("%H:%M")
                within_hours = AUTOTRADE_MARKET_OPEN <= now_str <= state.get("square_off_time", "15:15")

                if (state["enabled"] and open_count < max_positions and within_hours
                        and state.get("trades_today", 0) < state.get("max_trades_per_day", 3)):
                    scan_universe = _effective_scan_universe(state)
                    candidates, errors = scan_breakouts(scan_universe, state["candle_interval"],
                                                         state["breakout_lookback"],
                                                         strict=state.get("strict_breakout_filters", True))
                    state["last_scan_at"] = now_ist().isoformat()
                    state["last_scan_candidates"] = candidates[:10]
                    state["last_scan_universe_size"] = len(scan_universe)
                    state["last_error"] = "; ".join(errors[:3]) if errors else None
                    changed = True

                    if state["mode"] == "auto":
                        # Auto mode gets a materially higher bar than Manual: only "High" confidence
                        # (composite score >= 3), with trend AND momentum both agreeing AND volume
                        # confirmed -- not just whichever candidate first clears min_breakout_score.
                        # Manual mode still shows every candidate that clears min_breakout_score and
                        # lets the person decide, including lower-confidence ones.
                        auto_qualified = [c for c in candidates
                                           if c.get("confidence") == "High"
                                           and c["score"] >= max(state.get("min_breakout_score", 0.5), 3)
                                           and c.get("trend_aligned") is True
                                           and c.get("momentum_aligned") is True
                                           and c["volume_confirmed"]]
                        best = None
                        for c in auto_qualified:
                            # Re-verify right now, not just at scan time -- price can revert in the
                            # gap between a scan and actually committing capital.
                            fresh_order, ferr = _build_autotrade_order(c["symbol"], c["direction"], state["capital_per_trade"])
                            if ferr:
                                continue
                            if _breakout_still_valid(c["direction"], c["breakout_level"], fresh_order["spot"]) is False:
                                continue
                            best = c
                            break
                        if best:
                            trade, err = _execute_autotrade_entry(best, state)
                            if trade:
                                trades.append(trade)
                                state["trades_today"] = state.get("trades_today", 0) + 1
                            else:
                                state["last_error"] = f"Auto-entry failed for {best['symbol']}: {err}"

                if changed:
                    save_autotrade_state(state)
                    save_autotrade_trades(trades)
                sleep_for = state.get("poll_seconds", 20)
                if state.get("scan_all_fo"):
                    sleep_for = max(sleep_for, FO_SCAN_MIN_POLL_SECONDS)
        except Exception as e:
            logger.exception("autotrade loop error")
            try:
                err_state = load_autotrade_state()
                err_state["last_error"] = f"Loop error: {e}"
                save_autotrade_state(err_state)
            except Exception:
                pass
        time.sleep(max(5, sleep_for))



# ============================================================================
# CONTINUOUS INDEX BREAKOUT MONITOR
# NIFTY / BANKNIFTY / SENSEX / FINNIFTY
# Bullish breakout -> BUY ATM CE; bearish breakout -> BUY ATM PE.
# Closing a BUY option position always uses SELL. Paper mode never sends broker orders.
# ============================================================================
BREAKOUT_MONITOR_FILE = os.path.join(os.path.dirname(__file__), "breakout_monitor_state.json")
BREAKOUT_MONITOR_TRADES_FILE = os.path.join(os.path.dirname(__file__), "breakout_monitor_trades.json")
BREAKOUT_MONITOR_LOCK = threading.Lock()
BREAKOUT_MONITOR_SYMBOLS = ["NIFTY", "BANKNIFTY", "SENSEX", "FINNIFTY"]
BREAKOUT_MONITOR_DEFAULTS = {
    "enabled": False,
    "execution_mode": "paper",       # paper | live
    "interval": "5minute",
    "lookback": 20,
    "poll_seconds": 20,
    "capital_per_trade": 15000,
    "max_trades_per_day": 4,
    "max_concurrent_positions": 1,
    "max_daily_loss": 5000,
    "min_score": 6,                   # 6/7 may be monitored; 7/7 is highest quality
    "hard_stop_pct": 35,
    "trail_after_pct": 30,
    "trail_giveback_pct": 15,
    "square_off_time": "15:15",
    "symbols": BREAKOUT_MONITOR_SYMBOLS,
    "last_scan_at": None,
    "last_error": None,
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "day": None,
    "last_signals": {},
    "disarm_reason": None,
}


def _load_breakout_monitor_state():
    state = dict(BREAKOUT_MONITOR_DEFAULTS)
    if os.path.exists(BREAKOUT_MONITOR_FILE):
        try:
            with open(BREAKOUT_MONITOR_FILE, "r") as f:
                state.update(json.load(f))
        except Exception:
            pass
    return state


def _save_breakout_monitor_state(state):
    with BREAKOUT_MONITOR_LOCK:
        with open(BREAKOUT_MONITOR_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)


def _load_breakout_monitor_trades():
    if not os.path.exists(BREAKOUT_MONITOR_TRADES_FILE):
        return []
    try:
        with open(BREAKOUT_MONITOR_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_breakout_monitor_trades(trades):
    with BREAKOUT_MONITOR_LOCK:
        with open(BREAKOUT_MONITOR_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)


def _breakout_monitor_roll_day(state):
    today = now_ist().strftime("%Y-%m-%d")
    if state.get("day") != today:
        state["day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["disarm_reason"] = None
    return state


def _monitor_session_vwap(candles, upto_index):
    if upto_index < 0:
        return None
    last_dt = candles[upto_index].get("date")
    last_day = last_dt.date() if hasattr(last_dt, "date") else str(last_dt)[:10]
    session = []
    for c in candles[:upto_index + 1]:
        d = c.get("date")
        day = d.date() if hasattr(d, "date") else str(d)[:10]
        if day == last_day:
            session.append(c)
    vol = sum(float(c.get("volume", 0) or 0) for c in session)
    if vol <= 0:
        return None
    return sum(((float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0) * float(c.get("volume", 0) or 0) for c in session) / vol


def _breakout_monitor_signal(symbol, candles, lookback=20, option_chain_data=None):
    """Detailed 7-condition directional breakout signal for one index."""
    if len(candles) < max(lookback + 2, 55):
        return None, "Not enough candles"
    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    volumes = np.array([float(c.get("volume", 0) or 0) for c in candles], dtype=float)
    spot = closes[-1]
    resistance = float(np.max(highs[-(lookback + 1):-1]))
    support = float(np.min(lows[-(lookback + 1):-1]))
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = float(np.mean(tr[-lookback:])) if len(tr) >= lookback else float(np.mean(tr))
    if atr <= 0:
        return None, "ATR unavailable"
    ema20 = _ema(closes[-min(len(closes), 120):], 20)
    ema50 = _ema(closes[-min(len(closes), 120):], 50)
    rsi = _rsi(closes, 14)
    vwap = _monitor_session_vwap(candles, len(candles) - 1)
    avg_vol = float(np.mean(volumes[-(lookback + 1):-1]))
    rel_volume = float(volumes[-1] / avg_vol) if avg_vol > 0 else None
    bull_dist = spot - resistance
    bear_dist = support - spot
    bull_break = bull_dist >= 0.15 * atr
    bear_break = bear_dist >= 0.15 * atr
    if bull_break and not bear_break:
        direction = "CE"
    elif bear_break and not bull_break:
        direction = "PE"
    else:
        direction = None
    # Prefer a direction only when the breakout itself is clear; no trade in the middle.
    if direction == "CE":
        level = resistance
        close_pos = (spot - lows[-1]) / max(highs[-1] - lows[-1], 1e-9)
        strong_close = close_pos >= 0.60
        conditions = {
            "breakout": bull_break,
            "above_vwap": vwap is not None and spot > vwap,
            "ema20_50": ema20 is not None and ema50 is not None and ema20 > ema50,
            "rsi": rsi is not None and rsi > 55,
            "volume": rel_volume is not None and rel_volume >= 1.5,
            "level_broken": spot > resistance,
            "option_momentum": False,
        }
    elif direction == "PE":
        level = support
        close_pos = (highs[-1] - spot) / max(highs[-1] - lows[-1], 1e-9)
        strong_close = close_pos >= 0.60
        conditions = {
            "breakout": bear_break,
            "above_vwap": vwap is not None and spot < vwap,  # named consistently in UI; means below VWAP for PE
            "ema20_50": ema20 is not None and ema50 is not None and ema20 < ema50,
            "rsi": rsi is not None and rsi < 45,
            "volume": rel_volume is not None and rel_volume >= 1.5,
            "level_broken": spot < support,
            "option_momentum": False,
        }
    else:
        return {"symbol": symbol, "direction": None, "score": 0, "quality": "NO BREAKOUT", "spot": round(spot, 2),
                "vwap": round(vwap, 2) if vwap is not None else None, "ema20": round(ema20, 2) if ema20 else None,
                "ema50": round(ema50, 2) if ema50 else None, "rsi": round(rsi, 1) if rsi is not None else None,
                "resistance": round(resistance, 2), "support": round(support, 2), "rel_volume": round(rel_volume, 2) if rel_volume else None,
                "atr": round(atr, 2), "conditions": conditions if False else {"breakout": False, "above_vwap": False, "ema20_50": False, "rsi": False, "volume": False, "level_broken": False, "option_momentum": False},
                "checked_at": now_ist().isoformat(), "note": "No directional breakout beyond the 0.15 ATR noise floor."}, None

    # False-breakout guard: breakout candle should close strongly in the breakout direction.
    conditions["strong_close"] = strong_close
    # Option-side momentum: ATM option must move in the same direction over recent 5-min bars and beat opposite side.
    option_detail = {}
    if option_chain_data:
        chain = option_chain_data.get("chain") or []
        chain_spot = float(option_chain_data.get("spot") or spot)
        own = [o for o in chain if o.get("instrument_type") == direction and o.get("instrument_token")]
        opp = [o for o in chain if o.get("instrument_type") == ("PE" if direction == "CE" else "CE") and o.get("instrument_token")]
        if own:
            own_atm = min(own, key=lambda o: abs(float(o["strike"]) - chain_spot))
            opp_atm = min(opp, key=lambda o: abs(float(o["strike"]) - chain_spot)) if opp else None
            own_hist = opp_hist = []
            try:
                ex = option_chain_data.get("exchange", "NFO")
                own_hist = kite.historical_data(int(own_atm["instrument_token"]), now_ist() - timedelta(days=3), now_ist(), "5minute")
                if opp_atm:
                    opp_hist = kite.historical_data(int(opp_atm["instrument_token"]), now_ist() - timedelta(days=3), now_ist(), "5minute")
            except Exception as e:
                logger.warning("%s option momentum fetch failed: %s", symbol, e)
            own_c = [float(c["close"]) for c in own_hist if c.get("close") is not None]
            opp_c = [float(c["close"]) for c in opp_hist if c.get("close") is not None]
            own_change = (own_c[-1] / own_c[-2] - 1) if len(own_c) >= 2 and own_c[-2] else None
            opp_change = (opp_c[-1] / opp_c[-2] - 1) if len(opp_c) >= 2 and opp_c[-2] else None
            own_rise = len(own_c) >= 4 and own_c[-1] > own_c[-2] and own_c[-1] > own_c[-4]
            own_volume = float(own_atm.get("volume") or 0)
            opp_volume = float(opp_atm.get("volume") or 0) if opp_atm else 0
            conditions["option_momentum"] = bool(own_rise and (opp_change is None or own_change is None or own_change > opp_change) and (own_volume == 0 or opp_volume == 0 or own_volume >= opp_volume))
            option_detail = {"tradingsymbol": own_atm.get("tradingsymbol"), "strike": own_atm.get("strike"), "expiry": str(option_chain_data.get("expiry")),
                             "premium": own_atm.get("ltp"), "own_change_pct": round(own_change * 100, 2) if own_change is not None else None,
                             "opposite_change_pct": round(opp_change * 100, 2) if opp_change is not None else None,
                             "own_volume": own_volume, "opposite_volume": opp_volume, "exchange": option_chain_data.get("exchange", "NFO")}

    # Strong close is an additional quality guard but not counted as one of the seven displayed confirmations.
    score = sum(1 for k in ("breakout", "above_vwap", "ema20_50", "rsi", "volume", "level_broken", "option_momentum") if conditions[k])
    quality = "HIGH QUALITY BREAKOUT" if score == 7 and strong_close else ("QUALIFIED — 6/7+" if score >= 6 else ("WATCH — 5/7" if score == 5 else "NO TRADE"))
    return {
        "symbol": symbol, "direction": direction, "score": score, "max_score": 7, "quality": quality,
        "spot": round(spot, 2), "vwap": round(vwap, 2) if vwap is not None else None,
        "ema20": round(ema20, 2) if ema20 is not None else None, "ema50": round(ema50, 2) if ema50 is not None else None,
        "rsi": round(rsi, 1) if rsi is not None else None, "resistance": round(resistance, 2), "support": round(support, 2),
        "breakout_level": round(level, 2), "breakout_distance": round(abs(spot - level), 2), "atr": round(atr, 2),
        "rel_volume": round(rel_volume, 2) if rel_volume is not None else None, "strong_close": strong_close,
        "conditions": {k: bool(v) for k, v in conditions.items() if k != "strong_close"}, "option": option_detail,
        "checked_at": now_ist().isoformat(),
        "reason": f"{symbol} {direction}: {'above resistance' if direction == 'CE' else 'below support'} by {abs(spot-level):.2f} ({abs(spot-level)/atr:.2f} ATR); VWAP/EMA/RSI/volume/option momentum confirmations are shown individually."
    }, None


def _breakout_monitor_build_state_for_entry(state):
    # Reuse the existing, tested option-entry sizing/execution machinery with the breakout monitor's own limits.
    s = dict(AUTOTRADE_DEFAULTS)
    s.update({
        "execution_mode": "live" if state.get("execution_mode") == "live" else "track",
        "capital_per_trade": float(state.get("capital_per_trade", 15000)),
        "sl_mode": "pct", "sl_pct_of_premium": float(state.get("hard_stop_pct", 35)),
        "hard_stop_pct": float(state.get("hard_stop_pct", 35)), "exit_mode": "auto",
        "trail_after_pct": float(state.get("trail_after_pct", 30)), "trail_giveback_pct": float(state.get("trail_giveback_pct", 15)),
        "square_off_time": state.get("square_off_time", "15:15"), "mode": "auto",
    })
    return s


def _breakout_monitor_entry(signal, state):
    entry_state = _breakout_monitor_build_state_for_entry(state)
    return _execute_autotrade_entry(signal, entry_state)


def _breakout_monitor_exit(trade, reason):
    return _execute_autotrade_exit(trade, reason)


def _breakout_monitor_positions(state, trades):
    changed = False
    for trade in trades:
        if trade.get("status") != "open":
            continue
        key = f"{trade.get('exchange', 'NFO')}:{trade['tradingsymbol']}"
        try:
            q = kite_quote_bulk([key]).get(key)
            ltp = extract_price(q)
        except Exception as e:
            state["last_error"] = f"Quote failed for {trade['tradingsymbol']}: {e}"
            continue
        if not ltp:
            continue
        trade["last_ltp"] = ltp
        pnl_pct = (ltp / trade["entry_price"] - 1) * 100.0
        trade["peak_pnl_pct"] = max(float(trade.get("peak_pnl_pct", 0)), pnl_pct)
        reason = None
        hard_stop = float(trade.get("hard_stop_price") or trade["entry_price"] * (1 - state.get("hard_stop_pct", 35) / 100))
        if ltp <= hard_stop:
            reason = "Hard safety stop hit"
        elif trade["peak_pnl_pct"] >= state.get("trail_after_pct", 30) and pnl_pct <= trade["peak_pnl_pct"] - state.get("trail_giveback_pct", 15):
            reason = "Trailing exit after momentum/profit giveback"
        else:
            # Underlying reversal / breakout invalidation check.
            candles, err = _fetch_recent_intraday(trade["symbol"], state.get("interval", "5minute"), state.get("lookback", 20))
            if not err and candles:
                closes = np.array([float(c["close"]) for c in candles], dtype=float)
                ema20 = _ema(closes[-min(len(closes), 120):], 20)
                ema50 = _ema(closes[-min(len(closes), 120):], 50)
                rsi = _rsi(closes, 14)
                spot = closes[-1]
                level = float(trade.get("breakout_level") or spot)
                if trade["direction"] == "CE" and ((ema20 is not None and ema50 is not None and ema20 < ema50 and rsi is not None and rsi < 50) or spot < level):
                    reason = "Bullish breakout invalidated"
                elif trade["direction"] == "PE" and ((ema20 is not None and ema50 is not None and ema20 > ema50 and rsi is not None and rsi > 50) or spot > level):
                    reason = "Bearish breakout invalidated"
        if not reason and now_ist().strftime("%H:%M") >= state.get("square_off_time", "15:15"):
            reason = "End-of-day square-off"
        if reason:
            closed = _breakout_monitor_exit(trade, reason)
            if closed.get("realized_pnl") is not None:
                state["realized_pnl_today"] = round(state.get("realized_pnl_today", 0) + closed["realized_pnl"], 2)
            changed = True
    return changed


def _breakout_monitor_scan(state):
    signals = {}
    for symbol in state.get("symbols") or BREAKOUT_MONITOR_SYMBOLS:
        try:
            candles, err = _fetch_recent_intraday(symbol, state.get("interval", "5minute"), max(int(state.get("lookback", 20)), 55))
            if err:
                signals[symbol] = {"symbol": symbol, "error": err, "checked_at": now_ist().isoformat()}
                continue
            # Fetch option chain only after the underlying has produced a directional breakout.
            prelim, _ = _breakout_monitor_signal(symbol, candles, int(state.get("lookback", 20)), None)
            chain_data = None
            if prelim and prelim.get("direction"):
                chain_data, _ = get_chain_for_symbol(symbol)
            signal, err2 = _breakout_monitor_signal(symbol, candles, int(state.get("lookback", 20)), chain_data)
            if err2:
                signals[symbol] = {"symbol": symbol, "error": err2, "checked_at": now_ist().isoformat()}
            else:
                signals[symbol] = signal
        except Exception as e:
            logger.exception("Breakout monitor scan failed for %s", symbol)
            signals[symbol] = {"symbol": symbol, "error": str(e), "checked_at": now_ist().isoformat()}
    state["last_signals"] = signals
    state["last_scan_at"] = now_ist().isoformat()
    return signals


def _breakout_monitor_loop():
    while True:
        sleep_for = 20
        try:
            if SESSION.get("access_token"):
                state = _breakout_monitor_roll_day(_load_breakout_monitor_state())
                trades = _load_breakout_monitor_trades()
                changed = _breakout_monitor_positions(state, trades)
                now_str = now_ist().strftime("%H:%M")
                if state.get("enabled") and now_str >= "09:20" and now_str <= state.get("square_off_time", "15:15"):
                    signals = _breakout_monitor_scan(state)
                    open_count = sum(1 for t in trades if t.get("status") == "open")
                    if (state.get("realized_pnl_today", 0) <= -abs(state.get("max_daily_loss", 5000))):
                        state["enabled"] = False
                        state["disarm_reason"] = "Daily loss limit reached"
                        changed = True
                    elif state.get("trades_today", 0) < int(state.get("max_trades_per_day", 4)) and open_count < int(state.get("max_concurrent_positions", 1)):
                        ranked = [s for s in signals.values() if s.get("direction") and s.get("score", 0) >= int(state.get("min_score", 6)) and s.get("option", {}).get("tradingsymbol")]
                        ranked.sort(key=lambda x: (x.get("score", 0), x.get("breakout_distance", 0)), reverse=True)
                        if ranked:
                            candidate = ranked[0]
                            # Prevent repeated entries on the same breakout until its signal changes/position closes.
                            already = any(t.get("status") == "open" and t.get("symbol") == candidate["symbol"] for t in trades)
                            last_trade_key = f"{candidate['symbol']}:{candidate['direction']}:{candidate.get('breakout_level')}"
                            duplicate_recent = any(t.get("status") == "closed" and t.get("signal_key") == last_trade_key and t.get("closed_at", "")[:10] == state.get("day") for t in trades[-50:])
                            if not already and not duplicate_recent:
                                trade, err = _breakout_monitor_entry(candidate, state)
                                if trade:
                                    trade["signal_key"] = last_trade_key
                                    trade["breakout_monitor"] = True
                                    trades.append(trade)
                                    state["trades_today"] = int(state.get("trades_today", 0)) + 1
                                    changed = True
                                else:
                                    state["last_error"] = f"Entry failed for {candidate['symbol']}: {err}"
                if changed or state.get("last_signals"):
                    _save_breakout_monitor_state(state)
                    _save_breakout_monitor_trades(trades)
                sleep_for = max(10, int(state.get("poll_seconds", 20)))
        except Exception as e:
            logger.exception("Breakout monitor loop error")
        time.sleep(sleep_for)


def _breakout_backtest_symbol(symbol, interval, days, lookback, stop_atr, target_atr, min_score=5):
    """Historical breakout identification + underlying-point simulation.

    IMPORTANT: historical ATM option momentum/premium is deliberately NOT required here because
    reconstructing the correct historical ATM CE/PE for every candidate would require a separate
    option-chain history pass. The backtest therefore evaluates the six index-side confirmations:
    breakout, VWAP, EMA20/50, RSI, volume and resistance/support break, plus the strong-close
    false-breakout guard. It reports exactly how many raw breakouts were found and where the
    confirmations rejected them, instead of silently returning zero trades.
    """
    token, err = resolve_token_for_symbol(symbol)
    if err:
        return {"symbol": symbol, "error": err, "trades": []}

    end = now_ist()
    requested_days = min(max(int(days), 5), 120)
    # Fetch in <=20-day chunks. This avoids broker historical-range limits for intraday candles
    # and makes the backtest work consistently for both 5-minute and 15-minute intervals.
    start = end - timedelta(days=requested_days + 5)
    chunks = []
    cursor = start
    try:
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=20), end)
            part = kite.historical_data(token, cursor, chunk_end, interval)
            chunks.extend(part or [])
            cursor = chunk_end
            time.sleep(0.36)
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "trades": []}

    # De-duplicate and sort candles by timestamp.
    uniq = {}
    for c in chunks:
        d = c.get("date")
        key = d.isoformat() if hasattr(d, "isoformat") else str(d)
        uniq[key] = c
    candles = [uniq[k] for k in sorted(uniq)]
    if len(candles) < max(lookback + 55, 70):
        return {"symbol": symbol, "error": f"Only {len(candles)} candles returned; need at least {max(lookback + 55, 70)}.", "trades": []}

    closes = np.array([float(c["close"]) for c in candles], dtype=float)
    highs = np.array([float(c["high"]) for c in candles], dtype=float)
    lows = np.array([float(c["low"]) for c in candles], dtype=float)
    vols = np.array([float(c.get("volume", 0) or 0) for c in candles], dtype=float)

    min_score = max(1, min(int(min_score), 6))
    trades = []
    signals = []
    raw_breakouts = 0
    bullish_raw = bearish_raw = 0
    rejection_counts = {
        "breakout_size": 0, "vwap": 0, "ema": 0, "rsi": 0, "volume": 0,
        "strong_close": 0, "qualified": 0
    }
    in_trade = None
    first_qualified = []

    start_i = max(lookback + 55, 70)
    for i in range(start_i, len(candles)):
        tr = np.maximum(
            highs[1:i+1] - lows[1:i+1],
            np.maximum(np.abs(highs[1:i+1] - closes[:i]), np.abs(lows[1:i+1] - closes[:i]))
        )
        atr = float(np.mean(tr[-lookback:])) if len(tr) >= lookback else 0.0
        if atr <= 0:
            continue

        resistance = float(np.max(highs[i-lookback:i]))
        support = float(np.min(lows[i-lookback:i]))
        close = closes[i]
        candle_range = max(highs[i] - lows[i], 1e-9)
        vwap = _monitor_session_vwap(candles, i)
        ema20 = _ema(closes[max(0, i-119):i+1], 20)
        ema50 = _ema(closes[max(0, i-119):i+1], 50)
        rsi = _rsi(closes[:i+1], 14)
        avg_vol = float(np.mean(vols[i-lookback:i])) if i >= lookback else 0.0
        rv = vols[i] / avg_vol if avg_vol > 0 else 0.0

        bull_raw = close > resistance
        bear_raw = close < support
        if bull_raw or bear_raw:
            raw_breakouts += 1
            bullish_raw += int(bull_raw)
            bearish_raw += int(bear_raw)

        if not bull_raw and not bear_raw:
            # Still manage an open position below.
            pass
        else:
            direction = "CE" if bull_raw else "PE"
            level = resistance if bull_raw else support
            breakout_ok = abs(close - level) >= 0.15 * atr
            vwap_ok = (vwap is not None and ((close > vwap) if bull_raw else (close < vwap)))
            ema_ok = (ema20 is not None and ema50 is not None and ((ema20 > ema50) if bull_raw else (ema20 < ema50)))
            rsi_ok = (rsi is not None and ((rsi > 55) if bull_raw else (rsi < 45)))
            volume_ok = rv >= 1.5
            strong_close_ok = ((close - lows[i]) / candle_range >= 0.6) if bull_raw else ((highs[i] - close) / candle_range >= 0.6)
            conditions = [breakout_ok, vwap_ok, ema_ok, rsi_ok, volume_ok, True]
            score = sum(bool(x) for x in conditions)

            if not breakout_ok: rejection_counts["breakout_size"] += 1
            if not vwap_ok: rejection_counts["vwap"] += 1
            if not ema_ok: rejection_counts["ema"] += 1
            if not rsi_ok: rejection_counts["rsi"] += 1
            if not volume_ok: rejection_counts["volume"] += 1
            if not strong_close_ok: rejection_counts["strong_close"] += 1

            # The sixth item is the already-proven resistance/support break; strong-close is an
            # extra guard, not one of the six score points.
            qualified = breakout_ok and score >= min_score and strong_close_ok
            if qualified:
                rejection_counts["qualified"] += 1
                if len(first_qualified) < 12:
                    first_qualified.append({
                        "time": str(candles[i]["date"]), "direction": direction,
                        "price": round(close, 2), "level": round(level, 2),
                        "score": score, "rsi": round(rsi, 1) if rsi is not None else None,
                        "rv": round(rv, 2), "vwap": round(vwap, 2) if vwap is not None else None,
                    })
                signals.append({"i": i, "direction": direction, "level": level, "atr": atr,
                                "score": score, "time": candles[i]["date"]})

        # Manage existing underlying simulation.
        if in_trade:
            entry = in_trade["entry"]
            if in_trade["direction"] == "CE":
                stop = entry - stop_atr * in_trade["atr"]
                target = entry + target_atr * in_trade["atr"]
                reversal = close < in_trade["level"] or (ema20 is not None and ema50 is not None and ema20 < ema50 and rsi is not None and rsi < 50)
                hit_stop, hit_target = lows[i] <= stop, highs[i] >= target
                if hit_stop or hit_target or reversal or i == len(candles)-1:
                    if hit_stop:
                        exit_px, reason = stop, "ATR stop"
                    elif hit_target:
                        exit_px, reason = target, "ATR target"
                    else:
                        exit_px, reason = close, "Breakout invalidated/reversal"
                    pts = exit_px - entry
                    trades.append({**in_trade, "exit": round(exit_px,2), "points": round(pts,2),
                                   "result": "WIN" if pts > 0 else "LOSS", "exit_reason": reason,
                                   "entry_time": str(in_trade["entry_time"]), "exit_time": str(candles[i]["date"])})
                    in_trade = None
            else:
                stop = entry + stop_atr * in_trade["atr"]
                target = entry - target_atr * in_trade["atr"]
                reversal = close > in_trade["level"] or (ema20 is not None and ema50 is not None and ema20 > ema50 and rsi is not None and rsi > 50)
                hit_stop, hit_target = highs[i] >= stop, lows[i] <= target
                if hit_stop or hit_target or reversal or i == len(candles)-1:
                    if hit_stop:
                        exit_px, reason = stop, "ATR stop"
                    elif hit_target:
                        exit_px, reason = target, "ATR target"
                    else:
                        exit_px, reason = close, "Breakout invalidated/reversal"
                    pts = entry - exit_px
                    trades.append({**in_trade, "exit": round(exit_px,2), "points": round(pts,2),
                                   "result": "WIN" if pts > 0 else "LOSS", "exit_reason": reason,
                                   "entry_time": str(in_trade["entry_time"]), "exit_time": str(candles[i]["date"])})
                    in_trade = None

        # Enter only after the signal has been fully confirmed.
        if in_trade is None and signals and signals[-1].get("i") == i:
            sig = signals[-1]
            in_trade = {"direction": sig["direction"], "entry": float(close), "level": float(sig["level"]),
                        "atr": float(sig["atr"]), "score": int(sig["score"]), "entry_time": candles[i]["date"]}

    total_points = round(sum(t["points"] for t in trades), 2)
    wins = sum(1 for t in trades if t["points"] > 0)
    losses = sum(1 for t in trades if t["points"] <= 0)
    return {
        "symbol": symbol, "trades": trades, "trade_count": len(trades), "wins": wins, "losses": losses,
        "win_rate": round(100 * wins / len(trades), 1) if trades else 0, "points": total_points,
        "best_points": max([t["points"] for t in trades], default=0),
        "worst_points": min([t["points"] for t in trades], default=0),
        "candles": len(candles), "raw_breakouts": raw_breakouts,
        "bullish_raw": bullish_raw, "bearish_raw": bearish_raw,
        "qualified_signals": rejection_counts["qualified"],
        "rejection_counts": rejection_counts,
        "first_qualified": first_qualified,
        "min_score": min_score,
        "option_momentum_backtested": False,
        "note": "Historical test uses six index-side confirmations. Historical ATM CE/PE momentum and option-premium P&L are not fabricated from today's chain; option momentum is therefore shown as unavailable for the historical test. Strong-close is an additional false-breakout guard."
    }


@app.route("/api/breakout-monitor/state")
def breakout_monitor_state_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = _breakout_monitor_roll_day(_load_breakout_monitor_state())
    trades = _load_breakout_monitor_trades()
    _save_breakout_monitor_state(state)
    return jsonify({"state": state, "open_trades": [t for t in trades if t.get("status") == "open"],
                    "recent_trades": sorted(trades, key=lambda t: t.get("opened_at", ""), reverse=True)[:50]})


@app.route("/api/breakout-monitor/config", methods=["POST"])
def breakout_monitor_config_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    state = _load_breakout_monitor_state()
    allowed = {"interval", "lookback", "poll_seconds", "capital_per_trade", "max_trades_per_day", "max_concurrent_positions", "max_daily_loss", "min_score", "hard_stop_pct", "trail_after_pct", "trail_giveback_pct", "square_off_time", "symbols"}
    for k in allowed:
        if k in body:
            state[k] = body[k]
    state["symbols"] = [s.upper() for s in state.get("symbols", BREAKOUT_MONITOR_SYMBOLS) if s.upper() in BREAKOUT_MONITOR_SYMBOLS]
    _save_breakout_monitor_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/breakout-monitor/mode", methods=["POST"])
def breakout_monitor_mode_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    mode = body.get("mode")
    if mode not in ("paper", "live"):
        return jsonify({"error": "mode must be paper or live"}), 400
    if mode == "live" and body.get("ack") is not True:
        return jsonify({"error": "Live mode requires explicit confirmation that future entries/exits place REAL Zerodha orders."}), 400
    state = _load_breakout_monitor_state()
    state["execution_mode"] = mode
    _save_breakout_monitor_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/breakout-monitor/arm", methods=["POST"])
def breakout_monitor_arm_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    state = _breakout_monitor_roll_day(_load_breakout_monitor_state())
    if state.get("execution_mode") == "live" and body.get("ack") is not True:
        return jsonify({"error": "Arming LIVE breakout automation requires explicit confirmation."}), 400
    state["enabled"] = True
    state["disarm_reason"] = None
    _save_breakout_monitor_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/breakout-monitor/disarm", methods=["POST"])
def breakout_monitor_disarm_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = _load_breakout_monitor_state()
    state["enabled"] = False
    state["disarm_reason"] = "Manually stopped by user"
    _save_breakout_monitor_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/breakout-monitor/backtest")
def breakout_monitor_backtest_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        days = max(5, min(int(request.args.get("days", 30)), 120))
        interval = request.args.get("interval", "5minute")
        if interval not in ("5minute", "15minute"):
            return jsonify({"error": "Backtest interval must be 5minute or 15minute"}), 400
        lookback = max(10, min(int(request.args.get("lookback", 20)), 60))
        stop_atr = max(0.25, float(request.args.get("stop_atr", 1.0)))
        target_atr = max(0.5, float(request.args.get("target_atr", 2.0)))
        min_score = max(1, min(int(request.args.get("min_score", 5)), 6))
        results = [_breakout_backtest_symbol(s, interval, days, lookback, stop_atr, target_atr, min_score) for s in BREAKOUT_MONITOR_SYMBOLS]
        return jsonify({"days": days, "interval": interval, "lookback": lookback, "stop_atr": stop_atr, "target_atr": target_atr,
                        "results": results, "min_score": min_score, "note": "Historical backtest now identifies raw breakouts, shows confirmation rejection counts, and simulates underlying index points. Six index-side confirmations are testable historically; ATM option momentum is not fabricated from today's option chain."})
    except Exception as e:
        logger.exception("Breakout backtest failed")
        return jsonify({"error": f"Backtest failed: {e}"}), 500


@app.route("/api/nifty/bullish-breakout")
def nifty_bullish_breakout_route():
    """Read-only NIFTY bullish breakout quality scan; never places an order."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        interval = request.args.get("interval", "5minute")
        lookback = max(10, min(int(request.args.get("lookback", NIFTY_BREAKOUT_DEFAULT_LOOKBACK)), 100))
        candles, err = _fetch_recent_intraday("NIFTY", interval, max(lookback, 50))
        if err:
            return jsonify({"error": err}), 400
        chain_data, chain_err = get_chain_for_symbol("NIFTY")
        result, result_err = _nifty_bullish_breakout(candles, chain_data if not chain_err else None, lookback)
        if result_err:
            return jsonify({"error": result_err}), 400
        if chain_err:
            result["call_side_note"] = f"Option-chain momentum unavailable: {chain_err.get('error', chain_err)}"
        return jsonify({"signal": result})
    except Exception as e:
        logger.exception("NIFTY bullish breakout scan failed")
        return jsonify({"error": f"NIFTY breakout scan failed: {e}"}), 500


@app.route("/api/autotrade/state")
def autotrade_state_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = _autotrade_roll_day_if_needed(load_autotrade_state())
    save_autotrade_state(state)
    trades = load_autotrade_trades()
    open_trades = [t for t in trades if t["status"] == "open"]
    recent_trades = sorted(trades, key=lambda t: t["opened_at"], reverse=True)[:25]
    return jsonify({"state": state, "open_trades": open_trades, "recent_trades": recent_trades})


@app.route("/api/autotrade/config", methods=["POST"])
def autotrade_config():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    state = load_autotrade_state()
    for key in CONFIGURABLE_AUTOTRADE_KEYS:
        if key in body:
            state[key] = body[key]
    save_autotrade_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/autotrade/set-execution-mode", methods=["POST"])
def autotrade_set_execution_mode():
    """Switches between Track (paper, default) and Live (real orders) execution. Deliberately kept
    OUT of CONFIGURABLE_AUTOTRADE_KEYS / the bulk /api/autotrade/config route -- a stray "Save
    settings" click should never be able to silently flip a paper setup into placing real orders.
    Switching to Live requires the same explicit ack pattern as arming Auto mode. This only affects
    FUTURE entries -- any already-open trade keeps behaving under the execution_mode it was opened
    with (see trade["execution_mode"])."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    mode = body.get("mode")
    if mode not in ("live", "track"):
        return jsonify({"error": "mode must be 'live' or 'track'"}), 400
    if mode == "live" and body.get("ack") is not True:
        return jsonify({"error": "Switching to Live mode requires explicit confirmation (ack: true) "
                                  "that future auto-trade entries and exits will place REAL orders on "
                                  "your live Zerodha account."}), 400
    state = load_autotrade_state()
    state["execution_mode"] = mode
    save_autotrade_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/autotrade/arm", methods=["POST"])
def autotrade_arm():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    mode = body.get("mode", "manual")
    if mode not in ("manual", "auto"):
        return jsonify({"error": "mode must be 'manual' or 'auto'"}), 400
    state = load_autotrade_state()
    if mode == "auto" and state.get("execution_mode", "track") == "live" and body.get("ack") is not True:
        return jsonify({"error": "Auto-execute mode in LIVE execution requires explicit confirmation "
                                  "(ack: true) that this places real orders automatically. (Track mode "
                                  "does not require this -- it never places real orders.)"}), 400
    state = _autotrade_roll_day_if_needed(state)
    state["mode"], state["enabled"], state["disarm_reason"] = mode, True, None
    save_autotrade_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/autotrade/disarm", methods=["POST"])
def autotrade_disarm():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = load_autotrade_state()
    state["enabled"] = False
    state["disarm_reason"] = "Manually stopped by user."
    save_autotrade_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/autotrade/close-all", methods=["POST"])
def autotrade_close_all():
    """Kill switch for POSITIONS (separate from /disarm, which only stops new entries): force-exits
    any currently open auto-trade at market, right now."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    trades = load_autotrade_trades()
    state = load_autotrade_state()
    closed_any = False
    for trade in trades:
        if trade["status"] == "open":
            try:
                q = kite_quote_bulk([f"NFO:{trade['tradingsymbol']}"]).get(f"NFO:{trade['tradingsymbol']}")
                trade["last_ltp"] = extract_price(q) or trade["last_ltp"]
            except Exception:
                pass
            _execute_autotrade_exit(trade, "Manual kill-switch close-all")
            state["realized_pnl_today"] = round(state.get("realized_pnl_today", 0.0) + (trade["realized_pnl"] or 0.0), 2)
            closed_any = True
    save_autotrade_trades(trades)
    save_autotrade_state(state)
    return jsonify({"ok": True, "closed_any": closed_any, "trades": trades})


@app.route("/api/autotrade/rescan-one", methods=["POST"])
def autotrade_rescan_one():
    """Re-checks ONE symbol right now and returns its CURRENT breakout score -- for when a
    candidate has been sitting in the list since an earlier scan and you want to know whether it's
    still breaking out (and how strongly) before deciding to trade it, without waiting for or
    triggering a full universe rescan."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol = body.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    state = load_autotrade_state()
    candles, err = _fetch_recent_intraday(symbol, state["candle_interval"], state["breakout_lookback"])
    if err:
        return jsonify({"error": err}), 400
    result = _detect_breakout(candles, state["breakout_lookback"], strict=state.get("strict_breakout_filters", True))
    if not result:
        return jsonify({"candidate": None,
                         "message": f"{symbol} is no longer breaking out -- price has moved back inside its range."})
    return jsonify({"candidate": _annotate_candidate(result, symbol, state["breakout_lookback"])})


@app.route("/api/autotrade/scan")
def autotrade_scan():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = load_autotrade_state()
    if state.get("scan_all_fo"):
        # Explicit, on-demand "Scan now" click -- OK to scan the whole live F&O list in one go
        # (unlike the background loop, this isn't repeated every poll_seconds); calls are still
        # staggered inside scan_breakouts() to respect Kite's rate limit, so this can take up to
        # roughly a minute for the full universe.
        try:
            universe = list(INDEX_SYMBOLS.keys()) + fo_stock_universe()
        except Exception as e:
            return jsonify({"error": f"Could not load full F&O universe: {e}"}), 500
    else:
        universe = state["universe"]
    candidates, errors = scan_breakouts(universe, state["candle_interval"], state["breakout_lookback"],
                                         strict=state.get("strict_breakout_filters", True))
    return jsonify({"candidates": candidates, "errors": errors, "universe_size": len(universe)})


@app.route("/api/autotrade/preview-signal", methods=["POST"])
def autotrade_preview_signal():
    """Read-only: resolves the EXACT contract (ATM strike, expiry, live premium, lot size, quantity)
    that Execute would buy for this candidate, WITHOUT placing anything. Lets the UI show something
    concrete -- e.g. "BSE 25AUG 9500 CE, qty 1200 (~4 lots) @ ~₹18.40, ~₹22,080 total" -- before the
    user commits, instead of them finding out what was bought only after the fact."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol, direction = body.get("symbol"), body.get("direction")
    if not symbol or direction not in ("CE", "PE"):
        return jsonify({"error": "symbol and direction (CE/PE) required"}), 400
    state = load_autotrade_state()
    order_info, err = _build_autotrade_order(symbol, direction, body.get("capital_per_trade", state["capital_per_trade"]))
    if err:
        return jsonify({"error": err}), 400
    order_info["still_valid"] = _breakout_still_valid(direction, body.get("breakout_level"), order_info["spot"])
    return jsonify({"order_info": order_info})


@app.route("/api/autotrade/execute-signal", methods=["POST"])
def autotrade_execute_signal():
    """Manual-mode execution: places the ONE entry order for a candidate the user reviewed and
    explicitly clicked 'Execute' on."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    if not body.get("confirmed"):
        return jsonify({"error": "Confirmation flag not set -- nothing was placed."}), 400
    candidate = body.get("candidate")
    if not candidate or not candidate.get("symbol") or not candidate.get("direction"):
        return jsonify({"error": "candidate with symbol/direction required"}), 400
    state = _autotrade_roll_day_if_needed(load_autotrade_state())
    trades = load_autotrade_trades()
    open_count = sum(1 for t in trades if t["status"] == "open")
    max_positions = max(1, int(state.get("max_concurrent_positions", 1)))
    if open_count >= max_positions:
        return jsonify({"error": f"Max concurrent positions ({max_positions}) already open. Raise "
                                  f"'Max concurrent positions' in Settings, or close one first."}), 400
    trade, err = _execute_autotrade_entry(candidate, state)
    if err:
        return jsonify({"error": err}), 400
    trades.append(trade)
    state["trades_today"] = state.get("trades_today", 0) + 1
    save_autotrade_trades(trades)
    save_autotrade_state(state)
    return jsonify({"ok": True, "trade": trade})


# =============================================================================
# Dynamic Delta-Neutral Adjustment Engine — ADD-ON to the existing Iron Condor / Hedged Short
# Strangle strategy above. Never touches how a position is opened (build_strategy, execute_confirm
# are untouched); it only watches already-open positions you explicitly attach to it and proposes/
# executes adjustments once net delta drifts too far. See greeks.py / risk_management.py /
# adjustment_engine.py / execution.py / delta_neutral_logging.py / backtesting.py /
# delta_neutral_state.py for the individual modules this wires together.
# =============================================================================
DELTA_POSITIONS_FILE = os.path.join(os.path.dirname(__file__), "delta_engine_positions.json")
DELTA_LOG_FILE = os.path.join(os.path.dirname(__file__), "delta_engine_log.jsonl")
_delta_positions_lock = threading.Lock()

_delta_greeks_engine = PortfolioGreeksEngine(risk_free_rate=RISK_FREE_RATE)
_delta_logger = AdjustmentLogger(DELTA_LOG_FILE)

# Tick-level spot cache: refreshed by a dedicated 1-second-cadence thread per configured symbol so
# the "monitor spot every tick" requirement doesn't depend on the heavier option-chain refresh rate.
# A true broker websocket (KiteTicker) tick feed is a straightforward future upgrade of just this
# cache's refresh mechanism -- everything downstream reads from this dict, not from the network call
# directly, so swapping REST polling for a websocket callback later doesn't touch any other module.
_delta_spot_cache = {}
_delta_spot_cache_lock = threading.Lock()

SPOT_INDEX_TRADINGSYMBOL = {
    "NIFTY": "NSE:NIFTY 50", "BANKNIFTY": "NSE:NIFTY BANK",
    "FINNIFTY": "NSE:NIFTY FIN SERVICE", "MIDCPNIFTY": "NSE:NIFTY MID SELECT",
}


def load_delta_positions():
    if not os.path.exists(DELTA_POSITIONS_FILE):
        return []
    try:
        with open(DELTA_POSITIONS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_delta_positions(positions):
    with _delta_positions_lock:
        with open(DELTA_POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)


def _delta_spot_poll_loop():
    """Refreshes _delta_spot_cache at dn_state's spot_poll_seconds cadence (default 1s) for every
    symbol currently being monitored -- this is the tick-level spot watch. Cheap: one LTP call per
    symbol, not the full option chain."""
    while True:
        try:
            state = dn_load_state()
            if state.get("enabled"):
                monitored_symbols = {p["symbol"] for p in load_delta_positions() if p["status"] == "monitoring"}
                for sym in monitored_symbols:
                    inst = SPOT_INDEX_TRADINGSYMBOL.get(sym)
                    if not inst:
                        continue
                    try:
                        ltp_data = kite.ltp([inst])
                        price = ltp_data.get(inst, {}).get("last_price")
                        if price:
                            with _delta_spot_cache_lock:
                                _delta_spot_cache[sym] = price
                    except Exception:
                        pass
            time.sleep(max(state.get("spot_poll_seconds", 1), 1))
        except Exception:
            time.sleep(2)


def _make_delta_candidate_fetcher(position, chain_data, state):
    """Returns a candidate_fetcher closure bound to one already-fetched live option chain snapshot
    (avoids re-hitting the API once per candidate). Builds real, tradable roll/hedge candidates —
    see adjustment_engine.py's module docstring for the up/down scenario -> threatened-side mapping."""
    calls = sorted([o for o in chain_data["chain"] if o["instrument_type"] == "CE"], key=lambda x: x["strike"])
    puts = sorted([o for o in chain_data["chain"] if o["instrument_type"] == "PE"], key=lambda x: x["strike"])
    legs = position["legs"]
    qty = position["quantity"]
    low, high = state.get("delta_range_low", 0.15), state.get("delta_range_high", 0.20)
    mid_target = (low + high) / 2
    hedge_pct = state.get("hedge_distance_pct", 2.5) / 100.0
    min_premium = state.get("min_premium_for_adjustment", 8.0)

    def find_by_delta(options, target, sign):
        return min(options, key=lambda o: abs(o["delta"] - sign * target)) if options else None

    def leg_dict(o, role):
        return {"tradingsymbol": o["tradingsymbol"], "strike": o["strike"], "role": role,
                "quantity": -qty if role.startswith("sell") else qty, "ltp": o["ltp"], "delta": o["delta"]}

    def close_leg(role):
        return {"tradingsymbol": legs[role]["tradingsymbol"], "role": role, "quantity": -qty}

    def fetcher(scenario, pos, portfolio_greeks):
        candidates = []
        if scenario == "up":
            # Priority 1: roll the short PUT upward -- collects more premium AND adds offsetting
            # positive delta (see adjustment_engine.py docstring for why this offsets negative net delta).
            cur_put_k = legs["sell_put"]["strike"]
            pool = [o for o in puts if o["strike"] > cur_put_k]
            new_put = find_by_delta(pool, mid_target, -1)
            if new_put and new_put["ltp"] >= min_premium:
                delta_gain = abs(new_put["delta"] - legs["sell_put"]["delta"]) * qty
                candidates.append(dict(
                    action="roll_put_up",
                    description=f"Roll short put {cur_put_k} -> {new_put['strike']} (more premium, offsetting positive delta)",
                    legs_to_close=[close_leg("sell_put")], legs_to_open=[leg_dict(new_put, "sell_put")],
                    expected_delta_after=portfolio_greeks.net_delta + delta_gain,
                    additional_premium=(new_put["ltp"] - legs["sell_put"]["ltp"]) * qty,
                    margin_impact=0.0, risk_reduction_score=min(1.0, delta_gain / max(state.get("delta_threshold", 10.0), 1)),
                    probability_of_profit=0.60, expected_drawdown=abs(new_put["strike"] - cur_put_k) * qty * 0.3,
                ))
            # Priority 2: roll the short CALL further OTM -- directly cuts the threatened leg's delta.
            cur_call_k = legs["sell_call"]["strike"]
            pool = [o for o in calls if o["strike"] > cur_call_k]
            new_call = find_by_delta(pool, mid_target, +1)
            if new_call:
                delta_gain = abs(new_call["delta"] - legs["sell_call"]["delta"]) * qty
                candidates.append(dict(
                    action="roll_call_further_otm",
                    description=f"Roll short call {cur_call_k} -> {new_call['strike']} further OTM (cuts delta exposure directly)",
                    legs_to_close=[close_leg("sell_call")], legs_to_open=[leg_dict(new_call, "sell_call")],
                    expected_delta_after=portfolio_greeks.net_delta + delta_gain,
                    additional_premium=(new_call["ltp"] - legs["sell_call"]["ltp"]) * qty,
                    margin_impact=0.0, risk_reduction_score=min(1.0, delta_gain / max(state.get("delta_threshold", 10.0), 1)),
                    probability_of_profit=0.55, expected_drawdown=abs(new_call["strike"] - cur_call_k) * qty * 0.3,
                ))
        else:  # "down" -- mirror image, put side under stress
            cur_call_k = legs["sell_call"]["strike"]
            pool = [o for o in calls if o["strike"] < cur_call_k]
            new_call = find_by_delta(pool, mid_target, +1)
            if new_call and new_call["ltp"] >= min_premium:
                delta_gain = abs(new_call["delta"] - legs["sell_call"]["delta"]) * qty
                candidates.append(dict(
                    action="roll_call_down",
                    description=f"Roll short call {cur_call_k} -> {new_call['strike']} (more premium, offsetting negative delta)",
                    legs_to_close=[close_leg("sell_call")], legs_to_open=[leg_dict(new_call, "sell_call")],
                    expected_delta_after=portfolio_greeks.net_delta - delta_gain,
                    additional_premium=(new_call["ltp"] - legs["sell_call"]["ltp"]) * qty,
                    margin_impact=0.0, risk_reduction_score=min(1.0, delta_gain / max(state.get("delta_threshold", 10.0), 1)),
                    probability_of_profit=0.60, expected_drawdown=abs(new_call["strike"] - cur_call_k) * qty * 0.3,
                ))
            cur_put_k = legs["sell_put"]["strike"]
            pool = [o for o in puts if o["strike"] < cur_put_k]
            new_put = find_by_delta(pool, mid_target, -1)
            if new_put:
                delta_gain = abs(new_put["delta"] - legs["sell_put"]["delta"]) * qty
                candidates.append(dict(
                    action="roll_put_further_otm",
                    description=f"Roll short put {cur_put_k} -> {new_put['strike']} further OTM (cuts delta exposure directly)",
                    legs_to_close=[close_leg("sell_put")], legs_to_open=[leg_dict(new_put, "sell_put")],
                    expected_delta_after=portfolio_greeks.net_delta - delta_gain,
                    additional_premium=(new_put["ltp"] - legs["sell_put"]["ltp"]) * qty,
                    margin_impact=0.0, risk_reduction_score=min(1.0, delta_gain / max(state.get("delta_threshold", 10.0), 1)),
                    probability_of_profit=0.55, expected_drawdown=abs(new_put["strike"] - cur_put_k) * qty * 0.3,
                ))

        # Convert to Iron Fly: only meaningful for an existing iron_condor (hedges already in place) --
        # rolls BOTH short strikes to ATM for extra premium in exchange for a much tighter profit zone.
        if position.get("strategy_type") == "iron_condor" and calls and puts:
            atm_call = min(calls, key=lambda o: abs(o["strike"] - portfolio_greeks.spot))
            atm_put = min(puts, key=lambda o: abs(o["strike"] - portfolio_greeks.spot))
            added_premium = ((atm_call["ltp"] - legs["sell_call"]["ltp"]) + (atm_put["ltp"] - legs["sell_put"]["ltp"])) * qty
            candidates.append(dict(
                action="convert_iron_fly",
                description=f"Convert to Iron Fly: roll both shorts to ATM ({atm_call['strike']}/{atm_put['strike']}) for extra premium, tighter body",
                legs_to_close=[close_leg("sell_call"), close_leg("sell_put")],
                legs_to_open=[leg_dict(atm_call, "sell_call"), leg_dict(atm_put, "sell_put")],
                expected_delta_after=(atm_call["delta"] + atm_put["delta"]) * qty,
                additional_premium=added_premium, margin_impact=2000.0, risk_reduction_score=0.9,
                probability_of_profit=0.45,   # tighter body -> lower POP even though delta-neutral
                expected_drawdown=abs(portfolio_greeks.spot - legs["sell_call"]["strike"]) * qty * 0.2,
            ))

        # Add-hedge fallback: only meaningful for a naked strangle (an iron condor already carries
        # hedges) -- buys a protective option on the threatened side to hard-cap runaway delta/risk.
        if position.get("strategy_type") == "naked_strangle":
            if scenario == "up" and calls:
                target_k = legs["sell_call"]["strike"] * (1 + hedge_pct)
                hedge_opt = min(calls, key=lambda o: abs(o["strike"] - target_k))
                candidates.append(dict(
                    action="add_hedge",
                    description=f"Buy protective call {hedge_opt['strike']} ({hedge_pct*100:.1f}% OTM of short call) to cap runaway risk",
                    legs_to_close=[], legs_to_open=[leg_dict(hedge_opt, "buy_call")],
                    expected_delta_after=portfolio_greeks.net_delta + abs(hedge_opt["delta"]) * qty,
                    additional_premium=-hedge_opt["ltp"] * qty, margin_impact=-5000.0,
                    risk_reduction_score=0.8, probability_of_profit=0.5, expected_drawdown=hedge_opt["ltp"] * qty,
                ))
            elif scenario == "down" and puts:
                target_k = legs["sell_put"]["strike"] * (1 - hedge_pct)
                hedge_opt = min(puts, key=lambda o: abs(o["strike"] - target_k))
                candidates.append(dict(
                    action="add_hedge",
                    description=f"Buy protective put {hedge_opt['strike']} ({hedge_pct*100:.1f}% OTM of short put) to cap runaway risk",
                    legs_to_close=[], legs_to_open=[leg_dict(hedge_opt, "buy_put")],
                    expected_delta_after=portfolio_greeks.net_delta - abs(hedge_opt["delta"]) * qty,
                    additional_premium=-hedge_opt["ltp"] * qty, margin_impact=-5000.0,
                    risk_reduction_score=0.8, probability_of_profit=0.5, expected_drawdown=hedge_opt["ltp"] * qty,
                ))
        return candidates
    return fetcher


def _delta_position_portfolio_greeks(position, chain_data, spot):
    """Builds a PortfolioGreeks snapshot for one monitored position from a fresh chain fetch."""
    chain_by_symbol = {o["tradingsymbol"]: o for o in chain_data["chain"]}
    T = chain_data["T"]
    qty = position["quantity"]
    leg_inputs = []
    for role, leg in position["legs"].items():
        live = chain_by_symbol.get(leg["tradingsymbol"])
        ltp = live["ltp"] if live else leg["ltp"]
        iv = (live["iv"] / 100.0) if live else 0.15
        opt_type = "CE" if "call" in role else "PE"
        leg_inputs.append({
            "opt_type": opt_type, "strike": leg["strike"], "spot": spot, "T": T, "iv": iv, "ltp": ltp,
            "quantity": -qty if role.startswith("sell") else qty, "role": role,
            "tradingsymbol": leg["tradingsymbol"], "entry_premium": leg["ltp"],
        })
    return _delta_greeks_engine.portfolio_greeks(position["symbol"], spot, leg_inputs)


def _execute_delta_adjustment(position, candidate, trigger_reason, all_candidate_actions=None):
    """Executes ONE specific adjustment candidate against one position -- shared by the manual
    /api/delta-engine/suggestions/<id>/execute route (the only normal path now) and reusable for any
    future automation. Risk gates are re-checked fresh here regardless of what was true when the
    suggestion was first generated, so a stale approval can never bypass current limits."""
    state = dn_load_state()
    symbol = position["symbol"]
    adj_by_symbol = state.get("adjustments_today_by_symbol", {})
    mtm_by_symbol = state.get("daily_mtm_by_symbol", {})
    paused_symbols = set(state.get("paused_symbols_today", []))
    if symbol in paused_symbols:
        return False, f"{symbol} is paused for today (its own daily-loss limit tripped) -- adjustment blocked"

    risk_mgr = RiskManager(RiskLimits(
        max_adjustments_per_day=state.get("max_adjustments_per_day", 6),
        max_loss_per_position=state.get("max_loss_per_position", 15000.0),
        max_daily_mtm_loss=state.get("max_daily_mtm_loss", 25000.0),
        min_premium_for_adjustment=state.get("min_premium_for_adjustment", 8.0),
        profit_targets_pct=tuple(state.get("profit_targets_pct", [25, 50, 70, 90])),
        stop_loss_pct=state.get("stop_loss_pct", 200.0),
    ))
    position_mtm = (position.get("last_greeks") or {}).get("mtm", 0.0)
    proposed_premium = abs(candidate.legs_to_open[0]["ltp"]) if candidate.legs_to_open else None
    allowed, gate_reason = risk_mgr.can_adjust(
        adjustments_today=adj_by_symbol.get(symbol, 0), position_mtm=position_mtm,
        daily_mtm=mtm_by_symbol.get(symbol, 0.0), proposed_leg_premium=proposed_premium)
    if not allowed:
        return False, gate_reason

    executor = AdjustmentExecutor(place_basket_orders, product="NRML")
    result = executor.execute(candidate, execution_mode=state.get("execution_mode", "track"))
    if not result.ok:
        return False, f"Execution failed: {result.error}"

    for opened in candidate.legs_to_open:
        position["legs"][opened["role"]] = {
            "strike": opened["strike"], "ltp": opened["ltp"], "tradingsymbol": opened["tradingsymbol"],
        }
    position["pending_suggestion"] = None
    adj_by_symbol[symbol] = adj_by_symbol.get(symbol, 0) + 1
    state["adjustments_today_by_symbol"] = adj_by_symbol
    dn_save_state(state)
    _delta_logger.log_adjustment(
        ts=now_ist().isoformat(), symbol=symbol, spot=position.get("last_greeks", {}).get("spot"),
        delta_before=position.get("last_greeks", {}).get("net_delta", 0.0),
        delta_after=candidate.expected_delta_after, action=candidate.action,
        premium_collected=candidate.additional_premium, reason=trigger_reason,
        execution_mode=state.get("execution_mode", "track"),
        candidates_considered=all_candidate_actions or [candidate.action])
    return True, "Executed"


def _close_delta_position_legs(position, roles, reason):
    """Closes some or all legs of a monitored position on demand -- independent of the automatic
    profit-target/stop-loss logic, so you can act on your own judgment any time, in Track or Live."""
    state = dn_load_state()
    legs = position["legs"]
    roles = roles or list(legs.keys())
    close_legs = []
    for role in roles:
        if role not in legs:
            continue
        close_legs.append({"tradingsymbol": legs[role]["tradingsymbol"], "role": role,
                            "quantity": -position["quantity"] if role.startswith("sell") else position["quantity"]})
    if not close_legs:
        return False, "No matching open legs found to close"
    candidate = AdjustmentCandidate(action="manual_close", description=reason, legs_to_close=close_legs)
    executor = AdjustmentExecutor(place_basket_orders, product="NRML")
    result = executor.execute(candidate, execution_mode=state.get("execution_mode", "track"))
    if not result.ok:
        return False, f"Close failed: {result.error}"
    for role in roles:
        legs.pop(role, None)
    pnl_closed = (position.get("last_greeks") or {}).get("mtm", 0.0) if not legs else None
    if not legs:
        position["status"] = "closed"
        position["closed_at"] = now_ist().isoformat()
        position["exit_reason"] = reason
        position["realized_pnl"] = pnl_closed
        position["pending_suggestion"] = None
    _delta_logger.log_adjustment(
        ts=now_ist().isoformat(), symbol=position["symbol"], spot=(position.get("last_greeks") or {}).get("spot"),
        delta_before=(position.get("last_greeks") or {}).get("net_delta", 0.0), delta_after=0.0,
        action="manual_close", premium_collected=0.0, reason=reason,
        execution_mode=state.get("execution_mode", "track"), candidates_considered=[])
    return True, "Closed"


def _delta_engine_loop():
    """Background monitor loop for the Delta Neutral Adjustment Engine -- disarmed by default, only
    watches/acts on positions you've explicitly attached (via the Condor/Strangle builder's Position
    ID, or directly from your live broker positions). Recomputes portfolio Greeks and evaluates the
    adjustment threshold at greeks_poll_seconds cadence (default 5s); the underlying spot itself is
    refreshed far more often by _delta_spot_poll_loop (default 1s).

    IMPORTANT: this loop only ever ANALYZES and SUGGESTS. When a delta breach is detected it scores
    every candidate adjustment (probability of profit, risk reduction, expected extra premium, etc.)
    and stores them as position["pending_suggestion"] -- it does NOT execute anything automatically.
    You review the suggestion in the dashboard and choose whether/which one to execute, in Track or
    Live, via /api/delta-engine/suggestions/<id>/execute. The one thing that DOES still happen
    automatically is closing a position on a configured profit target or stop loss being hit -- that
    is risk protection, not a trading decision, so it isn't gated behind manual approval.

    Every symbol is analyzed and decided on INDEPENDENTLY: its own Greeks, its own delta-threshold
    check, its own adjustments-per-day / daily-loss budget. A breach on one symbol only pauses NEW
    adjustments for that specific symbol for the rest of the day; profit-target/stop-loss exits keep
    working per-position regardless of pause state."""
    while True:
        try:
            state = dn_load_state()
            today_str = now_ist().strftime("%Y-%m-%d")
            if state.get("day") != today_str:
                state["day"] = today_str
                state["adjustments_today_by_symbol"] = {}
                state["daily_mtm_by_symbol"] = {}
                state["paused_symbols_today"] = []
                dn_save_state(state)

            if not state.get("enabled"):
                time.sleep(2)
                continue

            adj_by_symbol = state.get("adjustments_today_by_symbol", {})
            mtm_by_symbol = state.get("daily_mtm_by_symbol", {})
            paused_symbols = set(state.get("paused_symbols_today", []))

            risk_mgr = RiskManager(RiskLimits(
                max_adjustments_per_day=state.get("max_adjustments_per_day", 6),
                max_loss_per_position=state.get("max_loss_per_position", 15000.0),
                max_daily_mtm_loss=state.get("max_daily_mtm_loss", 25000.0),
                min_premium_for_adjustment=state.get("min_premium_for_adjustment", 8.0),
                profit_targets_pct=tuple(state.get("profit_targets_pct", [25, 50, 70, 90])),
                stop_loss_pct=state.get("stop_loss_pct", 200.0),
            ))
            engine = AdjustmentEngine(delta_threshold=state.get("delta_threshold", 10.0),
                                       gamma_threshold=state.get("gamma_threshold"))
            executor = AdjustmentExecutor(place_basket_orders, product="NRML")

            positions = load_delta_positions()
            for position in positions:
                if position["status"] != "monitoring":
                    continue
                symbol = position["symbol"]
                try:
                    chain_data, err = get_chain_for_symbol(symbol)
                    if err:
                        state["last_error"] = f"{symbol}: {err.get('error')}"
                        continue
                    with _delta_spot_cache_lock:
                        spot = _delta_spot_cache.get(symbol, chain_data["spot"])
                    pg = _delta_position_portfolio_greeks(position, chain_data, spot)
                    position["last_greeks"] = pg.as_dict()
                    # Accumulate THIS symbol's daily MTM only -- separate bucket per symbol, never
                    # pooled with any other symbol's positions.
                    mtm_by_symbol[symbol] = round(mtm_by_symbol.get(symbol, 0.0) + pg.mtm, 2)

                    # Profit target / stop loss -- the ONE thing that stays automatic (risk
                    # protection, not a discretionary trading decision). Exit all legs immediately.
                    hit_target = risk_mgr.profit_target_hit(position.get("entry_credit", 0.0), pg.mtm)
                    hit_sl = risk_mgr.stop_loss_hit(position.get("entry_credit", 0.0), pg.mtm)
                    if hit_target or hit_sl:
                        reason = f"Profit target {hit_target}% reached" if hit_target else f"Stop loss ({state.get('stop_loss_pct')}%) hit"
                        close_legs = [{"tradingsymbol": leg["tradingsymbol"], "role": role,
                                       "quantity": -position["quantity"] if role.startswith("sell") else position["quantity"]}
                                      for role, leg in position["legs"].items()]
                        close_candidate = AdjustmentCandidate(action="close_all", description=reason,
                                                               legs_to_close=close_legs)
                        executor.execute(close_candidate, execution_mode=state.get("execution_mode", "track"))
                        position["status"] = "closed"
                        position["closed_at"] = now_ist().isoformat()
                        position["exit_reason"] = reason
                        position["realized_pnl"] = pg.mtm
                        position["pending_suggestion"] = None
                        _delta_logger.log_adjustment(ts=now_ist().isoformat(), symbol=symbol,
                            spot=spot, delta_before=pg.net_delta, delta_after=0.0, action="close_all",
                            premium_collected=pg.mtm, reason=reason, execution_mode=state.get("execution_mode", "track"))
                        continue

                    # This symbol's OWN daily-loss budget -- checked independently of every other symbol.
                    if risk_mgr.breached_daily_loss(mtm_by_symbol[symbol]) and symbol not in paused_symbols:
                        paused_symbols.add(symbol)
                        state["last_error"] = f"{symbol}: daily MTM loss (Rs {mtm_by_symbol[symbol]:.0f}) breached its own limit -- new adjustments paused for {symbol} only for the rest of today"

                    recommended, all_candidates, trigger_reason = engine.recommend(
                        position, pg, _make_delta_candidate_fetcher(position, chain_data, state))
                    state["last_recommendation"] = {
                        "symbol": symbol, "trigger_reason": trigger_reason,
                        "recommended": vars(recommended) if recommended else None,
                        "at": now_ist().isoformat(),
                    }
                    if recommended is None or recommended.action == "no_action":
                        position["pending_suggestion"] = None
                        continue

                    # Annotate each candidate with whether it's currently executable, so the
                    # dashboard can show "blocked: <reason>" without you having to click Execute
                    # to find out -- but the ACTUAL gate is always re-checked fresh at execute time.
                    annotated = []
                    for c in all_candidates:
                        proposed_premium = abs(c.legs_to_open[0]["ltp"]) if c.legs_to_open else None
                        ok, reason_txt = risk_mgr.can_adjust(
                            adjustments_today=adj_by_symbol.get(symbol, 0), position_mtm=pg.mtm,
                            daily_mtm=mtm_by_symbol[symbol], proposed_leg_premium=proposed_premium)
                        if symbol in paused_symbols:
                            ok, reason_txt = False, f"{symbol} paused today (daily-loss limit)"
                        cd = vars(c)
                        cd["executable"] = ok
                        cd["block_reason"] = None if ok else reason_txt
                        annotated.append(cd)

                    position["pending_suggestion"] = {
                        "trigger_reason": trigger_reason, "detected_at": now_ist().isoformat(),
                        "candidates": annotated,
                    }
                except Exception as e:
                    state["last_error"] = f"{symbol}: {e}"

            state["adjustments_today_by_symbol"] = adj_by_symbol
            state["daily_mtm_by_symbol"] = mtm_by_symbol
            state["paused_symbols_today"] = sorted(paused_symbols)
            state["last_scan_at"] = now_ist().isoformat()

            save_delta_positions(positions)
            dn_save_state(state)
        except Exception as e:
            try:
                state = dn_load_state()
                state["last_error"] = str(e)
                dn_save_state(state)
            except Exception:
                pass
        time.sleep(max(dn_load_state().get("greeks_poll_seconds", 5), 2))


# --- Delta Neutral Engine API routes ---
@app.route("/api/delta-engine/state")
def delta_engine_state():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    return jsonify(dn_load_state())


@app.route("/api/delta-engine/config", methods=["POST"])
def delta_engine_config():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    state = dn_load_state()
    for key in DELTA_ENGINE_CONFIGURABLE_KEYS:
        if key in body:
            state[key] = body[key]
    dn_save_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/delta-engine/set-execution-mode", methods=["POST"])
def delta_engine_set_execution_mode():
    """Same ack-gated pattern as /api/autotrade/set-execution-mode -- kept OUT of the bulk config
    route so a stray Settings save can never silently switch this to placing real orders."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    mode = body.get("mode")
    if mode not in ("live", "track"):
        return jsonify({"error": "mode must be 'live' or 'track'"}), 400
    if mode == "live" and body.get("ack") is not True:
        return jsonify({"error": "Switching to Live mode requires explicit confirmation (ack: true) "
                                  "that future adjustments will place REAL orders on your live Zerodha account."}), 400
    state = dn_load_state()
    state["execution_mode"] = mode
    dn_save_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/delta-engine/arm", methods=["POST"])
def delta_engine_arm():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = dn_load_state()
    state["enabled"], state["disarm_reason"] = True, None
    dn_save_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/delta-engine/disarm", methods=["POST"])
def delta_engine_disarm():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = dn_load_state()
    state["enabled"] = False
    dn_save_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/delta-engine/track/<pos_id>", methods=["POST"])
def delta_engine_track(pos_id):
    """Attaches an already-built Iron Condor / Strangle position (from positions.json, built via the
    existing /api/strategy endpoint) to the Delta Neutral Engine for monitoring. Does not place any
    order itself -- the position must already be a real (or, in execution_mode="track", intended)
    open position."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    position = find_position(pos_id)
    if not position:
        return jsonify({"error": "Position not found"}), 404
    if position.get("strategy_type") not in ("iron_condor", "naked_strangle"):
        return jsonify({"error": "Only Iron Condor / Naked Strangle positions are supported by the Delta Neutral Engine"}), 400
    legs = position["legs"]
    credit_per_share = sum(v["ltp"] for k, v in legs.items() if k.startswith("sell")) \
        - sum(v["ltp"] for k, v in legs.items() if k.startswith("buy"))
    tracked = {
        "id": pos_id, "symbol": position["symbol"], "strategy_type": position["strategy_type"],
        "legs": {k: {"strike": v["strike"], "ltp": v["ltp"], "tradingsymbol": v["tradingsymbol"]} for k, v in legs.items()},
        "quantity": position.get("quantity", position["lot_size"]), "lot_size": position["lot_size"],
        "entry_credit": round(credit_per_share * position.get("quantity", position["lot_size"]), 2),
        "status": "monitoring", "added_at": now_ist().isoformat(), "closed_at": None,
        "exit_reason": None, "realized_pnl": None, "last_greeks": None, "pending_suggestion": None,
    }
    positions = load_delta_positions()
    positions = [p for p in positions if p["id"] != pos_id]
    positions.append(tracked)
    save_delta_positions(positions)
    return jsonify({"ok": True, "tracked": tracked})


def _classify_broker_legs_for_symbol(symbol):
    """Finds every open NFO option leg under a given underlying (e.g. "SBIN") straight from your
    live Zerodha positions (the same data as the "Open F&O positions" table), and classifies each
    one as sell_call / buy_call / sell_put / buy_put using Kite's own authoritative instrument dump
    (strike/instrument_type per tradingsymbol) rather than guessing from the tradingsymbol text.
    This is what lets you attach a position you opened directly in Zerodha (or anywhere else) --
    no positions.json entry from this app's own Condor/Strangle builder is required."""
    symbol = symbol.upper()
    nfo, _ = get_instruments()
    meta_by_symbol = {i["tradingsymbol"]: i for i in nfo if i.get("name") == symbol and i.get("segment") == "NFO-OPT"}
    pos = kite.positions()
    net = pos.get("net", [])
    legs_by_role = {}
    quantity = None
    lot_size = None
    for p in net:
        ts = p.get("tradingsymbol")
        if p.get("exchange") != "NFO" or ts not in meta_by_symbol:
            continue
        qty = int(p.get("quantity") or 0)
        if qty == 0:
            continue
        meta = meta_by_symbol[ts]
        opt_type = meta.get("instrument_type")
        role = ("sell_" if qty < 0 else "buy_") + ("call" if opt_type == "CE" else "put")
        if role in legs_by_role:
            return None, (f"Found more than one open {role.replace('_', ' ')} leg on {symbol} -- "
                           f"this engine expects one clean 4-leg (or 2-leg) condor/strangle shape per "
                           f"symbol. Close/simplify down to one leg per role before attaching.")
        legs_by_role[role] = {
            "strike": meta.get("strike"), "tradingsymbol": ts,
            "ltp": p.get("last_price"), "entry_price": p.get("average_price"),
        }
        quantity = abs(qty)
        lot_size = meta.get("lot_size")
    if not legs_by_role or "sell_call" not in legs_by_role or "sell_put" not in legs_by_role:
        return None, (f"No open short call + short put pair found for {symbol} in your live Zerodha "
                       f"positions. This engine needs at least both short legs of the condor/strangle "
                       f"to be open right now.")
    strategy_type = "iron_condor" if {"buy_call", "buy_put"} <= legs_by_role.keys() else "naked_strangle"
    return {"symbol": symbol, "strategy_type": strategy_type, "legs": legs_by_role,
            "quantity": quantity, "lot_size": lot_size}, None


@app.route("/api/delta-engine/broker-legs/<symbol>")
def delta_engine_broker_legs(symbol):
    """Preview what would be attached for a symbol before committing -- lets the UI show the
    detected legs/strikes so you can confirm it matched the right position."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    result, err = _classify_broker_legs_for_symbol(symbol)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/api/delta-engine/attach-broker", methods=["POST"])
def delta_engine_attach_broker():
    """Attaches a position for monitoring straight from your LIVE Zerodha broker positions -- for
    a condor/strangle you opened outside this tool's own Condor/Strangle builder tab (e.g. placed
    directly in Kite), so there's no positions.json Position ID to type in. Just give the underlying
    symbol (e.g. "SBIN") and this finds and classifies the open legs itself."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    result, err = _classify_broker_legs_for_symbol(symbol)
    if err:
        return jsonify({"error": err}), 400
    legs = result["legs"]
    credit_per_share = (sum(v["entry_price"] for k, v in legs.items() if k.startswith("sell"))
                         - sum(v["entry_price"] for k, v in legs.items() if k.startswith("buy")))
    pos_id = f"BROKER-{symbol}-{int(time.time())}"
    tracked = {
        "id": pos_id, "symbol": symbol, "strategy_type": result["strategy_type"],
        "legs": {k: {"strike": v["strike"], "ltp": v["entry_price"], "tradingsymbol": v["tradingsymbol"]}
                 for k, v in legs.items()},
        "quantity": result["quantity"], "lot_size": result["lot_size"],
        "entry_credit": round(credit_per_share * result["quantity"], 2),
        "source": "broker", "status": "monitoring", "added_at": now_ist().isoformat(),
        "closed_at": None, "exit_reason": None, "realized_pnl": None, "last_greeks": None,
        "pending_suggestion": None,
    }
    positions = load_delta_positions()
    positions.append(tracked)
    save_delta_positions(positions)
    return jsonify({"ok": True, "tracked": tracked})


@app.route("/api/delta-engine/resume/<symbol>", methods=["POST"])
def delta_engine_resume_symbol(symbol):
    """Manually un-pauses a symbol whose own daily-loss limit tripped earlier today, if you've
    reviewed it and want the engine to resume proposing/executing adjustments for it (profit-target
    and stop-loss exits keep working for a paused symbol regardless -- this only affects new
    adjustments)."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    symbol = symbol.upper()
    state = dn_load_state()
    paused = [s for s in state.get("paused_symbols_today", []) if s != symbol]
    state["paused_symbols_today"] = paused
    dn_save_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/delta-engine/positions")
def delta_engine_positions():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    return jsonify({"positions": load_delta_positions()})


@app.route("/api/delta-engine/positions/<pos_id>/untrack", methods=["POST"])
def delta_engine_untrack(pos_id):
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_delta_positions()
    for p in positions:
        if p["id"] == pos_id:
            p["status"] = "closed"
            p["closed_at"] = now_ist().isoformat()
            p["exit_reason"] = "Manually untracked"
    save_delta_positions(positions)
    return jsonify({"ok": True})


@app.route("/api/delta-engine/suggestions")
def delta_engine_suggestions():
    """Every pending adjustment suggestion, one per monitored position that currently has a delta
    breach -- each candidate carries its probability of profit, risk-reduction score, expected extra
    premium, and whether it's currently executable given today's risk budget. Nothing here has been
    or will be executed automatically; you choose what (if anything) to act on."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_delta_positions()
    out = []
    for p in positions:
        if p["status"] == "monitoring" and p.get("pending_suggestion"):
            out.append({"position_id": p["id"], "symbol": p["symbol"], **p["pending_suggestion"]})
    return jsonify({"suggestions": out})


@app.route("/api/delta-engine/suggestions/<pos_id>/execute", methods=["POST"])
def delta_engine_execute_suggestion(pos_id):
    """Executes ONE specific candidate from a position's current pending suggestion -- the only way
    an adjustment is ever placed. Runs in whichever execution_mode (Track/Live) is currently set."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    action = body.get("action")
    if not action:
        return jsonify({"error": "action is required (the candidate's action name to execute)"}), 400
    positions = load_delta_positions()
    position = next((p for p in positions if p["id"] == pos_id), None)
    if not position:
        return jsonify({"error": "Position not found"}), 404
    suggestion = position.get("pending_suggestion")
    if not suggestion:
        return jsonify({"error": "No pending suggestion for this position -- it may have already been acted on or cleared"}), 400
    candidate_dict = next((c for c in suggestion["candidates"] if c["action"] == action), None)
    if not candidate_dict:
        return jsonify({"error": f"No candidate with action '{action}' in the current suggestion"}), 400
    candidate = AdjustmentCandidate(**{k: v for k, v in candidate_dict.items() if k in AdjustmentCandidate.__dataclass_fields__})
    ok, message = _execute_delta_adjustment(position, candidate, suggestion["trigger_reason"],
                                             all_candidate_actions=[c["action"] for c in suggestion["candidates"]])
    save_delta_positions(positions)
    if not ok:
        return jsonify({"error": message}), 400
    return jsonify({"ok": True, "message": message, "position": position})


@app.route("/api/delta-engine/suggestions/<pos_id>/dismiss", methods=["POST"])
def delta_engine_dismiss_suggestion(pos_id):
    """Clears the current pending suggestion without executing anything -- the next monitoring cycle
    will re-evaluate and generate a fresh one if the breach is still there."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    positions = load_delta_positions()
    for p in positions:
        if p["id"] == pos_id:
            p["pending_suggestion"] = None
    save_delta_positions(positions)
    return jsonify({"ok": True})


@app.route("/api/delta-engine/positions/<pos_id>/close-legs", methods=["POST"])
def delta_engine_close_legs(pos_id):
    """Manually closes some or all legs of a monitored position, in Track or Live (whichever
    execution_mode is currently set), independent of the automatic profit-target/stop-loss logic --
    for when you'd rather act on your own judgment. Body: {"roles": ["sell_call", ...]} -- omit or
    pass an empty list to close every open leg."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    roles = body.get("roles") or []
    positions = load_delta_positions()
    position = next((p for p in positions if p["id"] == pos_id), None)
    if not position:
        return jsonify({"error": "Position not found"}), 404
    reason = body.get("reason") or ("Manual close (all legs)" if not roles else f"Manual close: {', '.join(roles)}")
    ok, message = _close_delta_position_legs(position, roles, reason)
    save_delta_positions(positions)
    if not ok:
        return jsonify({"error": message}), 400
    return jsonify({"ok": True, "message": message, "position": position})


@app.route("/api/delta-engine/logs")
def delta_engine_logs():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    limit = int(request.args.get("limit", 200))
    return jsonify({"logs": _delta_logger.read_recent(limit)})




# ============================================================================
# AUTONOMOUS AI EVOLUTION ENGINE
# Separate paper-trading/learning engine. It does NOT modify the existing
# Iron Condor, Calendar, Auto Trade or Delta Neutral execution logic.
# It uses the same Zerodha/Kite session and runs in the background after login.
# ============================================================================
import sqlite3
from statistics import mean
AI_DB_FILE = os.path.join(os.path.dirname(__file__), "ai_evolution.db")
AI_POLL_SECONDS = int(os.environ.get("AI_POLL_SECONDS", "60"))
AI_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY"]
AI_LOCK = threading.Lock()

AI_DEFAULTS = {
    "min_score": 72.0, "min_rr": 1.5, "stop_pct": 0.75, "target_pct": 1.50,
    "risk_per_trade_pct": 1.0, "max_positions": 3, "learning_min_trades": 40,
    "challenger_min_trades": 30, "enabled": 1
}

# ---------------------------------------------------------------------------
# Historical pre-training / ML / Deep-Learning / Research-ensemble additions
# ---------------------------------------------------------------------------
AI_MODEL_DIR = os.path.join(os.path.dirname(__file__), "ai_models")
os.makedirs(AI_MODEL_DIR, exist_ok=True)
AI_HIST_INTERVAL = "5minute"
AI_HIST_BACKFILL_DAYS = int(os.environ.get("AI_HIST_BACKFILL_DAYS", "1825"))  # default 5 years; set 1095 for 3y etc.
AI_HIST_CHUNK_DAYS = 90             # Kite's 5-minute interval allows up to ~100 days per call
AI_HIST_CHUNK_PAUSE_SECONDS = float(os.environ.get("AI_HIST_CHUNK_PAUSE_SECONDS", "0.35"))  # pacing between
                                     # historical_data() calls — Kite's historical endpoint has its own,
                                     # stricter rate limit than the general API; this avoids 429s on a
                                     # multi-year backfill, which needs ~20 chunk calls per symbol at 90d/chunk.
AI_HIST_POLL_SECONDS = int(os.environ.get("AI_HIST_POLL_SECONDS", "1800"))   # 30 min — incremental refresh only
AI_ML_RETRAIN_MIN_HOURS = 6         # ML (LogisticRegression) is cheap even at multi-year scale
AI_DL_RETRAIN_MIN_HOURS = 24        # GRU training cost scales with dataset size — cap to once/day so a
                                     # 5-year backfill doesn't try to retrain 3 symbols' GRUs every 30 min
AI_SEQ_LEN = 20                     # GRU sequence length ("20 sequential market states")
AI_ML_MIN_SAMPLES = 300             # min rows before the first ML model can train
AI_DL_MIN_SAMPLES = 500             # min sequences before the first DL model can train
AI_BARS_PER_DAY_EST = 75            # ~09:15-15:30 in 5-min bars, used only for progress display

# Well-known Zerodha index instrument tokens, used as a fallback if the live
# instrument dump lookup below doesn't resolve (dump format/segment can vary).
AI_INDEX_TOKEN_FALLBACK = {"NIFTY": 256265, "BANKNIFTY": 260105, "FINNIFTY": 257801,
                           "MIDCPNIFTY": 288009, "SENSEX": 265}

def ai_index_instrument_token(symbol):
    """Resolve the Kite instrument_token needed for historical_data() calls on an index."""
    symbol = symbol.upper()
    try:
        key = INDEX_SYMBOLS.get(symbol, "").split(":")[-1]
        if symbol == "SENSEX":
            pool = get_bse_instruments()
        else:
            _, pool = get_instruments()
        for o in pool:
            if o.get("tradingsymbol") == key or o.get("name") == key:
                return int(o["instrument_token"])
    except Exception:
        pass
    return AI_INDEX_TOKEN_FALLBACK.get(symbol)

def ai_db():
    c=sqlite3.connect(AI_DB_FILE); c.row_factory=sqlite3.Row; return c

def ai_init_db():
    c=ai_db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS ai_settings(key TEXT PRIMARY KEY,value TEXT);
    CREATE TABLE IF NOT EXISTS ai_versions(id INTEGER PRIMARY KEY AUTOINCREMENT,version TEXT,parent_version TEXT,created_at TEXT,status TEXT,reason TEXT,params TEXT,metrics TEXT);
    CREATE TABLE IF NOT EXISTS ai_market(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,symbol TEXT,spot REAL,iv REAL,pcr REAL,volume REAL,trend REAL,momentum REAL,volatility REAL,snapshot TEXT);
    CREATE TABLE IF NOT EXISTS ai_decisions(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,version TEXT,symbol TEXT,score REAL,decision TEXT,reason TEXT,snapshot TEXT);
    CREATE TABLE IF NOT EXISTS ai_trades(id INTEGER PRIMARY KEY AUTOINCREMENT,version TEXT,ts TEXT,symbol TEXT,option_symbol TEXT,token INTEGER,side TEXT,entry REAL,stop REAL,target REAL,qty INTEGER,score REAL,setup_json TEXT,status TEXT,exit REAL,exit_ts TEXT,pnl REAL DEFAULT 0,r_multiple REAL DEFAULT 0,mfe REAL DEFAULT 0,mae REAL DEFAULT 0,exit_reason TEXT);
    CREATE TABLE IF NOT EXISTS ai_changes(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,from_version TEXT,to_version TEXT,change_json TEXT,hypothesis TEXT,before_metrics TEXT,after_metrics TEXT,impact TEXT,status TEXT);
    CREATE TABLE IF NOT EXISTS ai_learning(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,version TEXT,observation TEXT,hypothesis TEXT,action TEXT,evidence TEXT);
    CREATE TABLE IF NOT EXISTS ai_events(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,level TEXT,event TEXT,detail TEXT);
    CREATE TABLE IF NOT EXISTS ai_historical_candles(id INTEGER PRIMARY KEY AUTOINCREMENT,symbol TEXT,ts TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,UNIQUE(symbol,ts));
    CREATE TABLE IF NOT EXISTS ai_option_captures(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,symbol TEXT,tradingsymbol TEXT,strike REAL,instrument_type TEXT,ltp REAL,bid REAL,ask REAL,volume REAL,oi REAL);
    CREATE TABLE IF NOT EXISTS ai_models(id INTEGER PRIMARY KEY AUTOINCREMENT,model_type TEXT,trained_at TEXT,samples INTEGER,validation_samples INTEGER,accuracy REAL,auc REAL,architecture TEXT,path TEXT);
    """)
    for k,v in AI_DEFAULTS.items(): c.execute("INSERT OR IGNORE INTO ai_settings VALUES(?,?)",(k,str(v)))
    if not c.execute("SELECT 1 FROM ai_versions LIMIT 1").fetchone():
        c.execute("INSERT INTO ai_versions(version,parent_version,created_at,status,reason,params,metrics) VALUES(?,?,?,?,?,?,?)",
                  ("1.0",None,datetime.now().isoformat(),"CHAMPION","Initial autonomous paper strategy",json.dumps(AI_DEFAULTS),json.dumps({})))
    c.commit();c.close()

def ai_log(level,event,detail=""):
    try:
        c=ai_db();c.execute("INSERT INTO ai_events VALUES(NULL,?,?,?,?)",(datetime.now().isoformat(),level,event,detail));c.commit();c.close()
    except Exception: pass

def ai_settings():
    c=ai_db(); rows=c.execute("SELECT key,value FROM ai_settings").fetchall(); c.close(); p=dict(AI_DEFAULTS)
    for r in rows:
        try:p[r["key"]]=float(r["value"])
        except: pass
    return p

def ai_champion():
    c=ai_db();r=c.execute("SELECT * FROM ai_versions WHERE status='CHAMPION' ORDER BY id DESC LIMIT 1").fetchone();c.close();return dict(r) if r else None

def ai_params():
    r=ai_champion(); return json.loads(r["params"]) if r else ai_settings()

def ai_market_open():
    try:
        n=now_ist(); return n.weekday()<5 and datetime.strptime("09:15","%H:%M").time() <= n.time() <= datetime.strptime("15:30","%H:%M").time()
    except: return False

def ai_store_market(symbol, snapshot):
    c=ai_db(); c.execute("INSERT INTO ai_market(ts,symbol,spot,iv,pcr,volume,trend,momentum,volatility,snapshot) VALUES(?,?,?,?,?,?,?,?,?,?)",
      (datetime.now().isoformat(),symbol,snapshot.get("spot"),snapshot.get("iv"),snapshot.get("pcr"),snapshot.get("volume"),snapshot.get("trend"),snapshot.get("momentum"),snapshot.get("volatility"),json.dumps(snapshot)))
    c.commit();c.close()

def ai_market_history(symbol, n=120):
    c=ai_db();r=c.execute("SELECT * FROM ai_market WHERE symbol=? ORDER BY id DESC LIMIT ?",(symbol,n)).fetchall();c.close();return [dict(x) for x in reversed(r)]

def ai_collect_historical(symbol, days_back=AI_HIST_BACKFILL_DAYS):
    """Backfills/refreshes real 5-minute historical candles for `symbol` from Kite's
    historical_data API, chunked to respect the per-call day-range limit. Safe to call
    repeatedly — INSERT OR IGNORE on the (symbol, ts) UNIQUE constraint dedupes overlap.
    Paced with a short sleep between chunk calls (Kite's historical endpoint has a tighter
    rate limit than the general API) and logs per-chunk progress, since a multi-year backfill
    is ~20 chunk calls per symbol and would otherwise report nothing until fully done."""
    token = ai_index_instrument_token(symbol)
    if not token:
        ai_log('ERROR', 'HIST_COLLECT', f'{symbol}: could not resolve instrument token'); return 0
    to_date = now_ist().replace(tzinfo=None)
    from_date = to_date - timedelta(days=days_back)
    total_chunks = max(1, math.ceil(days_back / AI_HIST_CHUNK_DAYS))
    inserted = 0
    cursor = from_date
    chunk_i = 0
    c = ai_db()
    while cursor < to_date:
        chunk_i += 1
        chunk_end = min(cursor + timedelta(days=AI_HIST_CHUNK_DAYS), to_date)
        try:
            candles = kite.historical_data(token, cursor, chunk_end, AI_HIST_INTERVAL) or []
        except Exception as e:
            ai_log('ERROR', 'HIST_COLLECT', f'{symbol} chunk {chunk_i}/{total_chunks} {cursor.date()}..{chunk_end.date()}: {e}')
            candles = []
        chunk_inserted = 0
        for row in candles:
            d = row.get('date')
            ts = d.isoformat() if hasattr(d, 'isoformat') else str(d)
            cur = c.execute(
                "INSERT OR IGNORE INTO ai_historical_candles(symbol,ts,open,high,low,close,volume) VALUES(?,?,?,?,?,?,?)",
                (symbol, ts, row['open'], row['high'], row['low'], row['close'], row.get('volume', 0) or 0))
            chunk_inserted += cur.rowcount
        inserted += chunk_inserted
        if total_chunks > 1:
            c.commit()  # commit progressively on long backfills so partial progress survives a crash/restart
            ai_log('DATAFABRIC', 'HIST_COLLECT', f'{symbol}: chunk {chunk_i}/{total_chunks} ({cursor.date()}..{chunk_end.date()}) +{chunk_inserted} candles')
        cursor = chunk_end
        if cursor < to_date:
            time.sleep(AI_HIST_CHUNK_PAUSE_SECONDS)
    c.commit(); c.close()
    if inserted and total_chunks == 1:
        ai_log('DATAFABRIC', 'HIST_COLLECT', f'{symbol}: +{inserted} candles synchronized')
    return inserted

def ai_historical_symbol_stats(symbol):
    c = ai_db()
    r = c.execute("SELECT COUNT(*) n, MIN(ts) first_ts, MAX(ts) last_ts FROM ai_historical_candles WHERE symbol=?", (symbol,)).fetchone()
    c.close()
    return {"symbol": symbol, "n": r["n"] or 0, "first_ts": r["first_ts"], "last_ts": r["last_ts"]}

# ---- ML: engineered-feature logistic regression, chronological 80/20 split ----
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

def ai_build_ml_dataset(symbol, horizon=3, min_hist=30):
    """horizon=3 bars (~15 min ahead). Features are engineered from real historical closes only."""
    c = ai_db()
    rows = c.execute("SELECT close FROM ai_historical_candles WHERE symbol=? ORDER BY ts", (symbol,)).fetchall()
    c.close()
    closes = np.array([r["close"] for r in rows], dtype=float)
    n = len(closes)
    if n < min_hist + horizon + 50:
        return None
    feats, labels = [], []
    for i in range(min_hist, n - horizon):
        w = closes[i - min_hist:i + 1]
        ret1 = w[-1] / w[-2] - 1
        ret5 = w[-1] / w[-6] - 1
        ret20 = w[-1] / w[-21] - 1
        ma10 = w[-10:].mean(); ma30 = w[-30:].mean()
        diffs = np.diff(w[-11:]) / w[-11:-1]
        vol = float(np.std(diffs)) if len(diffs) else 0.0
        feats.append([ret1, ret5, ret20, ma10 / ma30 - 1, vol])
        labels.append(1.0 if closes[i + horizon] > closes[i] else 0.0)
    return np.array(feats), np.array(labels)

def ai_train_ml(symbol):
    ds = ai_build_ml_dataset(symbol)
    if ds is None: return None
    X, y = ds
    if len(X) < AI_ML_MIN_SAMPLES: return None
    split = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2: return None
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=500).fit(scaler.transform(Xtr), ytr)
    proba = clf.predict_proba(scaler.transform(Xte))[:, 1]
    acc = float(accuracy_score(yte, (proba > 0.5).astype(int)) * 100)
    try: auc = float(roc_auc_score(yte, proba))
    except Exception: auc = 0.5
    path = os.path.join(AI_MODEL_DIR, f"ml_{symbol}.pkl")
    with open(path, 'wb') as f: pickle.dump({"scaler": scaler, "clf": clf}, f)
    c = ai_db()
    c.execute("INSERT INTO ai_models(model_type,trained_at,samples,validation_samples,accuracy,auc,architecture,path) VALUES(?,?,?,?,?,?,?,?)",
              (f"ml_{symbol}", datetime.now().isoformat(), len(Xtr), len(Xte), acc, auc,
               "LogisticRegression · 5 engineered features · chronological 80/20", path))
    c.commit(); c.close()
    ai_log('DATAFABRIC', 'ML_TRAIN', f'{symbol}: samples={len(X)} acc={acc:.1f}% auc={auc:.3f}')
    return {"samples": len(X), "accuracy": acc, "auc": auc}

def ai_ml_predict(symbol):
    path = os.path.join(AI_MODEL_DIR, f"ml_{symbol}.pkl")
    if not os.path.exists(path): return None
    ds = ai_build_ml_dataset(symbol)
    if ds is None or len(ds[0]) == 0: return None
    X, _ = ds
    try:
        with open(path, 'rb') as f: bundle = pickle.load(f)
        return float(bundle["clf"].predict_proba(bundle["scaler"].transform(X[-1:]))[0, 1])
    except Exception:
        return None

def ai_ml_latest():
    c = ai_db()
    rows = c.execute("""SELECT model_type,samples,validation_samples,accuracy,auc,trained_at FROM ai_models
                         WHERE model_type LIKE 'ml_%' AND id IN
                         (SELECT MAX(id) FROM ai_models WHERE model_type LIKE 'ml_%' GROUP BY model_type)""").fetchall()
    c.close()
    if not rows: return {"samples": 0, "accuracy": 0, "auc": 0}
    return {"samples": sum(r["samples"] for r in rows),
            "accuracy": round(mean([r["accuracy"] for r in rows]), 2),
            "auc": round(mean([r["auc"] for r in rows]), 3)}

# ---- DL: a genuine from-scratch GRU (forward + backprop-through-time), no framework dep ----
class NumpyGRU:
    """Single-layer GRU -> sigmoid readout. Trained via full backprop-through-time with Adam.
    Verified on synthetic sequential tasks (train loss -> ~0, held-out accuracy ~90%+)."""
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.RandomState(seed)
        H, F = hidden_dim, input_dim
        s = lambda *shape: (rng.randn(*shape) * (1.0 / np.sqrt(shape[-1]))).astype(np.float64)
        self.H, self.F = H, F
        self.Wz, self.Wr, self.Wh = s(H, F), s(H, F), s(H, F)
        self.Uz, self.Ur, self.Uh = s(H, H), s(H, H), s(H, H)
        self.bz, self.br, self.bh = np.zeros(H), np.zeros(H), np.zeros(H)
        self.Wo = s(1, H); self.bo = np.zeros(1)
        self.params = ['Wz', 'Wr', 'Wh', 'Uz', 'Ur', 'Uh', 'bz', 'br', 'bh', 'Wo', 'bo']
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    @staticmethod
    def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def forward(self, X):
        T = X.shape[0]; H = self.H
        h = np.zeros(H); cache = []
        for t in range(T):
            x = X[t]
            z = self.sigmoid(self.Wz @ x + self.Uz @ h + self.bz)
            r = self.sigmoid(self.Wr @ x + self.Ur @ h + self.br)
            hhat = np.tanh(self.Wh @ x + self.Uh @ (r * h) + self.bh)
            h_new = (1 - z) * h + z * hhat
            cache.append((x, h.copy(), z, r, hhat, h_new.copy()))
            h = h_new
        o = self.sigmoid(self.Wo @ h + self.bo)[0]
        return o, h, cache

    def backward(self, cache, o, y):
        H = self.H
        grads = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        dL_do = (o - y)
        h_T = cache[-1][5]
        grads['Wo'] = dL_do * h_T.reshape(1, H); grads['bo'] = np.array([dL_do])
        dh_next = dL_do * self.Wo.flatten()
        for t in reversed(range(len(cache))):
            x, h_prev, z, r, hhat, h_new = cache[t]
            dh = dh_next
            dz = dh * (hhat - h_prev); dhhat = dh * z; dh_prev_from_h = dh * (1 - z)
            dhhat_raw = dhhat * (1 - hhat ** 2)
            grads['Wh'] += np.outer(dhhat_raw, x); grads['Uh'] += np.outer(dhhat_raw, r * h_prev); grads['bh'] += dhhat_raw
            dr_h_prev = self.Uh.T @ dhhat_raw
            dr = dr_h_prev * h_prev; dh_prev_from_r = dr_h_prev * r
            dz_raw = dz * z * (1 - z)
            grads['Wz'] += np.outer(dz_raw, x); grads['Uz'] += np.outer(dz_raw, h_prev); grads['bz'] += dz_raw
            dh_prev_from_z = self.Uz.T @ dz_raw
            dr_raw = dr * r * (1 - r)
            grads['Wr'] += np.outer(dr_raw, x); grads['Ur'] += np.outer(dr_raw, h_prev); grads['br'] += dr_raw
            dh_prev_from_r2 = self.Ur.T @ dr_raw
            dh_next = dh_prev_from_h + dh_prev_from_r + dh_prev_from_z + dh_prev_from_r2
        return grads

    def adam_step(self, grads, lr=0.008, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p in self.params:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * (g * g)
            mhat = self.m[p] / (1 - b1 ** self.t); vhat = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))

def _ai_clip_grads(grads, max_norm=5.0):
    total = float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))
    if total > max_norm:
        scale = max_norm / (total + 1e-8)
        for k in grads: grads[k] = grads[k] * scale
    return grads

def ai_build_dl_dataset(symbol, seq_len=AI_SEQ_LEN, horizon=5):
    """horizon=5 bars (~25 min ahead) — deliberately further out than the ML horizon so the
    two learners are asking genuinely different questions, matching the ensemble design."""
    c = ai_db()
    rows = c.execute("SELECT close,volume FROM ai_historical_candles WHERE symbol=? ORDER BY ts", (symbol,)).fetchall()
    c.close()
    closes = np.array([r["close"] for r in rows], dtype=float)
    vols = np.array([float(r["volume"] or 0) for r in rows], dtype=float)
    n = len(closes)
    if n < seq_len + horizon + 50:
        return None
    rets = np.diff(closes) / closes[:-1]
    rets = np.concatenate([[0.0], rets])
    vstd = vols.std() + 1e-6
    vnorm = (vols - vols.mean()) / vstd
    seqs, labels = [], []
    for i in range(seq_len, n - horizon):
        feat = np.stack([rets[i - seq_len:i], vnorm[i - seq_len:i]], axis=1)
        seqs.append(feat)
        labels.append(1.0 if closes[i + horizon - 1] > closes[i - 1] else 0.0)
    return np.array(seqs), np.array(labels)

def ai_train_dl(symbol):
    ds = ai_build_dl_dataset(symbol)
    if ds is None: return None
    X, y = ds
    if len(X) < AI_DL_MIN_SAMPLES: return None
    split = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    if len(set(ytr)) < 2 or len(set(yte)) < 2: return None
    gru = NumpyGRU(input_dim=X.shape[2], hidden_dim=32, seed=42)
    n_train = len(Xtr); batch = 32
    # Fixed compute budget (~350k sample-passes ≈ 5-8 min/symbol at this hidden size, measured)
    # instead of a fixed epoch count, so a 5-year backfill (~90k+ sequences) doesn't multiply
    # training time 20x versus a 60-day one — fewer epochs are needed anyway once there's more data.
    epochs = max(3, min(8, round(350_000 / max(n_train, 1))))
    idx = np.arange(n_train)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n_train, batch):
            b = idx[start:start + batch]
            acc = {p: np.zeros_like(getattr(gru, p)) for p in gru.params}
            for i in b:
                o, _, cache = gru.forward(Xtr[i])
                g = gru.backward(cache, o, ytr[i])
                for k in acc: acc[k] += g[k] / len(b)
            gru.adam_step(_ai_clip_grads(acc))
    preds = np.array([gru.forward(Xte[i])[0] for i in range(len(Xte))])
    acc_pct = float(((preds > 0.5).astype(int) == yte).mean() * 100)
    try: auc = float(roc_auc_score(yte, preds))
    except Exception: auc = 0.5
    path = os.path.join(AI_MODEL_DIR, f"dl_{symbol}.npz")
    weights = {p: getattr(gru, p) for p in gru.params}
    np.savez(path, **weights, input_dim=X.shape[2], hidden_dim=32)
    c = ai_db()
    c.execute("INSERT INTO ai_models(model_type,trained_at,samples,validation_samples,accuracy,auc,architecture,path) VALUES(?,?,?,?,?,?,?,?)",
              (f"dl_{symbol}", datetime.now().isoformat(), len(Xtr), len(Xte), acc_pct, auc,
               f"GRU-32 · {AI_SEQ_LEN} sequential market states · BPTT+Adam", path))
    c.commit(); c.close()
    ai_log('DATAFABRIC', 'DL_TRAIN', f'{symbol}: samples={len(X)} acc={acc_pct:.1f}% auc={auc:.3f}')
    return {"samples": len(X), "accuracy": acc_pct, "auc": auc}

def ai_dl_predict(symbol):
    path = os.path.join(AI_MODEL_DIR, f"dl_{symbol}.npz")
    if not os.path.exists(path): return None
    ds = ai_build_dl_dataset(symbol)
    if ds is None or len(ds[0]) == 0: return None
    X, _ = ds
    try:
        data = np.load(path)
        gru = NumpyGRU(int(data['input_dim']), int(data['hidden_dim']), seed=0)
        for p in gru.params: setattr(gru, p, data[p])
        o, _, _ = gru.forward(X[-1])
        return float(o)
    except Exception:
        return None

def ai_dl_latest():
    c = ai_db()
    rows = c.execute("""SELECT model_type,samples,validation_samples,accuracy,auc,trained_at FROM ai_models
                         WHERE model_type LIKE 'dl_%' AND id IN
                         (SELECT MAX(id) FROM ai_models WHERE model_type LIKE 'dl_%' GROUP BY model_type)""").fetchall()
    c.close()
    arch = f"GRU-32 · {AI_SEQ_LEN} sequential market states"
    if not rows:
        return {"status": "NOT TRAINED", "samples": 0, "accuracy": 0, "auc": 0, "architecture": arch, "updated_at": None}
    return {"status": "TRAINED", "samples": sum(r["samples"] for r in rows),
            "accuracy": round(mean([r["accuracy"] for r in rows]), 2),
            "auc": round(mean([r["auc"] for r in rows]), 3),
            "architecture": arch, "updated_at": max(r["trained_at"] for r in rows)}

def ai_should_retrain(model_type, symbol, min_hours=None):
    if min_hours is None:
        min_hours = AI_DL_RETRAIN_MIN_HOURS if model_type == 'dl' else AI_ML_RETRAIN_MIN_HOURS
    c = ai_db()
    r = c.execute("SELECT trained_at FROM ai_models WHERE model_type=? ORDER BY id DESC LIMIT 1", (f"{model_type}_{symbol}",)).fetchone()
    c.close()
    if not r: return True
    try:
        last = datetime.fromisoformat(r["trained_at"])
        return (datetime.now() - last).total_seconds() >= min_hours * 3600
    except Exception:
        return True

# ---- Research ensemble: combines ML + DL into a single probability/uncertainty/regime read ----
def ai_infer_regime():
    c = ai_db()
    rows = c.execute("SELECT volatility,trend FROM ai_market ORDER BY id DESC LIMIT 30").fetchall()
    c.close()
    if not rows: return "UNKNOWN"
    v = mean([r["volatility"] or 0 for r in rows]); t = mean([r["trend"] or 0 for r in rows])
    if v > 1.2: return "VOLATILE"
    if t > 17: return "TRENDING"
    return "CHOPPY"

def ai_research_snapshot():
    per_symbol = {}; probs = []
    for sym in AI_SYMBOLS:
        mlp = ai_ml_predict(sym); dlp = ai_dl_predict(sym)
        vals = [v for v in (mlp, dlp) if v is not None]
        if vals:
            per_symbol[sym] = {"ml": mlp, "dl": dlp, "ensemble": mean(vals)}
            probs.append(mean(vals))
    ml_latest = ai_ml_latest(); dl_latest = ai_dl_latest()
    if not probs:
        return {"status": "WARMING", "probability": None, "uncertainty": None, "regime": None, "auc": None, "per_symbol": {}}
    aucs = [a for a in (ml_latest.get("auc"), dl_latest.get("auc")) if a]
    return {"status": "ACTIVE" if (ml_latest["samples"] and dl_latest["samples"]) else "WARMING",
            "probability": mean(probs),
            "uncertainty": (max(probs) - min(probs)) if len(probs) > 1 else 0.15,
            "regime": ai_infer_regime(),
            "auc": mean(aucs) if aucs else None,
            "per_symbol": per_symbol}

def ai_learning_state():
    hist_total = sum(ai_historical_symbol_stats(s)["n"] for s in AI_SYMBOLS)
    hist_target = AI_HIST_BACKFILL_DAYS * AI_BARS_PER_DAY_EST * len(AI_SYMBOLS)
    hist_progress = min(1.0, hist_total / hist_target) if hist_target else 0
    ml = ai_ml_latest(); dl = ai_dl_latest()
    ml_progress = min(1.0, ml["samples"] / (AI_ML_MIN_SAMPLES * len(AI_SYMBOLS))) if ml["samples"] else 0
    dl_progress = min(1.0, dl["samples"] / (AI_DL_MIN_SAMPLES * len(AI_SYMBOLS))) if dl["samples"] else 0
    progress = 100 * (0.5 * hist_progress + 0.25 * ml_progress + 0.25 * dl_progress)
    if progress < 25: stage = "AUTONOMOUS BOOTSTRAP"
    elif progress < 60: stage = "HISTORICAL DATA COLLECTION"
    elif progress < 85: stage = "ML CALIBRATION"
    elif progress < 100: stage = "DEEP LEARNING CALIBRATION"
    else: stage = "AUTONOMOUS — FULL EVIDENCE"
    base_gate = float(ai_params()["min_score"])
    # Effective gate relaxes slightly while evidence is thin (more exploration to gather
    # outcomes faster) and tightens back to the champion's real gate as progress completes.
    effective = base_gate if progress >= 100 else round(base_gate - (1 - progress / 100) * 17, 1)
    return {"stage": stage, "progress": round(progress, 1), "effective_score": max(40.0, effective)}

def ai_historical_status():
    stats = [ai_historical_symbol_stats(s) for s in AI_SYMBOLS]
    total = sum(s["n"] for s in stats)
    target = AI_HIST_BACKFILL_DAYS * AI_BARS_PER_DAY_EST * len(AI_SYMBOLS)
    ml = ai_ml_latest(); dl = ai_dl_latest()
    if total == 0: status = "WAITING"
    elif total < target * 0.9: status = "COLLECTING — building chronological history"
    elif not dl.get("samples"): status = "TRAINING — building ML/DL models"
    else: status = "PRETRAINED — online learning active"
    model = {"status": "TRAINED" if ml.get("samples") else ("TRAINING" if total else "—"),
             "samples": ml.get("samples", 0),
             "validation_samples": int(ml.get("samples", 0) * 0.2) if ml.get("samples") else 0,
             "accuracy": ml.get("accuracy"), "auc": ml.get("auc")}
    return {"status": status, "interval": AI_HIST_INTERVAL, "symbols": stats, "model": model}

# ---- Live option-chain memory (real captures, throttled to once per 5 min per symbol) ----
_AI_OPT_LAST_CAPTURE = {}

def ai_capture_option_chain(symbol, chain):
    now = datetime.now()
    last = _AI_OPT_LAST_CAPTURE.get(symbol)
    if last and (now - last).total_seconds() < 300:
        return
    _AI_OPT_LAST_CAPTURE[symbol] = now
    c = ai_db(); ts = now.isoformat()
    for o in chain:
        c.execute("INSERT INTO ai_option_captures(ts,symbol,tradingsymbol,strike,instrument_type,ltp,bid,ask,volume,oi) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (ts, symbol, o.get("tradingsymbol"), o.get("strike"), o.get("instrument_type"),
                   o.get("ltp"), o.get("bid"), o.get("ask"), o.get("volume"), o.get("oi")))
    c.commit(); c.close()
    ai_log('DATAFABRIC', 'OPTION_CAPTURE', f'{symbol}: {len(chain)} contracts captured')

def ai_option_capture_stats():
    c = ai_db()
    total = c.execute("SELECT COUNT(*) n FROM ai_option_captures").fetchone()["n"]
    last = c.execute("SELECT MAX(ts) t FROM ai_option_captures").fetchone()["t"]
    per_symbol = []
    for sym in AI_SYMBOLS:
        r = c.execute("""SELECT COUNT(*) n, COUNT(DISTINCT tradingsymbol) contracts, MIN(ts) first_ts, MAX(ts) last_ts
                          FROM ai_option_captures WHERE symbol=?""", (sym,)).fetchone()
        per_symbol.append({"symbol": sym, "n": r["n"] or 0, "contracts": r["contracts"] or 0,
                            "first_ts": r["first_ts"], "last_ts": r["last_ts"]})
    c.close()
    return {"status": "RECORDING" if total else "WAITING", "rows": total, "last_capture_at": last, "symbols": per_symbol}

def ai_historical_loop():
    """Background thread: initial backfill, then periodic incremental refresh + retraining.
    Runs independently of the live paper-trading loop/market hours since it deals with
    historical (not live) data. Retraining is time-gated per symbol/model to avoid hammering
    the CPU with a multi-minute GRU training pass every cycle. Backfill is re-checked on every
    iteration (not just once at startup) so it actually runs once the user logs into Kite,
    even if the server process started before that login happened."""
    while True:
        sleep_s = AI_HIST_POLL_SECONDS
        try:
            if require_session():
                for sym in AI_SYMBOLS:
                    try:
                        stats = ai_historical_symbol_stats(sym)
                        if stats["n"] < 100:
                            ai_log('DATAFABRIC', 'HIST_BACKFILL', f'{sym}: starting {AI_HIST_BACKFILL_DAYS}-day backfill')
                            ai_collect_historical(sym, days_back=AI_HIST_BACKFILL_DAYS)
                        else:
                            ai_collect_historical(sym, days_back=3)
                    except Exception as e:
                        ai_log('ERROR', 'HIST_COLLECT', f'{sym}: {e}')
                for sym in AI_SYMBOLS:
                    try:
                        if ai_should_retrain('ml', sym): ai_train_ml(sym)
                    except Exception as e: ai_log('ERROR', 'ML_TRAIN', f'{sym}: {e}')
                    try:
                        if ai_should_retrain('dl', sym): ai_train_dl(sym)
                    except Exception as e: ai_log('ERROR', 'DL_TRAIN', f'{sym}: {e}')
            else:
                sleep_s = 30  # short retry cadence while waiting for the user to log in to Kite
        except Exception as e:
            ai_log('ERROR', 'HIST_LOOP', str(e)); sleep_s = 60
        time.sleep(sleep_s)

def ai_snapshot(symbol):
    data,err=get_chain_for_symbol(symbol)
    if err:return None,err
    spot=float(data["spot"]); chain=data["chain"]
    try: ai_capture_option_chain(symbol, chain)
    except Exception as e: ai_log('ERROR','OPTION_CAPTURE',f'{symbol}: {e}')
    calls=[x for x in chain if x.get("instrument_type")=="CE"]; puts=[x for x in chain if x.get("instrument_type")=="PE"]
    atm=min(chain,key=lambda x:abs(float(x["strike"])-spot)) if chain else None
    iv=float(atm.get("iv") or 0) if atm else 0
    pcr=compute_pcr_and_max_pain(chain).get("pcr") or 1.0
    vol=float(sum((x.get("volume") or 0) for x in chain)/max(len(chain),1))
    hist=ai_market_history(symbol,120); prices=[float(x["spot"]) for x in hist if x.get("spot") is not None]+[spot]
    ret5=((spot/prices[-6])-1)*100 if len(prices)>=6 else 0
    ret20=((spot/prices[-21])-1)*100 if len(prices)>=21 else ret5
    ma10=mean(prices[-10:]) if len(prices)>=10 else spot
    ma30=mean(prices[-30:]) if len(prices)>=30 else ma10
    trend=max(0,min(25,12.5+(ma10/ma30-1)*500))
    momentum=max(0,min(25,12.5+ret5*3+ret20*1.5))
    if len(prices)>=10:
        m=mean(prices[-10:]); sd=(mean([(z-m)**2 for z in prices[-10:]])**0.5); volatility=(sd/m*100) if m else 0
    else: volatility=0
    vol_score=max(0,min(15,15-abs(volatility-0.5)*8))
    pcr_score=max(0,min(15,7.5+(pcr-1)*15))
    iv_score=max(0,min(20,10+(iv-15)*0.6))
    score=trend+momentum+vol_score+pcr_score+iv_score
    side="LONG" if trend+momentum+pcr_score>=32 else "SHORT"
    option_type="CE" if side=="LONG" else "PE"
    candidates=[x for x in chain if x.get("instrument_type")==option_type and x.get("delta") is not None]
    if not candidates: return None,{"error":"No option candidates"}
    target=min(candidates,key=lambda x:abs(abs(float(x.get("delta",0)))-0.50))
    entry=float(target.get("mid") or target.get("ltp") or 0)
    if entry<=0: return None,{"error":"No tradable quote (mid/ltp missing or zero) for candidate option"}
    lot=int(data.get("lot_size") or target.get("lot_size") or 1)
    stop=entry*(1-float(ai_params()["stop_pct"])/100); target_price=entry*(1+float(ai_params()["target_pct"])/100)
    ml_p=ai_ml_predict(symbol); dl_p=ai_dl_predict(symbol)
    ens_vals=[v for v in (ml_p,dl_p) if v is not None]
    return {"symbol":symbol,"spot":spot,"score":round(score,2),"side":side,"option_type":option_type,
            "option_symbol":target.get("tradingsymbol"),"token":int(target.get("instrument_token")),"entry":entry,
            "stop":stop,"target":target_price,"lot_size":lot,"iv":iv,"pcr":pcr,"volume":vol,
            "trend":trend,"momentum":momentum,"volatility":volatility,"components":{"trend":round(trend,2),"momentum":round(momentum,2),"volatility":round(vol_score,2),"pcr":round(pcr_score,2),"iv":round(iv_score,2)},
            "rr":round((target_price-entry)/max(entry-stop,0.0001),2),
            "ml_probability":ml_p,"dl_probability":dl_p,
            "research_probability":mean(ens_vals) if ens_vals else None,
            "research_uncertainty":(max(ens_vals)-min(ens_vals)) if len(ens_vals)>1 else None,
            "market_regime":ai_infer_regime()},None

def ai_metrics(version=None):
    c=ai_db();q="SELECT * FROM ai_trades WHERE status='CLOSED'";a=[]
    if version:q+=" AND version=?";a.append(version)
    rows=c.execute(q,a).fetchall();c.close(); pnl=[float(r["pnl"]) for r in rows];rs=[float(r["r_multiple"]) for r in rows]
    wins=[x for x in pnl if x>0];loss=[x for x in pnl if x<0]
    return {"trades":len(rows),"win_rate":round(len(wins)/len(rows)*100,2) if rows else 0,"avg_r":round(sum(rs)/len(rs),3) if rs else 0,"profit_factor":round(sum(wins)/abs(sum(loss)),2) if loss else (99 if wins else 0),"pnl":round(sum(pnl),2)}

def ai_open_trades():
    c=ai_db();r=c.execute("SELECT * FROM ai_trades WHERE status='OPEN'").fetchall();c.close();return [dict(x) for x in r]

def ai_open_trade(x,ver):
    p=ai_params()
    if len(ai_open_trades())>=int(p["max_positions"]):return False
    c=ai_db();c.execute("INSERT INTO ai_trades(version,ts,symbol,option_symbol,token,side,entry,stop,target,qty,score,setup_json,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
      (ver,datetime.now().isoformat(),x["symbol"],x["option_symbol"],x["token"],x["side"],x["entry"],x["stop"],x["target"],x["lot_size"],x["score"],json.dumps(x),"OPEN"));c.commit();c.close();ai_log("TRADE","AUTO_PAPER_OPEN",f'{x["symbol"]} {x["option_symbol"]} score={x["score"]}');return True

def ai_close_trades():
    if not require_session():return
    for t in ai_open_trades():
        try:
            q=kite_quote_bulk([f'NFO:{t["option_symbol"]}']).get(f'NFO:{t["option_symbol"]}')
            price=extract_price(q) if q else None
            if price is None:continue
            # NOTE: "side" is the directional bias of the underlying (LONG bias -> buy CE,
            # SHORT bias -> buy PE) — the engine never sells options. ai_snapshot() always
            # computes stop below entry and target above entry regardless of side, so the
            # exit test and PnL sign must be identical for both sides. The previous
            # side-branching logic inverted stop/target checks for SHORT-bias trades,
            # closing them out almost immediately as false "STOP" hits with an incorrectly
            # POSITIVE pnl (entry-exit instead of exit-entry). Fixed here.
            hit=None
            if price<=t["stop"]:hit=("STOP",t["stop"])
            elif price>=t["target"]:hit=("TARGET",t["target"])
            if hit:
                exitp=hit[1];pnl=(exitp-t["entry"])*t["qty"]
                risk=abs(t["entry"]-t["stop"])*t["qty"];rm=pnl/risk if risk else 0
                c=ai_db();c.execute("UPDATE ai_trades SET status='CLOSED',exit=?,exit_ts=?,pnl=?,r_multiple=?,exit_reason=? WHERE id=?",(exitp,datetime.now().isoformat(),pnl,rm,hit[0],t["id"]));c.commit();c.close();ai_log("TRADE","AUTO_PAPER_CLOSE",f'id={t["id"]} {hit[0]} pnl={pnl:.2f} R={rm:.2f}')
        except Exception as e:ai_log("ERROR","AI_MONITOR",str(e))

def ai_record_decision(ver,x,decision,reason):
    c=ai_db();c.execute("INSERT INTO ai_decisions VALUES(NULL,?,?,?,?,?,?,?)",(datetime.now().isoformat(),ver,x["symbol"],x["score"],decision,reason,json.dumps(x)));c.commit();c.close()

def ai_learn():
    ch=ai_champion();
    if not ch:return
    p=json.loads(ch["params"]);m=ai_metrics(ch["version"])
    if m["trades"]<int(p["learning_min_trades"]):return
    c=ai_db();rows=c.execute("SELECT score,r_multiple FROM ai_trades WHERE version=? AND status='CLOSED' ORDER BY id DESC LIMIT 100",(ch["version"],)).fetchall();c.close()
    high=[r["r_multiple"] for r in rows if r["score"]>=80];low=[r["r_multiple"] for r in rows if r["score"]<80]
    cand=dict(p);change={};hyp=""
    if len(high)>=15 and len(low)>=10 and mean(high)>mean(low)+0.15:
        old=p["min_score"];new=min(90,old+5)
        if new!=old:cand["min_score"]=new;change={"min_score":[old,new]};hyp="High-score setups have materially better R expectancy."
    elif m["win_rate"]<45 and m["avg_r"]<0:
        old=p["min_rr"];new=min(2.5,old+0.2)
        if new!=old:cand["min_rr"]=new;change={"min_rr":[old,new]};hyp="Negative expectancy requires stricter reward-to-risk selection."
    elif m["win_rate"]>60 and m["avg_r"]>0.25:
        old=p["min_score"];new=max(65,old-2)
        if new!=old:cand["min_score"]=new;change={"min_score":[old,new]};hyp="Test whether broader selection retains strong expectancy."
    if not change:return
    newver=f'{float(ch["version"])+0.1:.1f}'
    c=ai_db();c.execute("INSERT INTO ai_versions(version,parent_version,created_at,status,reason,params,metrics) VALUES(?,?,?,?,?,?,?)",(newver,ch["version"],datetime.now().isoformat(),"CHALLENGER",hyp,json.dumps(cand),json.dumps(m)));c.execute("INSERT INTO ai_changes(ts,from_version,to_version,change_json,hypothesis,before_metrics,after_metrics,impact,status) VALUES(?,?,?,?,?,?,?,?,?)",(datetime.now().isoformat(),ch["version"],newver,json.dumps(change),hyp,json.dumps(m),json.dumps({}),"Awaiting challenger evidence","TESTING"));c.execute("INSERT INTO ai_learning VALUES(NULL,?,?,?,?,?,?)",(datetime.now().isoformat(),ch["version"],f'{m["trades"]} trades: win {m["win_rate"]}%, avgR {m["avg_r"]}',hyp,f'Created challenger {newver}',json.dumps(change)));c.commit();c.close();ai_log("LEARNING","CHALLENGER_CREATED",f'{ch["version"]}->{newver}')

def ai_evaluate_challenger():
    c=ai_db();ch=c.execute("SELECT * FROM ai_versions WHERE status='CHALLENGER' ORDER BY id LIMIT 1").fetchone();c.close()
    if not ch:return
    p=json.loads(ch["params"]);m=ai_metrics(ch["version"])
    if m["trades"]<int(p["challenger_min_trades"]):return
    parent=ai_metrics(ch["parent_version"]);better=(m["avg_r"]>parent["avg_r"]+0.05 and m["profit_factor"]>=parent["profit_factor"]) or (m["win_rate"]>=parent["win_rate"]+5 and m["avg_r"]>=parent["avg_r"])
    c=ai_db()
    if better:
        c.execute("UPDATE ai_versions SET status='RETIRED',metrics=? WHERE version=?",(json.dumps(parent),ch["parent_version"]));c.execute("UPDATE ai_versions SET status='CHAMPION',metrics=? WHERE version=?",(json.dumps(m),ch["version"]));c.execute("UPDATE ai_changes SET after_metrics=?,impact=?,status='ACCEPTED' WHERE to_version=?",(json.dumps(m),f'Improved vs parent: {m}',ch["version"]));action='ACCEPTED'
    else:
        c.execute("UPDATE ai_versions SET status='REJECTED',metrics=? WHERE version=?",(json.dumps(m),ch["version"]));c.execute("UPDATE ai_changes SET after_metrics=?,impact=?,status='ROLLED_BACK' WHERE to_version=?",(json.dumps(m),'Challenger failed; parent retained',ch["version"]));action='ROLLED_BACK'
    c.commit();c.close();ai_log('LEARNING',action,f'challenger={ch["version"]} metrics={m}')

def ai_cycle():
    if not ai_market_open() or not require_session():return
    ai_close_trades();p=ai_params();ch=ai_champion();ver=ch["version"] if ch else '1.0'
    for sym in AI_SYMBOLS:
        try:
            x,err=ai_snapshot(sym)
            if err: ai_log('ERROR','AI_SCAN',f'{sym}: {err}');continue
            ai_store_market(sym,x)
            approved=x["score"]>=p["min_score"] and x["rr"]>=p["min_rr"]
            reason=f'score={x["score"]} rr={x["rr"]} threshold={p["min_score"]}'
            ai_record_decision(ver,x,'APPROVE' if approved else 'REJECT',reason)
            if approved and not any(t['symbol']==sym for t in ai_open_trades()):ai_open_trade(x,ver)
        except Exception as e:ai_log('ERROR','AI_SCAN',f'{sym}: {e}')
    ai_learn();ai_evaluate_challenger()

def ai_loop():
    ai_init_db();ai_log('SYSTEM','START','Autonomous AI Evolution engine started; paper mode only')
    while True:
        try:
            if ai_settings().get('enabled',1):ai_cycle()
        except Exception as e:ai_log('ERROR','AI_ENGINE',str(e))
        time.sleep(AI_POLL_SECONDS)

@app.route('/api/ai-evolution/status')
def ai_status():
    ai_init_db();ch=ai_champion();p=ai_params();m=ai_metrics(ch['version'] if ch else None)
    return jsonify({'enabled':bool(ai_settings().get('enabled',1)),'mode':'AUTONOMOUS PAPER','connected':bool(SESSION.get('access_token')),'market_open':ai_market_open(),'poll_seconds':AI_POLL_SECONDS,'champion':ch['version'] if ch else None,'params':p,'open_trades':len(ai_open_trades()),'metrics':m,
                    'historical':ai_historical_status(),'ml':ai_ml_latest(),'deep_learning':ai_dl_latest(),
                    'research':ai_research_snapshot(),'learning':ai_learning_state(),
                    'timestamp':datetime.now().isoformat()})

@app.route('/api/ai-evolution/options')
def ai_options_api():
    return jsonify(ai_option_capture_stats())

@app.route('/api/ai-evolution/trades')
def ai_trades_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_trades ORDER BY id DESC LIMIT 200')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/decisions')
def ai_decisions_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_decisions ORDER BY id DESC LIMIT 200')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/changes')
def ai_changes_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_changes ORDER BY id DESC LIMIT 100')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/learning')
def ai_learning_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_learning ORDER BY id DESC LIMIT 100')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/versions')
def ai_versions_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_versions ORDER BY id DESC')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/events')
def ai_events_api():
    c=ai_db();r=[dict(x) for x in c.execute('SELECT * FROM ai_events ORDER BY id DESC LIMIT 150')];c.close();return jsonify(r)
@app.route('/api/ai-evolution/cycle',methods=['POST'])
def ai_force_cycle():
    ai_cycle();return jsonify({'ok':True})

ai_init_db()
threading.Thread(target=ai_loop,daemon=True).start()
threading.Thread(target=ai_historical_loop,daemon=True).start()

threading.Thread(target=_autotrade_loop, daemon=True).start()
threading.Thread(target=_breakout_monitor_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")



@app.route("/api/breakout/diagnostics")
def breakout_diagnostics():
    return jsonify({
        "historical_confirmations": ["breakout","vwap","ema","rsi","volume","support_resistance"],
        "thresholds": [4,5,6],
        "option_momentum_included": False
    })

if __name__ == "__main__":
    if "PUT_YOUR" in API_KEY or "PUT_YOUR" in API_SECRET:
        print("!! Set KITE_API_KEY and KITE_API_SECRET (env vars, or edit backend.py) before running.")
    print(f"Set your Kite app's Redirect URL to: {REDIRECT_URL}")
    if ALLOW_INSECURE_NEWS:
        print("!! ALLOW_INSECURE_NEWS is on — news headline fetches will skip TLS verification on failure.")
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
