"""
Kite Option-Selling Dashboard — local backend
------------------------------------------------
Run:  python backend.py
Then open: https://algo.wecon.in

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
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, send_from_directory, redirect

try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException
except ImportError:
    raise SystemExit("Missing dependency. Run: pip install kiteconnect flask numpy requests")

import numpy as np
import requests

# ---------------------------------------------------------------------------
# CONFIG — fill these in from https://developers.kite.trade (your app)
# SECURITY: set these as real environment variables (or a .env file loaded before
# this process starts) — do NOT hardcode real keys/secrets directly in this file,
# especially if this file is ever shared, committed to git, or pasted anywhere.
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("KITE_API_KEY", "b4j9bna5hdew1hh4")
API_SECRET = os.environ.get("KITE_API_SECRET", "mbrdjydzd9ckisvrp4tsqbtkkgojpzue")
REDIRECT_URL = os.environ.get("REDIRECT_URL", "https://algo.wecon.in/api/callback")

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
IV_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "iv_history.json")
IVR_LOOKBACK_DAYS = 252          # ~1 trading year of daily ATM-IV snapshots kept per symbol
IVR_MIN_HISTORY_DAYS = 20        # need at least this many stored days before trusting a real IV rank

# Liquidity gate applied to the ATM strike (both legs) before a stock is allowed into the
# "top picks" ranking — a calm, high-IV stock is still a bad pick if you can't get filled near mid.
MIN_ATM_TOTAL_OI = 500           # combined ATM CE+PE open interest, in contracts (lots), not shares
MAX_ATM_SPREAD_PCT = 4.0         # combined ATM CE+PE avg bid-ask spread, as % of mid price

# Composite score weights (must sum to 1.0). Higher composite = better candidate for this
# option-SELLING strategy: rich premium (high IV) + calm underlying + liquid enough to trade.
SCORE_WEIGHTS = {"iv_richness": 0.40, "calmness": 0.35, "liquidity": 0.25}

NEWS_FOR_TOP_N = 10              # only fetch headlines for the final top-N shown, to keep this fast
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
    today = datetime.now().date()
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

    quotes = {}
    chunk_size = 200
    for i in range(0, len(needed_keys), chunk_size):
        chunk = needed_keys[i:i + chunk_size]
        try:
            quotes.update(kite.quote(chunk))
        except Exception:
            continue  # skip this chunk rather than fail the whole screener

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
}

app = Flask(__name__, static_folder="static", static_url_path="")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("kite_dashboard")
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
    today = datetime.now().date()
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
    today = datetime.now().date()
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
        "added_on": position.get("added_on"), "closed_on": datetime.now().date().isoformat(),
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
    """Marketable LIMIT: BUY uses best Ask; SELL uses best Bid."""
    return (ask if ask is not None else ltp) if transaction_type == "BUY" else (bid if bid is not None else ltp)


def refresh_execution_quotes(position):
    """Fetch fresh LTP/Bid/Ask and recommended LIMIT prices for all entry legs."""
    quantity = position.get("quantity", position["lot_size"])
    leg_keys = leg_keys_for(position)
    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    quotes = kite.quote(inst_keys)

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
    include_news = request.args.get("news", "true").lower() == "true"
    today = datetime.now().date()
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

    SCREENER_CACHE["results"] = eligible + excluded
    SCREENER_CACHE["fetched_at"] = datetime.now()

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
    result["entry_event_warning"] = get_entry_warning()
    result["event_before_expiry"] = get_event_before_expiry(result["expiry"])

    entry_orders_for_charges = [{"price": lg["ltp"], "quantity": quantity,
                                  "transaction_type": "SELL" if k.startswith("sell") else "BUY"}
                                 for k, lg in result["legs"].items()]
    entry_charges = estimate_charges(entry_orders_for_charges)
    result["estimated_entry_charges"] = entry_charges
    if result["max_profit"] is not None:
        result["net_profit_after_entry_charges"] = round(result["max_profit"] - entry_charges["total"], 2)
    else:
        result["net_profit_after_entry_charges"] = None
    result["charges_note"] = ("Entry-side charges only (opening the position). If you square off "
                               "before expiry, exit-side charges apply too — see the Trade Section for "
                               "the running round-trip estimate once tracked. Approximate; verify against "
                               "your Kite contract note.")
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
        if k == "sell_put" and delta_put is not None:
            leg_details[k]["current_delta"] = round(delta_put, 3)

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
    today_str = datetime.now().date().isoformat()
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
    return ["sell_call", "buy_call", "sell_put", "buy_put"] if position.get("strategy_type") == "iron_condor" \
        else ["sell_call", "sell_put"]


def place_basket_orders(legs_to_place, product, order_type):
    """Places each leg as a separate real order. Stops immediately on the first failure rather
    than continuing — continuing could leave a partial, unintentionally unhedged position."""
    results = []
    for item in legs_to_place:
        txn_type = kite.TRANSACTION_TYPE_SELL if item["transaction_type"] == "SELL" else kite.TRANSACTION_TYPE_BUY
        quantity = int(item.get("quantity") or 1)
        reference_price = item.get("price")
        try:
            kwargs = dict(
                variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                tradingsymbol=item["tradingsymbol"], transaction_type=txn_type,
                quantity=quantity, product=getattr(kite, f"PRODUCT_{product}"),
                order_type=getattr(kite, f"ORDER_TYPE_{order_type}"),
                validity=kite.VALIDITY_DAY,
            )
            if order_type == "LIMIT" and reference_price:
                kwargs["price"] = float(reference_price)
            order_id = kite.place_order(**kwargs)
            results.append({"leg": item.get("leg", "?"), "tradingsymbol": item["tradingsymbol"],
                             "transaction_type": item["transaction_type"], "quantity": quantity,
                             "status": "placed", "order_id": order_id,
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
            "refreshed_at": datetime.now().strftime("%H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": f"Could not refresh live Bid/Ask: {e}"}), 502


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
    quotes = kite.quote(inst_keys)

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
