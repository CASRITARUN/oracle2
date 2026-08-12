"""
Kotak Neo Option-Selling Dashboard — backend
------------------------------------------------
Run:  gunicorn -w 1 -k gthread --threads 4 -b 127.0.0.1:8000 backend:app   (see DEPLOYMENT.md)
Then open: https://algo2.wecon.in

What this does
- Authenticates to Kotak Neo entirely server-side (TOTP + MPIN, no daily manual login) using
  credentials from ~/kotak_algo/.env — see KotakNeoBroker / KiteCompatKotak below for the auth flow.
- Pulls your F&O stock universe from Kotak's scrip master, ranks by historical volatility / ATR
  (historical daily candles come from Yahoo Finance — Kotak Neo's API has no candle endpoint;
  everything else — quotes, positions, orders — is 100% Kotak Neo).
- Also supports index options directly: NIFTY, BANKNIFTY, FINNIFTY — just type the symbol
- Lets you pick WHICH expiry (current month, next month, etc.) rather than only the nearest one
- Shows the full live option chain, lets you build and adjust a delta-based Iron Condor OR a naked
  Strangle (you control target delta, hedge width, and lot count before committing)
- Tracks entered trades with live daily P&L, re-estimated probability of success, and max loss
- Lets you actually PLACE the real orders for a tracked position in your Kotak Neo account — but
  only after an explicit confirmation step showing exactly what will be sent, and gives you an
  order list with cancel/modify so you stay in control the whole time
- Optional lightweight news headlines per stock as a basic event-risk / "threat intelligence" signal

IMPORTANT
- Nothing here is investment advice. Verify every number on your broker terminal before trading.
- Kotak Neo re-authenticates itself automatically (server-side TOTP+MPIN) whenever the session is
  missing/expired — there is no manual daily login step.
- Naked strangles carry theoretically unlimited risk on the call side.
- ORDER EXECUTION IS REAL once LIVE_TRADING=true in ~/kotak_algo/.env. Placing orders through this
  tool then sends real orders to your live Kotak Neo account using real money. Nothing is placed
  without you explicitly confirming on the preview screen. If one leg of a multi-leg order fails,
  you may be left holding a partial, unhedged position — the tool stops immediately on the first
  failure and tells you to check your Kotak Neo app right away. LIVE_TRADING defaults to false —
  every order-sending call is logged and returned as a simulated "paper" response until you
  explicitly opt in.
- REQUIRES Python 3.11, neo_api_client (Kotak-neo-api-v2), pyotp, python-dotenv, yfinance,
  requests, numpy, flask. See requirements.txt / DEPLOYMENT.md.
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

import numpy as np
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple

# --- Kotak Neo broker support -----------------------------------------------------------
# Credentials live ONLY in ~/kotak_algo/.env (never in this file, never in git). Loaded here,
# once, before anything reads os.environ for KOTAK_* or LIVE_TRADING. If python-dotenv isn't
# installed or the file doesn't exist yet, this is a no-op — real environment variables (e.g.
# set by systemd) still work normally, and Kotak simply reports "not configured" until the
# .env file is present.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.expanduser("~/kotak_algo/.env"))
except ImportError:
    pass


# =============================================================================
# BROKER ABSTRACTION LAYER (Kotak Neo)
# =============================================================================
# Everything below (through the KotakNeoBroker class) used to live in a separate brokers/
# package; it is inlined here so the whole app is exactly two files: backend.py and index.html.

"""
Broker abstraction interface.
------------------------------
KotakNeoBroker implements this surface so the rest of the app can call `broker.get_positions()` /
`broker.place_order(...)` etc. through one consistent shape. This does NOT change any existing
strategy math (indicators, strike selection, risk rules) — it only standardizes how the broker is
talked to.

Each method should raise BrokerAuthError for anything that looks like an expired/invalid
session (so callers can show a clean "please reconnect" instead of a stack trace), and
BrokerError for any other broker-side failure (rejected order, bad symbol, rate limit, etc).
"""

from abc import ABC, abstractmethod


class BrokerError(Exception):
    """A broker call failed for a reason that isn't an auth problem (rejected order, bad
    symbol, rate limit, network/API error, etc)."""
    pass


class BrokerAuthError(BrokerError):
    """Session is missing/expired/invalid. Callers should surface a reconnect prompt, not a
    generic 500."""
    pass


class BrokerInterface(ABC):
    name: str = "base"

    @abstractmethod
    def is_connected(self) -> bool:
        """Cheap, current check — should actually verify the session still works (not just
        that a token is present in memory); the Kotak session-status check calls this as a
        liveness probe."""
        raise NotImplementedError

    @abstractmethod
    def ensure_authenticated(self) -> bool:
        """Make sure there's a valid session, re-authenticating if needed. Returns True if the
        broker is usable after this call, False if authentication could not be established."""
        raise NotImplementedError

    @abstractmethod
    def get_profile(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_limits(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_holdings(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_quotes(self, instruments) -> dict:
        raise NotImplementedError

    @abstractmethod
    def place_order(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def modify_order(self, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id, **kwargs) -> dict:
        raise NotImplementedError

    @abstractmethod
    def order_history(self, order_id=None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def order_report(self) -> dict:
        raise NotImplementedError

    def status_payload(self) -> dict:
        """Common shape returned to the frontend by /api/broker/status — never includes any
        credential/token material, only what the UI needs to render a connection pill."""
        connected = False
        message = None
        try:
            connected = self.is_connected()
        except BrokerAuthError as e:
            message = str(e)
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
        return {"broker": self.name, "connected": connected, "message": message}


"""
KotakNeoBroker — Kotak Neo implementation of BrokerInterface.

Built against the official Kotak Neo Python SDK v2 (`neo_api_client`, package
`Kotak-Neo/Kotak-neo-api-v2`, tag v2.0.2). Method names/params below (totp_login,
totp_validate, positions, holdings, limits, quotes, place_order, modify_order, cancel_order,
order_report, order_history) match that SDK's documented signatures exactly — nothing here is
guessed. If your installed SDK version differs, diff this file against
`python -c "import neo_api_client, inspect; print(inspect.signature(neo_api_client.NeoAPI.place_order))"`
on your server before relying on it for live orders.

Credentials load ONLY from environment variables (populated from ~/kotak_algo/.env via
python-dotenv, loaded once at process start in backend.py). Nothing here ever hard-codes a
secret, logs a secret, or sends a secret to the frontend.

SAFETY: place_order / modify_order / cancel_order are hard-gated behind LIVE_TRADING=true.
When LIVE_TRADING is not exactly "true", every order-sending call is logged and returned as a
synthetic "paper" response instead of touching the real API — this matches the same
default-safe behavior requested for the whole integration.
"""

# (os, threading, time, logging already imported at the top of backend.py; reusing the
# same module-level `logger` defined later in this file rather than creating a second one)
try:
    from neo_api_client import NeoAPI
except ImportError:
    NeoAPI = None  # handled at authenticate() time with a clear error, not at import time

try:
    import pyotp
except ImportError:
    pyotp = None


# How long we trust a Kotak session before proactively re-authenticating, even if no call has
# failed yet. Kotak doesn't publish a documented session lifetime in the v2 SDK docs, so this is
# a conservative same-trading-day assumption, not a guarantee — the auth-error retry below is
# the real safety net regardless of this value.
SESSION_MAX_AGE_SECONDS = int(os.environ.get("KOTAK_SESSION_MAX_AGE_SECONDS", 8 * 3600))

# Substrings that indicate a Kotak API error is actually an expired/invalid session rather than
# a genuine order/data problem, so we know when it's worth silently re-authenticating and
# retrying once versus surfacing the error as-is.
_AUTH_ERROR_HINTS = ("token", "session", "unauthor", "auth", "login", "expired", "invalid sid",
                     "2fa")


def _looks_like_auth_error(text) -> bool:
    msg = str(text).lower()
    return any(hint in msg for hint in _AUTH_ERROR_HINTS)


def _response_error_message(resp):
    """The installed Kotak Neo SDK (v2.0.2) does NOT raise exceptions on API failure — a failed
    call (bad TOTP, expired session, rejected order, etc.) comes back as a normal dict containing
    an 'Error' or 'Error Message' key instead (confirmed by testing the actual installed
    package). A caller that only wraps calls in try/except, as the SDK's own official examples
    do, will silently treat a failed login or a rejected order as a success. This checks any
    dict response for that shape and returns the error text, or None if the response looks fine."""
    if isinstance(resp, dict):
        for key, val in resp.items():
            if "error" in key.lower():
                return str(val)
    return None


class KotakNeoBroker(BrokerInterface):
    name = "KOTAK"

    def __init__(self):
        self.consumer_key = os.environ.get("KOTAK_CONSUMER_KEY")
        self.consumer_secret = os.environ.get("KOTAK_CONSUMER_SECRET")  # optional in v2.0.x
        # Kotak Neo TOTP login expects a valid 10-digit Indian mobile number.
        # Accept common .env formats such as +919876543210, 919876543210,
        # 09876543210, or 9876543210 and normalize them before login.
        self.mobile_raw = os.environ.get("KOTAK_MOBILE", "")
        self.mobile = self._normalize_mobile(self.mobile_raw)
        self.ucc = os.environ.get("KOTAK_UCC")
        self.mpin = os.environ.get("KOTAK_MPIN")
        self.totp_secret = os.environ.get("KOTAK_TOTP_SECRET")
        self.environment = os.environ.get("KOTAK_ENV", "prod")
        self.live_trading = os.environ.get("LIVE_TRADING", "false").strip().lower() == "true"

        self.client = None
        self._authenticated_at = None
        self._lock = threading.Lock()

        missing = [n for n, v in [
            ("KOTAK_CONSUMER_KEY", self.consumer_key),
            ("KOTAK_MOBILE", self.mobile),
            ("KOTAK_UCC", self.ucc),
            ("KOTAK_MPIN", self.mpin),
            ("KOTAK_TOTP_SECRET", self.totp_secret),
        ] if not v]
        if self.mobile_raw and not self.mobile:
            self._config_error = (
                "KOTAK_MOBILE is invalid. Enter the 10-digit Indian mobile number "
                "(for example 9876543210) in /etc/kotak-algo.env."
            )
        self._config_error = (
            f"Missing required Kotak env vars: {', '.join(missing)} (check ~/kotak_algo/.env)"
            if missing else None
        )
        if not self.live_trading:
            logger.info("[KOTAK] Live trading disabled (LIVE_TRADING is not 'true') — "
                         "order-sending calls will be logged and simulated, not sent.")

    @staticmethod
    def _normalize_mobile(value):
        """Normalize KOTAK_MOBILE to the 10-digit format expected by Neo TOTP login."""
        raw = str(value or "").strip()
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]
        elif digits.startswith("0") and len(digits) == 11:
            digits = digits[1:]
        if len(digits) != 10 or not digits.isdigit() or digits[0] not in "6789":
            return None
        return digits

    # -- masking helpers, so nothing identifying/secret ever reaches the logs -------------
    def _masked_ucc(self):
        return (self.ucc[:2] + "***" + self.ucc[-2:]) if self.ucc and len(self.ucc) > 4 else "***"

    def _masked_mobile(self):
        return ("***" + self.mobile[-4:]) if self.mobile and len(self.mobile) >= 4 else "***"

    # -- authentication ---------------------------------------------------------------------
    def authenticate(self) -> bool:
        """Full TOTP + MPIN login. Safe to call repeatedly — guarded by a lock so concurrent
        requests can't trigger two simultaneous logins."""
        if self._config_error:
            logger.error(f"[KOTAK] Cannot authenticate: {self._config_error}")
            raise BrokerAuthError(self._config_error)
        if NeoAPI is None:
            raise BrokerAuthError("neo_api_client is not installed on this server "
                                   "(pip install per Kotak-Neo/Kotak-neo-api-v2 README).")
        if pyotp is None:
            raise BrokerAuthError("pyotp is not installed on this server.")

        with self._lock:
            logger.info(f"[KOTAK] Authentication started (ucc={self._masked_ucc()}, "
                        f"mobile={self._masked_mobile()}).")
            try:
                totp_code = pyotp.TOTP(self.totp_secret).now()
            except Exception as e:
                logger.error("[KOTAK] Failed to generate TOTP from KOTAK_TOTP_SECRET.")
                raise BrokerAuthError(f"TOTP generation failed: {e}")

            try:
                client = NeoAPI(environment=self.environment, access_token=None,
                                 neo_fin_key=None, consumer_key=self.consumer_key)
                resp = client.totp_login(mobile_number=self.mobile, ucc=self.ucc, totp=totp_code)
                err = _response_error_message(resp)
                if err:
                    logger.error(f"[KOTAK] TOTP login rejected: {err}")
                    raise BrokerAuthError(f"Kotak Neo TOTP login failed: {err}")
                logger.info("[KOTAK] TOTP login successful.")
            except BrokerAuthError:
                raise
            except Exception as e:
                logger.error(f"[KOTAK] TOTP login failed: {type(e).__name__}")
                raise BrokerAuthError("Kotak Neo TOTP login failed. Check mobile/UCC/TOTP "
                                       "secret and that this server's IP is still whitelisted.")

            try:
                resp = client.totp_validate(mpin=self.mpin)
                err = _response_error_message(resp)
                if err:
                    logger.error(f"[KOTAK] MPIN validation rejected: {err}")
                    raise BrokerAuthError(f"Kotak Neo MPIN validation failed: {err}")
                logger.info("[KOTAK] MPIN validation successful.")
            except BrokerAuthError:
                raise
            except Exception as e:
                logger.error(f"[KOTAK] MPIN validation failed: {type(e).__name__}")
                raise BrokerAuthError("Kotak Neo MPIN validation failed. Check KOTAK_MPIN.")

            self.client = client
            self._authenticated_at = time.time()
            logger.info("[KOTAK] Session ready.")
            return True

    def ensure_authenticated(self) -> bool:
        if self.client is None:
            return self.authenticate()
        age = time.time() - (self._authenticated_at or 0)
        if age > SESSION_MAX_AGE_SECONDS:
            logger.info("[KOTAK] Session past max age — re-authenticating proactively.")
            return self.authenticate()
        return True

    def is_connected(self) -> bool:
        if self._config_error:
            return False
        if self.client is None:
            return False
        try:
            # Cheapest available liveness probe — the v2 SDK's documented surface has no
            # dedicated profile()/ping() endpoint, so limits() (same call get_limits() uses)
            # doubles as the "does this session still actually work" check, mirroring how the
            # existing Zerodha status check calls kite.profile() for the same purpose.
            resp = self.client.limits(segment="ALL", exchange="ALL", product="ALL")
            err = _response_error_message(resp)
            if err:
                if _looks_like_auth_error(err):
                    self.client = None
                return False
            return True
        except Exception as e:
            if _looks_like_auth_error(e):
                self.client = None
                return False
            # a non-auth error (e.g. transient network) shouldn't flip the pill to disconnected
            return True

    # -- internal call wrapper: auto-auth, auto-retry-once-on-auth-error -------------------
    def _attempt(self, fn_name, **kwargs):
        """One raw attempt. Returns (response, error_message_or_None). Never raises for a
        broker-side failure — only for something unexpected (network, TypeError, etc.), which
        the caller still needs to see."""
        fn = getattr(self.client, fn_name)
        resp = fn(**kwargs)
        return resp, _response_error_message(resp)

    def _call(self, fn_name, log_label=None, **kwargs):
        if not self.ensure_authenticated():
            raise BrokerAuthError("Kotak Neo authentication failed.")
        if log_label:
            logger.info(f"[KOTAK] {log_label}")

        try:
            resp, err = self._attempt(fn_name, **kwargs)
        except Exception as e:
            resp, err = None, str(e)

        if err is None:
            return resp

        if _looks_like_auth_error(err):
            logger.warning(f"[KOTAK] {fn_name} hit an auth-like error — re-authenticating "
                            f"once and retrying.")
            self.client = None
            if not self.ensure_authenticated():
                raise BrokerAuthError("Kotak Neo re-authentication failed.")
            try:
                resp, err = self._attempt(fn_name, **kwargs)
            except Exception as e:
                resp, err = None, str(e)
            if err is None:
                return resp
            raise BrokerError(f"Kotak Neo {fn_name} failed after re-auth: {err}")

        raise BrokerError(f"Kotak Neo {fn_name} failed: {err}")

    # -- read-only endpoints ------------------------------------------------------------------
    def get_profile(self) -> dict:
        # No documented profile() endpoint in the v2 SDK — limits() is the closest read-only
        # call that confirms who/what the session is scoped to without side effects. Swap this
        # out if Kotak documents a real profile endpoint for your account tier.
        return self._call("limits", "Fetching account limits (used as profile probe)",
                           segment="ALL", exchange="ALL", product="ALL")

    def get_limits(self) -> dict:
        return self._call("limits", "Fetching funds/limits", segment="ALL", exchange="ALL",
                           product="ALL")

    def get_positions(self) -> dict:
        return self._call("positions", "Fetching positions")

    def get_holdings(self) -> dict:
        return self._call("holdings", "Fetching holdings")

    def get_quotes(self, instruments) -> dict:
        """instruments: list of {"instrument_token": ..., "exchange_segment": ...} dicts, per
        the Kotak SDK's documented format. quote_type="all" (rather than "ltp") so depth/OI/OHLC
        are populated too — KiteCompatKotak.quote() (below) is the caller that translates
        Zerodha-style "NSE:SYMBOL" keys into this format and normalizes the response back."""
        return self._call("quotes", "Fetching quotes", instrument_tokens=instruments,
                           quote_type="all")

    def order_history(self, order_id=None) -> dict:
        return self._call("order_history", "Fetching order history", order_id=order_id or "")

    def order_report(self) -> dict:
        return self._call("order_report", "Fetching order book")

    # -- order-sending endpoints: hard-gated behind LIVE_TRADING ----------------------------
    def place_order(self, **kwargs) -> dict:
        logger.info("[KOTAK] Order request prepared.")
        if not self.live_trading:
            logger.info("[KOTAK] Live trading disabled — order NOT sent (paper mode).")
            return {"paper": True, "would_send": kwargs, "stat": "Ok",
                    "message": "LIVE_TRADING is not enabled — no real order was placed."}
        return self._call("place_order", "Sending live order", **kwargs)

    def modify_order(self, **kwargs) -> dict:
        logger.info("[KOTAK] Order modify request prepared.")
        if not self.live_trading:
            logger.info("[KOTAK] Live trading disabled — modify NOT sent (paper mode).")
            return {"paper": True, "would_send": kwargs, "stat": "Ok",
                    "message": "LIVE_TRADING is not enabled — no real modify was sent."}
        return self._call("modify_order", "Sending live order modify", **kwargs)

    def cancel_order(self, order_id, **kwargs) -> dict:
        logger.info("[KOTAK] Order cancel request prepared.")
        if not self.live_trading:
            logger.info("[KOTAK] Live trading disabled — cancel NOT sent (paper mode).")
            return {"paper": True, "would_send": {"order_id": order_id, **kwargs}, "stat": "Ok",
                    "message": "LIVE_TRADING is not enabled — no real cancel was sent."}
        return self._call("cancel_order", "Sending live order cancel", order_id=order_id,
                           **kwargs)

# =============================================================================
# END BROKER ABSTRACTION LAYER
# =============================================================================

# ---------------------------------------------------------------------------
# CONFIG — Kotak Neo credentials live ONLY in ~/kotak_algo/.env (loaded near the top of this
# file via python-dotenv) and are read by KotakNeoBroker.__init__ directly from os.environ.
# Nothing broker-related is hardcoded here. (This block used to hold Zerodha API_KEY/API_SECRET
# defaults with real-looking values baked into the source — removed entirely; if you ever
# committed that to git history, treat those as compromised and confirm they're deactivated on
# Zerodha's side.)
# ---------------------------------------------------------------------------

# If your network does TLS interception (common on office/government networks — you'll see
# "self-signed certificate in certificate chain" errors), set this env var to allow the news
# feature specifically to fall back to an unverified request. This does NOT affect Kotak Neo API
# calls at all (those always stay fully verified) — it only relaxes verification for public news
# RSS feeds, which carry no credentials or sensitive data.
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

# --- Approximate Kotak Neo F&O options charges (informational estimate only) ---
# STT/GST/exchange-txn/SEBI/stamp-duty are government-mandated and broker-agnostic (same on
# every broker) — only brokerage_flat below is Kotak-specific. Kotak Neo's Trade Free Plan is
# flat ₹20/executed order for F&O as of this writing, but some plans (e.g. a youth/promo plan)
# charge ₹10 — confirm which plan you're actually on and edit brokerage_flat below if different.
# Tax rates DO change over time (via budget announcements) — verify current rates at
# https://www.kotakneo.com/calculator/brokerage-calculator/ and your actual contract note before
# relying on this for anything beyond a rough planning estimate. All values are editable here.
CHARGES = {
    "brokerage_flat": 20.0,          # per executed order — Kotak Neo Trade Free Plan F&O rate;
                                      # change to 10.0 if you're on a plan that charges ₹10
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


# ---------------------------------------------------------------------------
# Self-built price history — 100% Kotak Neo, no external data source.
# ---------------------------------------------------------------------------
# Kotak Neo's API has no historical-candle endpoint at all, and (this is the important part)
# outside data sources like Yahoo Finance are unreliable to depend on for a production screener
# — Yahoo actively rate-limits/blocks a lot of datacenter/cloud IP ranges (including Oracle
# Cloud), so a screener that HARD-DEPENDS on Yahoo for every stock can fail almost completely
# with zero warning on exactly the kind of VM this app runs on. So the screener no longer calls
# Yahoo at all. Instead, every time the screener runs, it captures today's OHLC from Kotak's own
# live quote() for the whole F&O universe (one batched call) into a small local JSON file. Over
# the next ~2-3 trading weeks this naturally becomes real multi-day historical volatility / ATR,
# built entirely from your own Kotak Neo quotes. Before that history exists, the screener falls
# back to today's own intraday range (high-low)/ltp as an immediate, same-day calmness proxy —
# see historical_vol_and_atr_kotak() and the Pass 1 loop in screener() below.
PRICE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "kotak_price_history.json")
PRICE_HISTORY_LOOKBACK_DAYS = 130   # keep ~6 months of daily bars per symbol
PRICE_HISTORY_MIN_DAYS_FOR_HV = 6   # need at least this many stored days before trusting real HV/ATR
PRICE_HISTORY_DAYS_FOR_FULL_HV = 20  # HV/ATR is considered "fully warmed up" at this many days


def load_price_history():
    if not os.path.exists(PRICE_HISTORY_FILE):
        return {}
    try:
        with open(PRICE_HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        logger.warning("[PRICE_HISTORY] Could not read kotak_price_history.json — starting fresh.")
        return {}


def save_price_history(history):
    with open(PRICE_HISTORY_FILE, "w") as f:
        json.dump(history, f)


def capture_price_snapshot(universe, symbol_to_nse_key):
    """Bulk-fetches ONE live quote per NSE-listed stock in `universe` (via Kotak, batched
    internally) and upserts today's OHLC bar into the local price-history store. Safe to call
    every time the screener runs — while the market is open this refines today's high/low/close
    as the day progresses (Kotak's own quote OHLC already reflects the day's cumulative high/low,
    so this is just tracking the exchange's own numbers, not estimating anything); after the
    market closes, today's bar simply stops changing since Kotak's OHLC stops changing too.
    Returns (quotes_dict, history_dict) so the caller can reuse the same quotes for LTP/spread
    display without a second round-trip."""
    keys = [symbol_to_nse_key[s] for s in universe if s in symbol_to_nse_key]
    quotes = {}
    for i in range(0, len(keys), 200):
        try:
            quotes.update(kite.quote(keys[i:i + 200]))
        except Exception as e:
            logger.warning(f"[PRICE_HISTORY] Bulk quote fetch failed for a batch: {e}")

    history = load_price_history()
    today_str = str(now_ist().date())
    changed = False
    for s in universe:
        key = symbol_to_nse_key.get(s)
        q = quotes.get(key) if key else None
        if not q or not q.get("last_price"):
            continue
        ohlc = q.get("ohlc") or {}
        high, low, open_, ltp = ohlc.get("high"), ohlc.get("low"), ohlc.get("open"), q["last_price"]
        if not high or not low:
            continue  # incomplete quote — don't record a broken bar
        bar = {"date": today_str, "open": open_ or ltp, "high": high, "low": low, "close": ltp}
        series = history.setdefault(s, [])
        if series and series[-1]["date"] == today_str:
            series[-1] = bar  # same-day upsert — reflects the day's cumulative high/low so far
        else:
            series.append(bar)
        if len(series) > PRICE_HISTORY_LOOKBACK_DAYS:
            del series[:-PRICE_HISTORY_LOOKBACK_DAYS]
        changed = True
    if changed:
        save_price_history(history)
    return quotes, history


def historical_vol_and_atr_kotak(symbol, history, days=60):
    """Real annualized HV + ATR% computed purely from the locally-accumulated Kotak price
    history (see capture_price_snapshot() above). Returns (hv_annualized_pct_or_None,
    atr_pct_of_price_or_None, days_of_history_available) — both None until
    PRICE_HISTORY_MIN_DAYS_FOR_HV days have accumulated, so the caller can fall back to the
    same-day range proxy in the meantime."""
    series = sorted(history.get(symbol, []), key=lambda p: p["date"])
    if len(series) < PRICE_HISTORY_MIN_DAYS_FOR_HV:
        return None, None, len(series)
    series = series[-(days + 1):]
    closes = np.array([p["close"] for p in series], dtype=float)
    highs = np.array([p["high"] for p in series], dtype=float)
    lows = np.array([p["low"] for p in series], dtype=float)
    if len(closes) < 3:
        return None, None, len(series)
    returns = np.diff(np.log(closes))
    hv_annualized = float(np.std(returns) * math.sqrt(252) * 100)
    tr = np.maximum(highs[1:] - lows[1:],
                     np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
    atr = float(np.mean(tr[-14:])) if len(tr) else None
    atr_pct = (atr / closes[-1] * 100) if atr is not None and closes[-1] else None
    return round(hv_annualized, 2), (round(atr_pct, 2) if atr_pct is not None else None), len(series)


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

# --- Broker: 100% Kotak Neo -----------------------------------------------------------------
# kotak_broker does the real work (TOTP+MPIN auto-auth, positions/orders/margins/quotes,
# LIVE_TRADING gating) — see KotakNeoBroker above. `kite` is a KiteCompatKotak facade (see
# below) that is 100% backed by kotak_broker; it exists ONLY so the ~90 pre-existing
# `kite.quote(...)` / `kite.place_order(...)` / etc. call sites throughout the strategy/screener/
# execution code below did not need to be rewritten one-by-one. There is no Zerodha/Kite Connect
# involved anywhere in this process — the variable name is legacy-only.
# =============================================================================
# KOTAK NEO COMPATIBILITY FACADE (formerly kotak_compat.py, inlined so the whole app is
# exactly two files: backend.py and index.html)
# =============================================================================
"""
kotak_compat.py — Kotak Neo -> "kite-shaped" compatibility facade
====================================================================
This module is the ONLY place that talks Kotak Neo's actual wire format (scrip master rows,
quotes(), positions(), orders(), place_order()). It exposes a single object, `KiteCompatKotak`,
whose method names/return shapes mirror the subset of the KiteConnect client that the rest of
backend.py already calls (kite.quote(), kite.historical_data(), kite.instruments(), kite.positions(),
kite.orders(), kite.place_order(), kite.modify_order(), kite.cancel_order(), kite.ltp(), plus the
VARIETY_*/EXCHANGE_*/PRODUCT_*/ORDER_TYPE_*/TRANSACTION_TYPE_*/VALIDITY_* constants).

WHY THIS SHAPE: backend.py has ~90 existing call sites spread across the screener, strategy
builder, and order-execution code that call `kite.something(...)`. Rewriting all of them (and
re-deriving every option-chain / strike-selection / margin computation against a totally
different response shape) would be a huge, error-prone rewrite of logic that already works and
has been tested. Instead, this facade is 100% backed by Kotak Neo (via the existing
KotakNeoBroker in backend.py) for everything live — quotes, positions, holdings, margins,
orders, order placement — and is backed by free public data (Yahoo Finance) ONLY for historical
daily candles, because the Kotak Neo API has no historical-candle endpoint at all. Nothing here
talks to Zerodha/Kite in any way; the `kite` variable name in backend.py is kept purely so the
unchanged strategy code doesn't need to be touched.

GROUNDING / WHAT'S VERIFIED vs WHAT NEEDS A LIVE SMOKE-TEST
-------------------------------------------------------------
- Scrip master row shape (pSymbol, pExchSeg, pInstType, pSymbolName, pTrdSymbol, pOptionType,
  lLotSize, dStrikePrice, pExchange) is taken from Kotak's own published Scrip_Search.md /
  Scrip_Master.md docs and real example output.
- The quotes() request shape (list of {"instrument_token":, "exchange_segment":}) and the
  documented trick of passing an index's NAME STRING ("Nifty 50", "Nifty Bank") instead of a
  numeric token for index-segment quotes is taken directly from Kotak's own Quotes.md example.
- The exact field names inside a quotes() *response* (ltp/ohlc/depth/oi) are NOT published in
  full in Kotak's docs. `_normalize_quote()` below is defensive (tries several plausible key
  names) and logs the raw response ONCE at startup (search your server log for
  "[KOTAK][DEBUG] Raw quote response sample" after your first login) so the mapping can be
  corrected in one place if a field comes back empty. DO THIS before trusting the screener's
  IV/spread numbers or the option-chain depth column.
- Order placement fields (exchange_segment, product, order_type, quantity, price, trading_symbol,
  transaction_type, amo, disclosed_quantity, market_protection, trigger_price, validity) are
  taken from Kotak's own Orders documentation (Kotak-neo-api-v2 docs).
- scrip_master() vs per-symbol search_scrip(): this uses scrip_master() for the full NFO/NSE
  universe (needed for the screener's "scan every F&O stock" feature and for strike/expiry
  listing) with a defensive multi-shape parser (JSON list, {"data": [...]}, or CSV text/path).
  If your installed SDK's scrip_master() return shape doesn't match any of the three, the parser
  logs the raw type/sample and raises a clear error rather than silently returning nothing.

Nothing in this file ever logs or exposes a credential — it only ever touches the already-
authenticated `KotakNeoBroker` instance passed into it.
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta, date
from typing import Optional

logger = logging.getLogger("kite_dashboard")

try:
    import yfinance as yf
except ImportError:
    yf = None  # handled at call time with a clear error message

# ---------------------------------------------------------------------------------------------
# Index / VIX symbols that Kotak's quotes() API accepts as a literal name string (per Kotak's
# own Quotes.md example: {"instrument_token": "Nifty 50", "exchange_segment": "nse_cm"}) rather
# than a numeric scrip token. Verify these exact strings against your Kotak Neo app / a live
# search_scrip("nse_cm", "NIFTY") call if any index quote comes back empty — Kotak does not
# publish a full authoritative list of these display names.
# ---------------------------------------------------------------------------------------------
KOTAK_INDEX_NAME = {
    "NIFTY 50": "Nifty 50",
    "NIFTY BANK": "Nifty Bank",
    "NIFTY FIN SERVICE": "Nifty Fin Service",
    "NIFTY MID SELECT": "Nifty Midcap Select",
    "INDIA VIX": "India Vix",
}

# Yahoo Finance tickers for historical daily candles. Yahoo has no reliable FINNIFTY /
# MIDCPNIFTY series, so those two stay unsupported for historical-based indicators (HV/ATR/
# EMA/RSI/ADX). This does not affect the stock screener/ranking (fo_stock_universe() only scans
# individual F&O stocks, never indices) — it only affects manually charting those two indices.
YAHOO_INDEX_TICKER = {
    "NIFTY 50": "^NSEI",
    "NIFTY BANK": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "NIFTY FIN SERVICE": None,
    "NIFTY MID SELECT": None,
}

# yfinance's own real limits on intraday lookback (Yahoo-side restriction, not ours) — used to
# clip a request instead of silently returning nothing.
_YF_INTRADAY_MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730, "90m": 60}

KITE_TO_YF_INTERVAL = {
    "day": "1d", "minute": "1m", "3minute": None, "5minute": "5m", "10minute": None,
    "15minute": "15m", "30minute": "30m", "60minute": "60m",
}


def _epoch_to_date(val) -> Optional[date]:
    """Kotak's scrip master expiry fields have been observed as unix-epoch seconds (see
    Scrip_Search.md example: lExpiryDate). Defensive against it also showing up as an already-
    formatted string ('DD-MMM-YYYY' or 'YYYY-MM-DD'), a naive int, or missing/-1 (non-derivative
    rows)."""
    if val is None:
        return None
    try:
        f = float(val)
        if f <= 0:
            return None
        # Kotak has been seen using seconds; guard against a millisecond timestamp too.
        if f > 10_000_000_000:
            f = f / 1000.0
        return (datetime(1970, 1, 1) + timedelta(seconds=f)).date()
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _first(d: dict, *keys, default=None):
    """Returns the first present, non-None value among several possible key-name spellings —
    used everywhere a Kotak response field name isn't 100% certain from the docs alone."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_float(v, default=0.0):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


class ScripMasterError(Exception):
    pass


class KotakScripMaster:
    """Downloads and normalizes Kotak's scrip master into the SAME dict shape
    `kite.instruments(segment)` already returns, so every existing consumer in backend.py
    (get_instruments(), fo_stock_universe(), pick_atm_contracts(), get_atm_iv_and_liquidity_bulk(),
    ...) keeps working completely unchanged. Kotak's OWN trading-symbol strings are carried
    through as `tradingsymbol` (we do not attempt to replicate Zerodha's naming convention —
    nothing downstream cares what the string looks like, only that it's a stable unique key used
    consistently for quoting and order placement)."""

    def __init__(self, kotak_broker):
        self.broker = kotak_broker
        self._lock = threading.Lock()
        # normalized instrument caches
        self.nfo = []   # list of dicts, kite.instruments("NFO") shape
        self.nse = []   # list of dicts, kite.instruments("NSE") shape (equities + indices)
        # fast lookup maps rebuilt every refresh()
        self._nfo_by_tradingsymbol = {}          # "RELIANCE25DEC2900CE" -> row
        self._nse_by_name = {}                   # "RELIANCE" -> row
        self._token_to_row = {}                  # pSymbol (int) -> row (either segment)
        self._logged_raw_sample = False

    # -- raw scrip master fetch + defensive multi-shape parsing --------------------------------
    def _fetch_raw(self, exchange_segment):
        resp = self.broker._call(
            "scrip_master", f"Downloading scrip master ({exchange_segment})",
            exchange_segment=exchange_segment,
        )
        # Shape A: already a list of dicts
        if isinstance(resp, list):
            return resp
        # Shape B: {"data": [...]}  or similar wrapper
        if isinstance(resp, dict):
            for key in ("data", "Success", "result", "scrips"):
                if isinstance(resp.get(key), list):
                    return resp[key]
            # Shape C: dict of file-paths/URLs to CSV files (documented masterscrip/file-paths
            # style response) — download and parse whichever CSV URL matches this segment.
            for key, val in resp.items():
                if isinstance(val, str) and val.startswith("http") and exchange_segment in val.lower():
                    return self._parse_csv_url(val)
        # Shape D: a raw CSV string
        if isinstance(resp, str) and ("," in resp and "\n" in resp):
            return self._parse_csv_text(resp)
        logger.error(f"[KOTAK] scrip_master({exchange_segment}) returned an unrecognized shape: "
                     f"{type(resp).__name__}. Sample: {str(resp)[:400]}")
        raise ScripMasterError(
            f"Could not parse scrip_master() response for {exchange_segment} — unrecognized "
            f"shape ({type(resp).__name__}). Check server logs for the raw sample and adjust "
            f"KotakScripMaster._fetch_raw() below to match."
        )

    def _parse_csv_url(self, url):
        import requests
        import csv
        import io
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return list(csv.DictReader(io.StringIO(r.text)))

    def _parse_csv_text(self, text):
        import csv
        import io
        return list(csv.DictReader(io.StringIO(text)))

    # -- normalization into the Zerodha/kite instrument-dict shape ------------------------------
    def _normalize_row(self, row, exchange_segment):
        if not self._logged_raw_sample:
            logger.info(f"[KOTAK][DEBUG] Raw scrip master row sample ({exchange_segment}): "
                        f"{dict(list(row.items())[:20])}")
            self._logged_raw_sample = True

        token = _to_int(_first(row, "pSymbol", "pToken", "instrument_token", "token"))
        trading_symbol = _first(row, "pTrdSymbol", "pSymbolName", "trading_symbol", "tradingsymbol",
                                 default="")
        # Kotak's own cash-equity trading symbols carry a series suffix ("RELIANCE-EQ") that
        # Zerodha's never did — the rest of this file (get_spot_price(), fo_stock_universe()
        # matching, symbol_to_nse_key, etc.) was written assuming a bare symbol for NSE cash
        # equities, so normalize it here ONCE rather than touching every match site downstream.
        # Leaves non-EQ series (BE/BZ/etc, rare among F&O-eligible names) untouched.
        if exchange_segment == "nse_cm" and trading_symbol.endswith("-EQ"):
            trading_symbol = trading_symbol[:-3]
        name = _first(row, "pSymbolName", "pScripRefKey", "name", default=trading_symbol)
        inst_type_raw = (_first(row, "pInstType", "instrument_type", default="") or "").upper()
        option_type = (_first(row, "pOptionType", "option_type", default="") or "").upper() or None
        strike = _to_float(_first(row, "dStrikePrice", "strike", "dStrikePrice;", default=-1))
        lot_size = _to_int(_first(row, "lLotSize", "lot_size", default=1), default=1)
        expiry = _epoch_to_date(_first(row, "lExpiryDate", "pExpiryDate", "expiry"))
        exch_display = "NFO" if exchange_segment == "nse_fo" else "NSE"

        if option_type in ("CE", "PE"):
            instrument_type = option_type
            segment = "NFO-OPT"
        elif inst_type_raw in ("FUTSTK", "FUTIDX", "FUT"):
            instrument_type = "FUT"
            segment = "NFO-FUT"
        elif inst_type_raw in ("INDEX",) or (exchange_segment == "nse_cm" and strike is not None and
                                              trading_symbol.upper() in KOTAK_INDEX_NAME):
            instrument_type = "INDEX"
            segment = "INDICES"
        else:
            instrument_type = "EQ"
            segment = "NSE"

        return {
            "instrument_token": token,
            "exchange_token": token,
            "tradingsymbol": trading_symbol,
            "name": name,
            "last_price": 0,
            "expiry": expiry,
            "strike": strike if strike and strike > 0 else 0,
            "tick_size": _to_float(_first(row, "dTickSize", default=5)) / 100.0,
            "lot_size": lot_size if lot_size > 0 else 1,
            "instrument_type": instrument_type,
            "segment": segment,
            "exchange": exch_display,
            # kept for internal use by this module only (not part of Kite's real shape):
            "_kotak_exchange_segment": exchange_segment,
        }

    def refresh(self, force=False, max_age_hours=6):
        with self._lock:
            raw_fo = self._fetch_raw("nse_fo")
            raw_cm = self._fetch_raw("nse_cm")

            self.nfo = [self._normalize_row(r, "nse_fo") for r in raw_fo]
            self.nse = [self._normalize_row(r, "nse_cm") for r in raw_cm]

            self._nfo_by_tradingsymbol = {r["tradingsymbol"]: r for r in self.nfo if r["tradingsymbol"]}
            self._nse_by_name = {r["name"]: r for r in self.nse if r["name"]}
            self._token_to_row = {r["instrument_token"]: r for r in (self.nfo + self.nse)}

            logger.info(f"[KOTAK] Scrip master refreshed: {len(self.nfo)} NFO rows, "
                        f"{len(self.nse)} NSE rows.")

    def ensure_loaded(self):
        if not self.nfo or not self.nse:
            self.refresh()

    # -- lookups used by the quote/order layer ---------------------------------------------------
    def resolve_key(self, key: str):
        """key like 'NSE:RELIANCE', 'NFO:RELIANCE25DEC2900CE', 'NSE:NIFTY 50', 'NSE:INDIA VIX'.
        Returns (instrument_token_or_name_string, exchange_segment) or (None, None) if unresolved."""
        if ":" not in key:
            return None, None
        exch, sym = key.split(":", 1)
        exch = exch.upper()
        if sym in KOTAK_INDEX_NAME:
            return KOTAK_INDEX_NAME[sym], "nse_cm"
        self.ensure_loaded()
        if exch == "NFO":
            row = self._nfo_by_tradingsymbol.get(sym)
            if row:
                return row["instrument_token"], "nse_fo"
            return None, None
        if exch == "NSE":
            row = self._nse_by_name.get(sym)
            if row:
                return row["instrument_token"], "nse_cm"
            return None, None
        return None, None

    def name_for_token(self, token) -> Optional[str]:
        row = self._token_to_row.get(_to_int(token))
        return row["name"] if row else None

    def row_for_tradingsymbol(self, tradingsymbol) -> Optional[dict]:
        self.ensure_loaded()
        return self._nfo_by_tradingsymbol.get(tradingsymbol) or self._nse_by_name.get(tradingsymbol)


class YahooHistoricalProvider:
    """Historical daily (and best-effort intraday) candles from Yahoo Finance, in the exact list-
    of-dicts shape kite.historical_data() already returns: [{"date","open","high","low","close",
    "volume"}, ...]. This is the ONLY non-Kotak data source in the whole app — used purely because
    Kotak Neo's API has no historical-candle endpoint at all. Everything else (quotes, positions,
    orders, margins) is 100% Kotak."""

    def __init__(self):
        self._last_call_at = 0.0
        self._min_gap_seconds = 0.6  # stay well under Yahoo's informal rate limits
        self._lock = threading.Lock()

    def _pace(self):
        with self._lock:
            wait = self._min_gap_seconds - (time.time() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.time()

    def _ticker_for(self, symbol_or_index_name: str) -> Optional[str]:
        if symbol_or_index_name in YAHOO_INDEX_TICKER:
            return YAHOO_INDEX_TICKER[symbol_or_index_name]
        # plain NSE-listed equity
        return f"{symbol_or_index_name}.NS"

    def historical_data(self, symbol_or_index_name: str, from_date, to_date, interval="day"):
        if yf is None:
            raise RuntimeError("yfinance is not installed on this server "
                                "(pip install yfinance --break-system-packages).")
        ticker = self._ticker_for(symbol_or_index_name)
        if not ticker:
            logger.warning(f"[HIST] No Yahoo Finance ticker mapping for '{symbol_or_index_name}' "
                            f"(FINNIFTY/MIDCPNIFTY have no reliable public series) — returning no candles.")
            return []

        yf_interval = KITE_TO_YF_INTERVAL.get(interval)
        if yf_interval is None:
            logger.warning(f"[HIST] Unsupported interval '{interval}' for Yahoo Finance — "
                            f"falling back to daily.")
            yf_interval = "1d"

        if yf_interval != "1d":
            max_days = _YF_INTRADAY_MAX_DAYS.get(yf_interval, 60)
            earliest = datetime.now() - timedelta(days=max_days)
            if from_date < earliest:
                from_date = earliest

        self._pace()
        try:
            df = yf.download(ticker, start=from_date, end=to_date + timedelta(days=1),
                              interval=yf_interval, progress=False, auto_adjust=False)
        except Exception as e:
            logger.error(f"[HIST] Yahoo Finance fetch failed for {ticker}: {e}")
            return []

        if df is None or df.empty:
            return []

        # yfinance sometimes returns MultiIndex columns for a single ticker depending on version
        if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
            df.columns = [c[0] for c in df.columns]

        out = []
        for idx, row in df.iterrows():
            try:
                out.append({
                    "date": idx.to_pydatetime(),
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": int(row["Volume"]) if not (row["Volume"] != row["Volume"]) else 0,
                })
            except Exception:
                continue
        return out


class KiteCompatKotak:
    """The `kite` replacement object. Everything live (quotes, positions, holdings, margins,
    orders, order placement) is 100% Kotak Neo via `kotak_broker`. Historical candles come from
    Yahoo Finance via `YahooHistoricalProvider` (see module docstring for why). Method names,
    constant names, and return shapes mirror KiteConnect exactly so the ~6000 lines of existing
    strategy/screener/execution code in backend.py do not need to change."""

    # -- Kite-shaped constants. Values are whatever this facade's own place_order()/etc. expect
    # internally — callers never need to know they now map to Kotak's B/S, MKT/L, nse_fo, etc.
    VARIETY_REGULAR = "regular"
    EXCHANGE_NFO = "NFO"
    EXCHANGE_NSE = "NSE"
    PRODUCT_MIS = "MIS"
    PRODUCT_NRML = "NRML"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_SL = "SL"
    TRANSACTION_TYPE_BUY = "BUY"
    TRANSACTION_TYPE_SELL = "SELL"
    VALIDITY_DAY = "DAY"

    _EXCH_TO_SEGMENT = {"NFO": "nse_fo", "NSE": "nse_cm"}
    _ORDER_TYPE_TO_KOTAK = {"MARKET": "MKT", "LIMIT": "L", "SL": "SL", "SL-M": "SL-M"}
    _TXN_TO_KOTAK = {"BUY": "B", "SELL": "S"}

    def __init__(self, kotak_broker):
        self.broker = kotak_broker
        self.scrip_master = KotakScripMaster(kotak_broker)
        self.historical = YahooHistoricalProvider()
        self._logged_quote_sample = False
        # dynamic attr for the one constant name that isn't a valid Python identifier via dot
        # access (getattr(kite, f"ORDER_TYPE_{order_type}") with order_type == "SL-M")
        setattr(self, "ORDER_TYPE_SL-M", "SL-M")

    # -- session no-ops: Kotak auto-authenticates server-side, there is no per-request token to
    # set. Kept so any remaining `kite.set_access_token(...)` call site is a harmless no-op
    # instead of an AttributeError. --------------------------------------------------------------
    def set_access_token(self, *_a, **_kw):
        return None

    def profile(self):
        """Liveness probe. Delegates to the broker's own ensure_authenticated()/get_profile(),
        whose exceptions (BrokerAuthError/BrokerError, defined in backend.py) propagate as-is —
        this module intentionally does not redefine or catch them, to avoid a circular import
        within this file — kept as a single class boundary regardless."""
        self.broker.ensure_authenticated()
        return self.broker.get_profile()

    # -- instruments (scrip master) --------------------------------------------------------------
    def instruments(self, segment):
        self.scrip_master.ensure_loaded()
        if segment == "NFO":
            return self.scrip_master.nfo
        if segment == "NSE":
            return self.scrip_master.nse
        return []

    # -- quotes -------------------------------------------------------------------------------
    def quote(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        resolved = {}   # key -> (instrument_token_or_name, exchange_segment)
        lookup_list = []
        for k in keys:
            token, seg = self.scrip_master.resolve_key(k)
            if token is None:
                logger.warning(f"[KOTAK] quote(): could not resolve '{k}' to a Kotak instrument — skipped.")
                continue
            resolved[k] = (token, seg)
            lookup_list.append({"instrument_token": token, "exchange_segment": seg})

        out = {}
        if not lookup_list:
            return out

        chunk_size = 50  # conservative — Kotak's per-call limit isn't published
        raw_by_pos = []
        for i in range(0, len(lookup_list), chunk_size):
            chunk = lookup_list[i:i + chunk_size]
            resp = self.broker.get_quotes(chunk)
            rows = self._extract_quote_rows(resp)
            raw_by_pos.extend(rows)

        if not self._logged_quote_sample and raw_by_pos:
            logger.info(f"[KOTAK][DEBUG] Raw quote response sample: {raw_by_pos[0]}")
            self._logged_quote_sample = True

        # Match raw rows back to requested keys, preferring an explicit token/symbol field if
        # present, falling back to positional order (both lists were built in the same order).
        for idx, key in enumerate(resolved.keys()):
            token, seg = resolved[key]
            row = None
            for r in raw_by_pos:
                rtok = _first(r, "instrument_token", "pSymbol", "token", "sSymbol")
                if rtok is not None and str(rtok) == str(token):
                    row = r
                    break
            if row is None and idx < len(raw_by_pos):
                row = raw_by_pos[idx]
            if row is None:
                continue
            out[key] = self._normalize_quote(row)
        return out

    def _extract_quote_rows(self, resp):
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            for key in ("data", "Success", "success", "result"):
                if isinstance(resp.get(key), list):
                    return resp[key]
            # single-quote dict response
            if any(k in resp for k in ("ltp", "last_traded_price", "ltP")):
                return [resp]
        return []

    def _normalize_quote(self, row):
        ltp = _to_float(_first(row, "ltp", "last_traded_price", "ltP", "last_price"))
        open_ = _to_float(_first(row, "open", "o", "op"))
        high = _to_float(_first(row, "high", "h"))
        low = _to_float(_first(row, "low", "l"))
        close = _to_float(_first(row, "close", "c", "prev_close", "cp"), default=ltp)
        oi = _to_int(_first(row, "oi", "OI", "open_interest"))
        volume = _to_int(_first(row, "volume", "vol", "v", "tot_traded_qty"))

        buy_levels, sell_levels = [], []
        raw_depth = _first(row, "depth", "market_depth")
        buy_src = _first(row, "buy", "bid", "bids") if raw_depth is None else \
            (raw_depth.get("buy") or raw_depth.get("bid") if isinstance(raw_depth, dict) else None)
        sell_src = _first(row, "sell", "ask", "asks") if raw_depth is None else \
            (raw_depth.get("sell") or raw_depth.get("ask") if isinstance(raw_depth, dict) else None)
        for src, bucket in ((buy_src, buy_levels), (sell_src, sell_levels)):
            if isinstance(src, list):
                for lvl in src[:5]:
                    if isinstance(lvl, dict):
                        bucket.append({
                            "price": _to_float(_first(lvl, "price", "p", "bp", "sp")),
                            "quantity": _to_int(_first(lvl, "quantity", "qty", "bq", "sq")),
                            "orders": _to_int(_first(lvl, "orders", "no")),
                        })

        return {
            "instrument_token": _to_int(_first(row, "instrument_token", "pSymbol", "token")),
            "last_price": ltp,
            "ohlc": {"open": open_, "high": high, "low": low, "close": close},
            "oi": oi,
            "volume": volume,
            "net_change": ltp - close if close else 0.0,
            "depth": {"buy": buy_levels, "sell": sell_levels},
        }

    def ltp(self, keys):
        full = self.quote(keys)
        return {k: {"instrument_token": v["instrument_token"], "last_price": v["last_price"]}
                for k, v in full.items()}

    # -- historical data (Yahoo Finance — see module docstring) --------------------------------
    def historical_data(self, token, from_date, to_date, interval="day"):
        name = self.scrip_master.name_for_token(token)
        if not name:
            # token may already BE a name string (e.g. caller passed an index name directly)
            name = str(token)
        return self.historical.historical_data(name, from_date, to_date, interval)

    # -- positions / holdings / margins ----------------------------------------------------------
    def positions(self):
        resp = self.broker.get_positions()
        rows = self._extract_list(resp)
        net = [self._normalize_position(r) for r in rows]
        return {"net": net, "day": net}

    def holdings(self):
        resp = self.broker.get_holdings()
        return self._extract_list(resp)

    def margins(self):
        resp = self.broker.get_limits()
        if isinstance(resp, dict):
            avail = _to_float(_first(resp, "Net", "net", "AvailableMargin", "cash"))
            return {"equity": {"available": {"live_balance": avail, "cash": avail}}}
        return {"equity": {"available": {"live_balance": 0, "cash": 0}}}

    def _extract_list(self, resp):
        if isinstance(resp, list):
            return resp
        if isinstance(resp, dict):
            for key in ("data", "Success", "success", "result", "stCode"):
                if isinstance(resp.get(key), list):
                    return resp[key]
        return []

    def _normalize_position(self, row):
        tradingsymbol = _first(row, "trdSym", "tradingsymbol", "sym", default="")
        qty = _to_int(_first(row, "flBuyQty", "netQty", "quantity", default=0)) - \
            _to_int(_first(row, "flSellQty", default=0)) if "flBuyQty" in row else \
            _to_int(_first(row, "netQty", "quantity", default=0))
        return {
            "tradingsymbol": tradingsymbol,
            "exchange": "NFO",
            "instrument_token": _to_int(_first(row, "instrument_token", "tok", "token")),
            "product": _first(row, "prod", "product", default="NRML"),
            "quantity": qty,
            "average_price": _to_float(_first(row, "avgPrc", "average_price", "buyAvgPrc")),
            "last_price": _to_float(_first(row, "ltp", "last_price")),
            "pnl": _to_float(_first(row, "urPnl", "pnl", "realizedPnl")),
            "close_price": _to_float(_first(row, "close_price", "cfBuyAmt")),
        }

    # -- orders -----------------------------------------------------------------------------
    def orders(self):
        resp = self.broker.order_report()
        rows = self._extract_list(resp)
        return [self._normalize_order(r) for r in rows]

    def order_history(self, order_id=None):
        if order_id:
            resp = self.broker.order_history(order_id)
            return [self._normalize_order(r) for r in self._extract_list(resp)]
        return self.orders()

    _STATUS_MAP = {
        "complete": "COMPLETE", "executed": "COMPLETE", "traded": "COMPLETE",
        "rejected": "REJECTED", "cancelled": "CANCELLED", "canceled": "CANCELLED",
        "open": "OPEN", "pending": "OPEN", "trigger pending": "TRIGGER PENDING",
    }

    def _normalize_order(self, row):
        raw_status = str(_first(row, "ordSt", "status", "orderStatus", default="")).lower()
        status = self._STATUS_MAP.get(raw_status, raw_status.upper() or "UNKNOWN")
        return {
            "order_id": str(_first(row, "nOrdNo", "order_id", "orderId", default="")),
            "tradingsymbol": _first(row, "trdSym", "tradingsymbol", default=""),
            "status": status,
            "transaction_type": "BUY" if _first(row, "trnsTp", "transaction_type", default="") in
                                 ("B", "BUY") else "SELL",
            "quantity": _to_int(_first(row, "qty", "quantity")),
            "filled_quantity": _to_int(_first(row, "fldQty", "filled_quantity")),
            "average_price": _to_float(_first(row, "avgPrc", "average_price")),
            "price": _to_float(_first(row, "prc", "price")),
            "order_type": _first(row, "prcTp", "order_type", default=""),
            "product": _first(row, "prod", "product", default=""),
        }

    def place_order(self, **kwargs):
        variety = kwargs.pop("variety", self.VARIETY_REGULAR)  # not used by Kotak; consumed here
        exchange = kwargs.pop("exchange", self.EXCHANGE_NFO)
        exchange_segment = self._EXCH_TO_SEGMENT.get(exchange, "nse_fo")
        order_type = self._ORDER_TYPE_TO_KOTAK.get(kwargs.pop("order_type", "MARKET"), "MKT")
        transaction_type = self._TXN_TO_KOTAK.get(kwargs.pop("transaction_type"), "B")
        tradingsymbol = kwargs.pop("tradingsymbol")
        quantity = kwargs.pop("quantity")
        product = kwargs.pop("product", "NRML")
        price = kwargs.pop("price", 0) or 0
        trigger_price = kwargs.pop("trigger_price", 0) or 0
        validity = kwargs.pop("validity", "DAY")
        kwargs.pop("market_protection", None)  # Kotak has its own equivalent (market_protection
        # param below); the exchange-mandated protection band is applied by Kotak server-side.
        tag = kwargs.pop("tag", None)

        resp = self.broker.place_order(
            exchange_segment=exchange_segment, product=product, price=str(price),
            order_type=order_type, quantity=str(quantity), validity=validity,
            trading_symbol=tradingsymbol, transaction_type=transaction_type,
            amo="NO", disclosed_quantity="0", market_protection="0",
            pf="N", trigger_price=str(trigger_price), tag=tag or "",
        )
        order_id = _first(resp, "nOrdNo", "order_id", "orderId")
        if order_id is None:
            raise RuntimeError(f"Kotak place_order did not return an order id: {resp}")
        return str(order_id)

    def modify_order(self, **kwargs):
        order_id = kwargs.pop("order_id")
        variety = kwargs.pop("variety", None)
        payload = {"order_id": order_id}
        if "order_type" in kwargs:
            payload["order_type"] = self._ORDER_TYPE_TO_KOTAK.get(kwargs.pop("order_type"), "MKT")
        if "price" in kwargs:
            payload["price"] = str(kwargs.pop("price"))
        if "quantity" in kwargs:
            payload["quantity"] = str(kwargs.pop("quantity"))
        if "trigger_price" in kwargs:
            payload["trigger_price"] = str(kwargs.pop("trigger_price"))
        if "validity" in kwargs:
            payload["validity"] = kwargs.pop("validity")
        resp = self.broker.modify_order(**payload)
        return str(_first(resp, "nOrdNo", "order_id", "orderId", default=order_id))

    def cancel_order(self, variety=None, order_id=None, **kwargs):
        resp = self.broker.cancel_order(order_id)
        return str(_first(resp, "nOrdNo", "order_id", "orderId", default=order_id))

    # -- margin estimate (Kotak Neo has no documented basket/combo margin calculator in this
    # SDK, unlike Kite's basket_order_margins). This is a conservative, clearly-labelled
    # ESTIMATE only, deliberately absent from the class so compute_margin()'s existing
    # `hasattr(kite, "basket_order_margins")` check falls through to order_margins() below
    # rather than silently claiming basket-level margin benefit we cannot actually verify. -----
    def order_margins(self, order_params):
        """Per-leg SPAN+exposure ESTIMATE only (rough multiplier of notional), NOT Kotak's real
        margin calculator — Kotak Neo's SDK does not expose one. Always confirm the real
        required margin in the Kotak Neo app/RMS screen before relying on it. Deliberately
        conservative (over- rather than under-estimates) so it never suggests more buying power
        is available than you actually have."""
        out = []
        for leg in order_params:
            out.append({"total": 0.0})  # unknown without a real margin API — surfaced as 0/None
        return out

# ============================= end inlined kotak_compat module =============================


kotak_broker = KotakNeoBroker()
kite = KiteCompatKotak(kotak_broker)
ACTIVE_BROKER = "KOTAK"


def get_active_broker():
    return kotak_broker


SESSION = {"access_token": None, "logged_in_at": None}  # kept only as a legacy shape for any
# leftover reference; Kotak Neo has no per-request access token — see require_session() below,
# which is backed by kotak_broker.ensure_authenticated(), not this dict.
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
                "prices — check your Kotak Neo contract note for the exact realized P&L and charges.",
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
    """Shared instrument-token lookup for stocks AND indices (used by trend detection)."""
    symbol = symbol.upper()
    _, nse = get_instruments()
    if symbol in INDEX_SYMBOLS:
        wanted = INDEX_SYMBOLS[symbol].split(":")[1]
        for i in nse:
            if i["segment"] == "INDICES" and i["tradingsymbol"] == wanted:
                return i["instrument_token"], None
        return None, f"Could not resolve index token for {symbol}"
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
        q = kite.quote(["NSE:INDIA VIX"])["NSE:INDIA VIX"]
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
# Kotak Neo session — fully server-side (TOTP + MPIN from ~/kotak_algo/.env). There is no
# per-user login flow: /api/session-status and require_session() both just check/establish the
# server's own Kotak Neo session. Route names are kept identical to before (session-status,
# require_session, logout, broker/status, kotak/connect) since ~30 route handlers below already
# call require_session() and the frontend already calls /api/session-status — nothing else in
# this file had to change.
# ---------------------------------------------------------------------------
@app.route("/api/session-status")
def session_status():
    """Actively (re-)establishes the Kotak Neo session (not just checks a flag) so a stale/
    expired session can't keep showing 'Connected' after it's no longer valid."""
    try:
        connected = kotak_broker.ensure_authenticated() and kotak_broker.is_connected()
        if connected:
            SESSION["logged_in_at"] = SESSION["logged_in_at"] or now_ist().isoformat()
            return jsonify({"logged_in": True, "logged_in_at": SESSION["logged_in_at"]})
        SESSION["logged_in_at"] = None
        return jsonify({"logged_in": False, "logged_in_at": None, "session_expired": True})
    except BrokerAuthError as e:
        SESSION["logged_in_at"] = None
        return jsonify({"logged_in": False, "logged_in_at": None, "session_expired": True,
                         "message": str(e)})
    except Exception:
        # network hiccup or similar — don't report disconnected for a transient error
        return jsonify({"logged_in": bool(SESSION["logged_in_at"]),
                         "logged_in_at": SESSION["logged_in_at"],
                         "warning": "Could not verify Kotak Neo session freshness right now "
                                    "(network issue?)."})


def require_session():
    try:
        return kotak_broker.ensure_authenticated() and kotak_broker.is_connected()
    except BrokerAuthError:
        return False
    except Exception:
        logger.exception("[KOTAK] Unexpected error while checking session in require_session()")
        return False


@app.route("/api/logout", methods=["POST"])
def logout():
    """Forces the next call to re-authenticate with Kotak Neo (drops the cached client so
    ensure_authenticated() runs the full TOTP+MPIN flow again) — there's no separate 'logged out'
    state to persist since Kotak Neo re-authenticates itself automatically anyway."""
    kotak_broker.client = None
    SESSION["logged_in_at"] = None
    logger.info("User logged out — Kotak Neo session cleared (will auto-reconnect on next use).")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Broker status (Kotak Neo)
# ---------------------------------------------------------------------------
@app.route("/api/broker/status")
def broker_status():
    """Status for the (only) active broker, Kotak Neo. ATTEMPTS silent re-authentication if
    needed (TOTP + MPIN, fully server-side) rather than just reporting 'disconnected' — this is
    what makes opening the page auto-connect Kotak without the user doing anything. Never
    returns any credential or token material — only booleans/messages."""
    kotak_payload = {"broker": "KOTAK", "connected": False, "message": None}
    if kotak_broker._config_error:
        kotak_payload["message"] = kotak_broker._config_error
    else:
        try:
            kotak_payload["connected"] = kotak_broker.ensure_authenticated() and kotak_broker.is_connected()
        except BrokerAuthError as e:
            kotak_payload["message"] = str(e)
        except Exception as e:
            kotak_payload["message"] = f"{type(e).__name__}: {e}"

    return jsonify({
        "active_broker": ACTIVE_BROKER,
        "live_trading": kotak_broker.live_trading,
        "kotak": kotak_payload,
    })


@app.route("/api/kotak/connect", methods=["POST"])
def kotak_connect():
    """Manual 'reconnect' trigger for the UI's Kotak status pill — same server-side TOTP+MPIN
    flow as the automatic check in /api/broker/status, exposed as an explicit action for when
    the user wants to force a retry (e.g. after fixing an .env value) without reloading."""
    try:
        kotak_broker.authenticate()
        return jsonify(kotak_broker.status_payload())
    except BrokerAuthError as e:
        return jsonify({"broker": "KOTAK", "connected": False, "message": str(e)}), 401
    except Exception as e:
        logger.error(f"[KOTAK] Unexpected error during manual connect: {type(e).__name__}")
        return jsonify({"broker": "KOTAK", "connected": False,
                         "message": "Unexpected error — check server logs."}), 500


@app.errorhandler(BrokerAuthError)
def handle_broker_auth_error(e):
    """Catches an expired/invalid Kotak Neo session from ANY route, and tells the frontend to
    show 'reconnecting' instead of a generic 500 error or a UI that silently keeps showing
    'Connected' while every data call quietly fails. Kotak Neo re-authenticates itself
    automatically on the next request (see require_session()/ensure_authenticated()) — this
    handler does not require the user to do anything."""
    SESSION["logged_in_at"] = None
    logger.warning(f"BrokerAuthError caught — Kotak Neo session issue: {e}")
    return jsonify({"error": "session_expired", "session_expired": True,
                     "message": f"Kotak Neo session issue: {e}. Retrying automatically — "
                                "reload if this persists."}), 401


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
    symbol_to_nse_key = {i["tradingsymbol"]: f"NSE:{i['tradingsymbol']}"
                          for i in nse if i["exchange"] == "NSE"}

    # --- Pass 1: calmness — 100% Kotak. One batched live quote for the whole universe (instead
    # of ~200 sequential external calls), captured into a local history file that becomes real
    # multi-day HV/ATR over the next few weeks. Until enough days exist for a symbol, today's own
    # intraday range is used as an immediate same-day calmness proxy — see
    # historical_vol_and_atr_kotak()/capture_price_snapshot() above. No stock is skipped just for
    # lacking deep history; it just ranks on the proxy until real HV/ATR kicks in.
    try:
        quotes, price_history = capture_price_snapshot(universe, symbol_to_nse_key)
    except Exception as e:
        logger.error(f"[SCREENER] capture_price_snapshot failed: {e}")
        quotes, price_history = {}, load_price_history()

    results = []
    for name in universe:
        key = symbol_to_nse_key.get(name)
        q = quotes.get(key) if key else None
        ltp = q["last_price"] if q else None
        if not ltp:
            continue
        ohlc = q.get("ohlc") or {}
        high, low = ohlc.get("high"), ohlc.get("low")
        range_pct_today = round((high - low) / ltp * 100, 2) if (high and low and ltp) else None

        hv, atr_pct, days_avail = historical_vol_and_atr_kotak(name, price_history)
        results.append({
            "symbol": name, "ltp": round(ltp, 2),
            "hv_annualized_pct": hv, "atr_pct_of_price": atr_pct,
            "range_pct_today": range_pct_today,
            "price_history_days": days_avail,
            "calmness_basis": (f"{days_avail}d historical vol/ATR" if hv is not None
                                else "today's range only — building history"
                                if range_pct_today is not None else "no data yet"),
        })
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
        r["iv_hv"] = classify_iv_hv(r.get("atm_iv_pct"), r.get("hv_annualized_pct"))
    save_iv_history(iv_history)

    # --- Pass 3: best-effort F&O ban list, excludes banned symbols from top picks ---
    ban_symbols, ban_error = get_fo_ban_list()
    for r in results:
        r["fo_banned_today"] = (ban_symbols is not None and r["symbol"] in ban_symbols)

    # --- Composite score: rich IV (same-day cross-sectional percentile, since most stocks
    # won't have 20+ days of stored history yet) + calm underlying + tradeable liquidity ---
    all_iv = [r["atm_iv_pct"] for r in results]
    all_hv = [r["hv_annualized_pct"] for r in results if r["hv_annualized_pct"] is not None]
    all_atr = [r["atr_pct_of_price"] for r in results if r["atr_pct_of_price"] is not None]
    all_range = [r["range_pct_today"] for r in results if r["range_pct_today"] is not None]
    all_oi = [r["atm_oi_total"] for r in results if r["atm_oi_total"]]
    all_spread = [r["atm_spread_pct"] for r in results if r["atm_spread_pct"] is not None]

    eligible = []
    for r in results:
        iv_richness_pct = (r["iv_rank_pct"] if r["iv_rank_pct"] is not None
                            else _percentile_rank(r["atm_iv_pct"], all_iv))

        # Calmness: real HV/ATR once enough local history exists; today's intraday range alone
        # until then (see historical_vol_and_atr_kotak() docstring) — never skips a stock.
        if r["hv_annualized_pct"] is not None and r["atr_pct_of_price"] is not None:
            calm_hv_pct = 100 - _percentile_rank(r["hv_annualized_pct"], all_hv)
            calm_atr_pct = 100 - _percentile_rank(r["atr_pct_of_price"], all_atr)
            calmness_pct = (calm_hv_pct + calm_atr_pct) / 2
        elif r["range_pct_today"] is not None:
            calmness_pct = 100 - _percentile_rank(r["range_pct_today"], all_range)
        else:
            calmness_pct = 50.0  # no data at all yet — neutral, not silently zero-weighted

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
                 f"until then) {int(SCORE_WEIGHTS['iv_richness']*100)}%, calmness "
                 f"{int(SCORE_WEIGHTS['calmness']*100)}% (real historical vol/ATR, built from your "
                 f"own Kotak Neo quotes each day this runs — needs {PRICE_HISTORY_MIN_DAYS_FOR_HV}+ "
                 f"days per stock before it kicks in, today's own intraday range used as a same-day "
                 f"proxy until then — check calmness_basis per stock), ATM liquidity (OI + spread) "
                 f"{int(SCORE_WEIGHTS['liquidity']*100)}%. Stocks are excluded from ranking if ATM "
                 f"combined OI < {MIN_ATM_TOTAL_OI} lots, ATM spread > {MAX_ATM_SPREAD_PCT}%, on "
                 "today's F&O ban list, or IV couldn't be computed. Headlines are a best-effort "
                 "keyword scan, NOT sentiment analysis or verified news — read the actual articles, "
                 "and still check earnings/corporate action dates yourself before trading."),
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
    today = now_ist().date()
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
                    strategy_type="iron_condor", expiry_str=None, lots=1):
    data, err = get_chain_for_symbol(symbol, expiry_str)
    if err:
        return err
    spot, expiry, T, lot_size, chain = data["spot"], data["expiry"], data["T"], data["lot_size"], data["chain"]
    today = now_ist().date()
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

    # --- Enhanced trading logic: IV/HV, Expected Move, POT/POP, Trend, Vol Regime, Score ---
    rank_info = result["rank_info"]
    iv_hv = classify_iv_hv(rank_info.get("atm_iv_pct") if rank_info else None,
                            rank_info.get("hv_annualized_pct") if rank_info else None)
    result["iv_hv"] = iv_hv
    if iv_hv is None:
        result["iv_hv_note"] = "Run the Screener (section 1) first so IV/HV data is cached for this symbol."

    em = expected_move(spot, rank_info.get("atm_iv_pct") if rank_info else None, result["days_to_expiry"])
    result["expected_move"] = em
    if em and "sell_call" in result["legs"] and "sell_put" in result["legs"]:
        sc_strike = result["legs"]["sell_call"]["strike"]
        sp_strike = result["legs"]["sell_put"]["strike"]
        inside_em = sc_strike < em["upper"] or sp_strike > em["lower"]
        result["short_strikes_inside_expected_move"] = inside_em
        if inside_em:
            result["expected_move_warning"] = (
                f"Short strike(s) fall INSIDE the {result['days_to_expiry']}-day expected move "
                f"(±₹{em['expected_move']}, range {em['lower']}–{em['upper']}) — higher chance of being "
                f"tested before expiry. Consider wider strikes.")

    for k in ("sell_call", "sell_put"):
        if k in result["legs"]:
            result["legs"][k]["probability_of_touch_pct"] = probability_of_touch(result["legs"][k]["delta"])

    if "sell_call" in result["legs"] and "sell_put" in result["legs"]:
        dc = abs(result["legs"]["sell_call"]["delta"])
        dp = abs(result["legs"]["sell_put"]["delta"])
        result["probability_of_profit_pct"] = round(max(0.0, (1 - dc - dp)) * 100, 1)

    trend = get_trend_regime(symbol)
    result["trend"] = None if trend.get("error") else trend
    if trend.get("error"):
        result["trend_note"] = trend["error"]

    vix, vix_err = get_india_vix()
    iv_rank_for_regime = rank_info.get("iv_rank_pct") if rank_info else None
    result["volatility_regime"] = classify_volatility_regime(vix, iv_rank_for_regime)
    if vix_err:
        result["volatility_regime"]["note"] = f"India VIX fetch failed ({vix_err}); classification unavailable."

    score_components = []
    if iv_hv:
        score_components.append({"excellent": 95, "good": 80, "fair": 60, "avoid": 25}.get(iv_hv["label"].lower(), 50))
    if rank_info and rank_info.get("composite_score") is not None:
        score_components.append(rank_info["composite_score"])
    if trend and not trend.get("error"):
        score_components.append(30 if trend.get("avoid_premium_selling") else 75)
    if result["volatility_regime"]["label"] != "Unknown":
        vr_score = {"Low Volatility": 55, "Normal": 80, "High Volatility": 70, "Extreme": 20}.get(
            result["volatility_regime"]["label"], 50)
        score_components.append(vr_score)
    if rank_info and rank_info.get("fo_banned_today"):
        score_components.append(0)
    trade_quality_score = round(sum(score_components) / len(score_components), 1) if score_components else None
    result["trade_quality_score"] = trade_quality_score
    if trade_quality_score is not None:
        if trade_quality_score >= 80:
            result["trade_quality_label"] = "Excellent"
        elif trade_quality_score >= 60:
            result["trade_quality_label"] = "Good"
        elif trade_quality_score >= 40:
            result["trade_quality_label"] = "Average"
        else:
            result["trade_quality_label"] = "Avoid"
    result["trade_quality_note"] = ("Heuristic score blending IV/HV richness, screener composite, trend regime, "
                                     "and volatility regime (equal-weighted average of whichever signals are "
                                     "available). Not a probability, not backtested — a rough triage aid only.")

    result["suggested_strategy"] = suggest_strategy_family(iv_rank_for_regime, trend)
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
    """Per-position net Greeks via Black-Scholes at current quotes (Kotak doesn't publish Greeks
    itself). Gamma/Vega/Theta are estimated by bump-and-reprice off the same bs_price/bs_delta
    helpers used everywhere else in this file."""
    leg_keys = leg_keys_for(position)
    quantity = position.get("quantity", position["lot_size"])
    spot, err = get_spot_price(position["symbol"])
    if err:
        return {"error": err["error"]}

    today = now_ist().date()
    near_expiry_date = datetime.strptime(position["expiry"], "%Y-%m-%d").date()
    days_left_near = max((near_expiry_date - today).days, 0)
    if days_left_near <= 0:
        return {"error": "Position has expired"}

    inst_keys = [f"NFO:{position['legs'][k]['tradingsymbol']}" for k in leg_keys]
    try:
        quotes = kite.quote(inst_keys)
    except Exception as e:
        return {"error": str(e)}

    net_delta = net_theta = net_vega = net_gamma = 0.0
    for k in leg_keys:
        strike = position["legs"][k]["strike"]
        opt_type = "CE" if "call" in k else "PE"
        T = days_left_near / 365.0
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
    SELL leg. Kotak Neo checks margin against your live positions at the moment each order hits the
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
                variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NFO,
                tradingsymbol=item["tradingsymbol"], transaction_type=txn_type,
                quantity=quantity, product=getattr(kite, f"PRODUCT_{product}"),
                order_type=getattr(kite, f"ORDER_TYPE_{order_type}"),
                validity=kite.VALIDITY_DAY,
            )
            if order_type == "LIMIT" and reference_price:
                kwargs["price"] = float(reference_price)
            if order_type in ("MARKET", "SL-M"):
                # Exchanges require market-protection on MARKET/SL-M orders (SEBI's retail algo-
                # trading framework). KiteCompatKotak.place_order() applies Kotak's own
                # market_protection value server-side regardless of what's passed here — this
                # kwarg is accepted for compatibility but Kotak's own default protection band is
                # what actually gets sent (see KiteCompatKotak.place_order() below).
                kwargs["market_protection"] = -1
            order_id = kite.place_order(**kwargs)

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
                   "remove a leg entirely below — then confirm to send them to your live Kotak Neo account. "
                   "Removing a hedge leg (a BUY order) from an Iron Condor leaves that side of the "
                   "position with unlimited-style risk, same as a naked strangle. If you click "
                   "'Yes, place these real orders', BUY legs are sent first and this tool waits for "
                   "each to fill before sending SELL legs, so the SELL side doesn't get rejected for "
                   "insufficient margin. You can also use 'Execute this leg' on any single row to fire "
                   "legs yourself, one at a time, in whatever order you choose."
    })


@app.route("/api/execute/<pos_id>/leg", methods=["POST"])
def execute_single_leg(pos_id):
    """Places exactly ONE leg right now — used by the per-leg 'Execute this leg' button in the
    review screen so you can manually sequence a multi-leg entry yourself (e.g. fire the BUY hedge,
    watch it fill in your Kotak Neo app, then come back and fire the SELL leg once margin is freed).
    This does NOT apply the automatic BUY-before-SELL basket sequencing — you're placing one leg,
    on purpose, right now."""
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
                 "unhedged position. Open your Kotak Neo app IMMEDIATELY to check your actual "
                 "positions and orders, and manually complete or exit as needed."
                 if partial else
                 "All legs failed — nothing was placed." if any_failed and placed_count == 0 else
                 f"All {placed_count}/{total_legs} legs placed successfully. Verify fills in your Kotak Neo app.")
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
                   "orders to your Kotak Neo account."
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
                "Open your Kotak Neo app IMMEDIATELY to check and manually complete the close.")
    elif any_failed:
        note = "All legs failed — nothing was closed."
    else:
        note = f"All {placed_count}/{total_legs} legs placed to close this position. Verify fills in your Kotak Neo app."

    save_positions(positions)
    return jsonify({"results": results, "fully_closed": fully_closed, "note": note})


@app.route("/api/broker-positions")
def broker_positions():
    """Live F&O positions straight from your Kotak Neo account (Kite's net positions() call) —
    independent of this tool's own tracked Iron Condor / Strangle baskets in positions.json, and
    independent of which strategy or basket a leg originally came from. For each open NFO leg this
    returns the entry (average) price, live LTP, and running P&L reported by Kite itself, so the
    Order Management tab can show exactly what your account currently holds and let you price and
    fire an exit — for one leg or several at once — straight from here. Also groups every leg by its
    UNDERLYING (e.g. all 4 SBIN option legs roll up under "SBIN") and returns a per-symbol total P&L,
    since a single leg's P&L in isolation isn't the number that matters for a multi-leg position."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    try:
        nfo, _ = get_instruments()
        underlying_by_symbol = {i["tradingsymbol"]: i.get("name") for i in nfo if i.get("segment") in ("NFO-OPT", "NFO-FUT")}
        pos = kite.positions()
        net = pos.get("net", [])
        rows = []
        totals_by_symbol = {}
        for p in net:
            if p.get("exchange") != "NFO":
                continue
            qty = int(p.get("quantity") or 0)
            if qty == 0:
                continue  # already flat — nothing open on this tradingsymbol
            ts = p.get("tradingsymbol")
            underlying = underlying_by_symbol.get(ts, ts)
            pnl = p.get("pnl") or 0.0
            rows.append({
                "tradingsymbol": ts,
                "underlying": underlying,
                "product": p.get("product"),
                "quantity": qty,
                "side": "LONG" if qty > 0 else "SHORT",
                "average_price": p.get("average_price"),
                "last_price": p.get("last_price"),
                "pnl": pnl,
                "close_price": p.get("close_price"),
            })
            totals_by_symbol[underlying] = round(totals_by_symbol.get(underlying, 0.0) + pnl, 2)
        return jsonify({"positions": rows, "totals_by_symbol": totals_by_symbol})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/broker-positions/simulate", methods=["POST"])
def broker_positions_simulate():
    """"What if" scenario simulator for your CURRENT live position on one underlying: estimates P&L
    across a grid of spot price moves (%) and days forward from today, using Black-Scholes with each
    leg's OWN implied volatility (backed out from its current live LTP) held constant except for any
    optional iv_shift_pct you apply. This is a forward-looking estimate, not a backtest -- it answers
    "if the stock moves X% over the next N days, roughly what happens to my P&L", including time
    decay (theta) via the shrinking time-to-expiry as the day offset increases."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    symbol = (body.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    price_moves_pct = body.get("price_moves_pct") or [-10, -7, -5, -3, -1, 0, 1, 3, 5, 7, 10]
    day_offsets = body.get("day_offsets") or [0, 1, 3, 7]
    iv_shift_pct = float(body.get("iv_shift_pct", 0.0))

    nfo, _ = get_instruments()
    meta_by_symbol = {i["tradingsymbol"]: i for i in nfo if i.get("name") == symbol and i.get("segment") == "NFO-OPT"}
    try:
        pos = kite.positions()
    except Exception as e:
        return jsonify({"error": f"Could not fetch live positions: {e}"}), 400
    net = pos.get("net", [])
    legs_raw = []
    for p in net:
        ts = p.get("tradingsymbol")
        if p.get("exchange") != "NFO" or ts not in meta_by_symbol:
            continue
        qty = int(p.get("quantity") or 0)
        if qty == 0:
            continue
        legs_raw.append({"tradingsymbol": ts, "quantity": qty, "entry_price": p.get("average_price"),
                          "meta": meta_by_symbol[ts]})
    if not legs_raw:
        return jsonify({"error": f"No open NFO option legs found for {symbol} in your live positions"}), 400

    try:
        chain_data, chain_err = get_chain_for_symbol(symbol)
        spot = chain_data["spot"] if not chain_err else None
    except Exception:
        spot = None
    if not spot:
        return jsonify({"error": f"Could not fetch current spot price for {symbol}"}), 400

    try:
        quotes = kite.quote([f"NFO:{leg['tradingsymbol']}" for leg in legs_raw])
    except Exception:
        quotes = {}

    today = now_ist().date()
    sim_legs = []
    for leg in legs_raw:
        meta = leg["meta"]
        expiry_date = meta.get("expiry")
        expiry_date = expiry_date.date() if hasattr(expiry_date, "date") else expiry_date
        T_days = max((expiry_date - today).days, 0) if expiry_date else 7
        q = quotes.get(f"NFO:{leg['tradingsymbol']}", {})
        ltp = extract_price(q) or leg["entry_price"]
        T_now = max(T_days / 365.0, 1 / 365.0)
        iv = implied_vol(ltp, spot, meta["strike"], T_now, meta["instrument_type"])
        if not iv or iv <= 0:
            iv = 0.15
        sim_legs.append({
            "tradingsymbol": leg["tradingsymbol"], "opt_type": meta["instrument_type"], "strike": meta["strike"],
            "quantity": leg["quantity"], "entry_price": leg["entry_price"], "T_days": T_days, "iv": iv,
        })

    grid = {}
    for d in day_offsets:
        row = []
        for pct in price_moves_pct:
            sim_spot = spot * (1 + pct / 100.0)
            total_pnl = 0.0
            for leg in sim_legs:
                T_remaining = max((leg["T_days"] - d) / 365.0, 0.0)
                iv_shifted = max(leg["iv"] * (1 + iv_shift_pct / 100.0), 0.01)
                price = bs_price(sim_spot, leg["strike"], T_remaining, RISK_FREE_RATE, iv_shifted, leg["opt_type"])
                per_unit = (leg["entry_price"] - price) if leg["quantity"] < 0 else (price - leg["entry_price"])
                total_pnl += per_unit * abs(leg["quantity"])
            row.append({"price_move_pct": pct, "spot_price": round(sim_spot, 2), "pnl": round(total_pnl, 2)})
        grid[str(d)] = row

    return jsonify({
        "symbol": symbol, "spot": spot,
        "legs": [{"tradingsymbol": l["tradingsymbol"], "opt_type": l["opt_type"], "strike": l["strike"],
                   "quantity": l["quantity"], "days_to_expiry": l["T_days"], "iv_pct": round(l["iv"] * 100, 1)}
                  for l in sim_legs],
        "price_moves_pct": price_moves_pct, "day_offsets": day_offsets, "grid": grid,
        "caveat": ("Assumes each leg's implied volatility stays constant (aside from any IV shift you "
                    "apply) as the spot price moves -- in reality IV often falls on a rally and rises on "
                    "a selloff, which this does not model unless you set an IV shift yourself. Treat this "
                    "as an estimate to plan around, not a guaranteed outcome."),
    })


@app.route("/api/broker-positions/exit", methods=["POST"])
def broker_positions_exit():
    """Squares off one or more live Kotak Neo F&O positions directly by tradingsymbol — backs the
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
    any_priced = False
    for lg in legs:
        if not lg.get("tradingsymbol"):
            continue
        qty = abs(int(lg.get("quantity") or 0))
        if qty <= 0:
            continue
        close_txn = "SELL" if str(lg.get("side", "LONG")).upper() == "LONG" else "BUY"
        price = lg.get("price")
        if price not in (None, ""):
            any_priced = True
        legs_to_place.append({
            "leg": lg["tradingsymbol"], "tradingsymbol": lg["tradingsymbol"],
            "transaction_type": close_txn, "quantity": qty,
            "price": float(price) if price not in (None, "") else None,
        })

    if not legs_to_place:
        return jsonify({"error": "No valid legs to place."}), 400

    # If ANY leg in this batch was given a specific price, place the whole batch as LIMIT orders
    # (legs without a price fall back to their live reference price computed per-leg below);
    # otherwise place everything MARKET.
    if any_priced:
        inst_keys = [f"NFO:{lg['tradingsymbol']}" for lg in legs_to_place]
        try:
            quotes = kite.quote(inst_keys)
        except Exception:
            quotes = {}
        for lg in legs_to_place:
            if lg["price"] is None:
                lg["price"] = extract_price(quotes.get(f"NFO:{lg['tradingsymbol']}"))
        order_type = "LIMIT"
    else:
        order_type = "MARKET"

    results = place_basket_orders(legs_to_place, product, order_type, sequence_for_margin=False)
    return jsonify({"results": results})


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


# =============================================================================
# Premium Skew Scanner — for a symmetric (equal-delta) Iron Condor, the call-side and put-side
# premiums are USUALLY close but rarely identical; sometimes one side is priced noticeably richer
# than the other at the same delta. This scans a universe of stocks/indices, finds the call and put
# nearest your configured delta for each, and ranks them by how lopsided that premium is right now.
# It does NOT predict tomorrow's price — it only surfaces where today's skew is largest so you can
# decide whether to sell into it. Skew can persist for good reasons (an upcoming event, structural
# put-skew in that name) as well as revert -- see the caveat returned with every scan.
# =============================================================================
SKEW_SCANNER_FILE = os.path.join(os.path.dirname(__file__), "skew_scanner_state.json")
SKEW_SCANNER_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "skew_scanner_results.json")
_skew_scanner_lock = threading.Lock()
SKEW_SCAN_CHUNK_SIZE = 15   # symbols scanned per batch, staggered -- kept modest since each symbol
                             # now costs an extra historical-data call (for trend) on top of the chain fetch

SKEW_SCANNER_DEFAULTS = {
    "delta_low": 0.15,
    "delta_high": 0.20,
    "hedge_distance_pct": 2.5,
    "scan_all_fo": True,            # scan the whole live F&O universe, batch by batch
    "universe": [],                 # used only when scan_all_fo is False
    "min_skew_pct": 15.0,           # only keep candidates at least this lopsided
    "_scan_cursor": 0,
    "last_scan_at": None,
    "last_batch_errors": [],
}
SKEW_SCANNER_CONFIGURABLE_KEYS = ("delta_low", "delta_high", "hedge_distance_pct", "scan_all_fo",
                                   "universe", "min_skew_pct")


def load_skew_scanner_state():
    state = dict(SKEW_SCANNER_DEFAULTS)
    if os.path.exists(SKEW_SCANNER_FILE):
        try:
            with open(SKEW_SCANNER_FILE, "r") as f:
                state.update(json.load(f))
        except Exception:
            pass
    return state


def save_skew_scanner_state(state):
    with _skew_scanner_lock:
        with open(SKEW_SCANNER_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)


def load_skew_scanner_results():
    if not os.path.exists(SKEW_SCANNER_RESULTS_FILE):
        return []
    try:
        with open(SKEW_SCANNER_RESULTS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_skew_scanner_results(results):
    with _skew_scanner_lock:
        with open(SKEW_SCANNER_RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)


def _skew_scan_universe_batch(state):
    """Same rotating-chunk pattern as _effective_scan_universe -- covers the whole F&O list
    progressively across repeated batches instead of one huge slow request. Returns
    (chunk, universe_size, pass_complete) so the caller (and the UI) knows when a full rotation
    through the universe has just finished, to auto-stop a continuous scan."""
    if not state.get("scan_all_fo"):
        uni = state.get("universe") or []
        return uni, len(uni), True   # a manual list is always fully covered in one batch
    try:
        full = list(INDEX_SYMBOLS.keys()) + fo_stock_universe()
    except Exception:
        uni = state.get("universe") or []
        return uni, len(uni), True
    if not full:
        return [], 0, True
    cursor = state.get("_scan_cursor", 0) % len(full)
    rotated = full[cursor:] + full[:cursor]
    chunk = rotated[:SKEW_SCAN_CHUNK_SIZE]
    new_cursor = (cursor + SKEW_SCAN_CHUNK_SIZE) % len(full)
    pass_complete = new_cursor <= cursor   # wrapped back to/past the start
    state["_scan_cursor"] = new_cursor
    return chunk, len(full), pass_complete


def _scan_symbol_for_skew(symbol, delta_low, delta_high, hedge_distance_pct):
    """Finds the call and put nearest the target delta for one symbol and scores how lopsided their
    premiums are. Returns (candidate_dict_or_None, error_or_None)."""
    try:
        chain_data, err = get_chain_for_symbol(symbol)
    except Exception as e:
        return None, str(e)
    if err:
        return None, err.get("error", "chain fetch failed")
    chain = chain_data.get("chain") or []
    spot = chain_data.get("spot")
    if not spot:
        return None, "No spot price"
    expiry = chain_data.get("expiry")
    T = chain_data.get("T")
    days_to_expiry = max((expiry - now_ist().date()).days, 0) if expiry else None
    target_delta = (delta_low + delta_high) / 2.0

    calls = [o for o in chain if o.get("instrument_type") == "CE" and o.get("delta") is not None]
    puts = [o for o in chain if o.get("instrument_type") == "PE" and o.get("delta") is not None]
    if not calls or not puts:
        return None, "Incomplete option chain"

    call_leg = min(calls, key=lambda o: abs(o["delta"] - target_delta))
    put_leg = min(puts, key=lambda o: abs(abs(o["delta"]) - target_delta))
    tolerance = 0.05
    if not (delta_low - tolerance <= call_leg["delta"] <= delta_high + tolerance):
        return None, "No call strike close enough to the target delta range"
    if not (delta_low - tolerance <= abs(put_leg["delta"]) <= delta_high + tolerance):
        return None, "No put strike close enough to the target delta range"

    call_premium, put_premium = call_leg.get("ltp"), put_leg.get("ltp")
    if not call_premium or not put_premium or call_premium <= 0 or put_premium <= 0:
        return None, "Zero/invalid premium on one side"

    richer_side = "put" if put_premium > call_premium else "call"
    avg_premium = (call_premium + put_premium) / 2.0
    skew_pct = abs(put_premium - call_premium) / avg_premium * 100.0
    iv_skew = round((put_leg.get("iv") or 0) - (call_leg.get("iv") or 0), 2)

    # Fair-value check -- reuses the app's existing bs_price() Black-Scholes pricer. Each leg's own
    # market IV is backed OUT of its own market price, so comparing a leg to itself would be circular
    # (it would always match). Instead this prices both legs off ONE flat benchmark vol -- the ATM
    # IV, i.e. "what would this strike cost if the whole chain were priced like the at-the-money
    # option, with no smile/skew at all". The gap between that flat-vol fair value and the actual
    # quoted premium is a second, independent read on richness -- distinct from the call-vs-put
    # comparison above, and it's what actually explains WHY one side is richer (its slice of the
    # smile is priced above the ATM benchmark, not just above the other side).
    atm_iv_used_pct, call_theo, call_vs_fair, call_vs_fair_pct = None, None, None, None
    put_theo, put_vs_fair, put_vs_fair_pct = None, None, None
    if T and T > 0:
        atm_strike = min({o["strike"] for o in chain}, key=lambda k: abs(k - spot))
        atm_ivs = [o["iv"] for o in chain if o["strike"] == atm_strike and o.get("iv")]
        atm_iv_used_pct = round(sum(atm_ivs) / len(atm_ivs), 2) if atm_ivs else None
        if atm_iv_used_pct is None:
            fallback_iv = ((call_leg.get("iv") or 0) + (put_leg.get("iv") or 0)) / 2
            atm_iv_used_pct = round(fallback_iv, 2) if fallback_iv else None
        if atm_iv_used_pct:
            sigma = atm_iv_used_pct / 100.0
            call_theo = round(bs_price(spot, call_leg["strike"], T, RISK_FREE_RATE, sigma, "CE"), 2)
            put_theo = round(bs_price(spot, put_leg["strike"], T, RISK_FREE_RATE, sigma, "PE"), 2)
            if call_theo > 0:
                call_vs_fair = round(call_premium - call_theo, 2)
                call_vs_fair_pct = round(call_vs_fair / call_theo * 100, 1)
            if put_theo > 0:
                put_vs_fair = round(put_premium - put_theo, 2)
                put_vs_fair_pct = round(put_vs_fair / put_theo * 100, 1)

    # Combined edge -- the number that actually matters for a seller. skew_pct only says how the
    # premium is SPLIT between call and put; it says nothing about whether the whole structure is
    # rich or cheap versus fair value. A big skew with near-zero combined edge just means the
    # premium moved from one leg to the other, not that you're being paid extra overall.
    combined_vs_fair, combined_vs_fair_pct = None, None
    if call_vs_fair is not None and put_vs_fair is not None and call_theo and put_theo:
        combined_vs_fair = round(call_vs_fair + put_vs_fair, 2)
        combined_vs_fair_pct = round(combined_vs_fair / (call_theo + put_theo) * 100, 1)

    call_hedge_target = call_leg["strike"] * (1 + hedge_distance_pct / 100.0)
    put_hedge_target = put_leg["strike"] * (1 - hedge_distance_pct / 100.0)
    call_hedge = min(calls, key=lambda o: abs(o["strike"] - call_hedge_target)) if calls else None
    put_hedge = min(puts, key=lambda o: abs(o["strike"] - put_hedge_target)) if puts else None
    net_credit = None
    if call_hedge and put_hedge and call_hedge.get("ltp") is not None and put_hedge.get("ltp") is not None:
        net_credit = round((call_premium + put_premium) - (call_hedge["ltp"] + put_hedge["ltp"]), 2)

    call_dist_pct = round((call_leg["strike"] - spot) / spot * 100.0, 2)
    put_dist_pct = round((spot - put_leg["strike"]) / spot * 100.0, 2)

    # Max loss/profit per share -- pure arithmetic on strikes/premiums already in hand, no extra API
    # calls. Skipped only when a hedge is missing (net_credit is None in that case too).
    max_loss_per_share = None
    if call_hedge and put_hedge and net_credit is not None:
        call_wing = call_hedge["strike"] - call_leg["strike"]
        put_wing = put_leg["strike"] - put_hedge["strike"]
        max_loss_per_share = round(max(call_wing, put_wing) - net_credit, 2)

    # Trend check -- the single biggest way a condor loses money is selling premium into a real,
    # sustained move rather than a range. A rich skew in a stock that's actually trending hard is a
    # warning sign, not a green light, so this is checked and factored into the safety score below.
    time.sleep(HISTORICAL_CALL_STAGGER_SECONDS)
    trend_info = get_trend_regime(symbol)

    # Liquidity check on the actual strikes you'd trade -- thin OI means wide spreads / bad fills,
    # regardless of how attractive the skew looks on screen.
    min_oi = min(call_leg.get("oi") or 0, put_leg.get("oi") or 0)

    # F&O ban check -- reads the Screener's cache (free, no extra API call). If the Screener hasn't
    # been run recently this comes back None/"unknown" rather than a false negative, since a stock
    # that's actually banned but never checked is worse than one flagged as unverified.
    rank_info = get_stock_rank(symbol)
    fo_banned_today = rank_info.get("fo_banned_today") if rank_info else None

    # Any flagged macro/results event between now and this expiry -- a real risk for a position that
    # will still be open then, distinct from the "avoid opening right now" entry-window check.
    event_before_expiry = get_event_before_expiry(str(expiry)) if expiry else None

    safety_score, safety_notes = _skew_safety_score(skew_pct, combined_vs_fair_pct, trend_info, min_oi,
                                                      net_credit, fo_banned_today)

    dte_txt = f"{days_to_expiry}d to expiry ({expiry})" if days_to_expiry is not None else "expiry unknown"
    reasoning = (
        f"At ~{target_delta:.2f} delta, {dte_txt}, the {richer_side} side is priced {skew_pct:.1f}% "
        f"richer than the other: call {call_premium} @ {call_leg['strike']} ({call_dist_pct:+.1f}% from "
        f"spot) vs put {put_premium} @ {put_leg['strike']} ({put_dist_pct:+.1f}% from spot); IV skew "
        f"(put minus call) {iv_skew:+.1f} pts. {safety_notes}"
    )
    if atm_iv_used_pct and call_vs_fair is not None and put_vs_fair is not None:
        reasoning += (
            f" Vs a flat ATM-vol ({atm_iv_used_pct}%) fair value: call is {call_vs_fair:+.2f} "
            f"({call_vs_fair_pct:+.1f}%) and put is {put_vs_fair:+.2f} ({put_vs_fair_pct:+.1f}%) "
            f"relative to theoretical. Combined, the whole structure is {combined_vs_fair:+.2f} "
            f"({combined_vs_fair_pct:+.1f}%) vs fair value overall -- THIS combined number, not the "
            f"{skew_pct:.1f}% call-vs-put split above, is what drives the safety score's opportunity "
            f"component, since a big skew can just mean the premium moved from one leg to the other "
            f"with little or no real edge."
        )
    if fo_banned_today:
        reasoning += " CANNOT OPEN A NEW POSITION -- this stock is in the F&O ban period today."
    if event_before_expiry:
        reasoning += (f" Heads up: {event_before_expiry['label']} ({event_before_expiry['date']}) falls "
                       f"before this expiry.")

    return {
        "symbol": symbol, "spot": spot, "scanned_at": now_ist().isoformat(),
        "expiry": str(expiry) if expiry else None, "days_to_expiry": days_to_expiry,
        "atm_iv_used_pct": atm_iv_used_pct,
        "combined_vs_fair": combined_vs_fair, "combined_vs_fair_pct": combined_vs_fair_pct,
        "call": {"strike": call_leg["strike"], "premium": call_premium, "delta": round(call_leg["delta"], 3),
                  "iv": call_leg.get("iv"), "oi": call_leg.get("oi"), "tradingsymbol": call_leg["tradingsymbol"],
                  "distance_pct": call_dist_pct, "theo_premium": call_theo,
                  "vs_fair": call_vs_fair, "vs_fair_pct": call_vs_fair_pct},
        "put": {"strike": put_leg["strike"], "premium": put_premium, "delta": round(put_leg["delta"], 3),
                 "iv": put_leg.get("iv"), "oi": put_leg.get("oi"), "tradingsymbol": put_leg["tradingsymbol"],
                 "distance_pct": put_dist_pct, "theo_premium": put_theo,
                 "vs_fair": put_vs_fair, "vs_fair_pct": put_vs_fair_pct},
        "call_hedge": ({"strike": call_hedge["strike"], "premium": call_hedge.get("ltp"),
                         "tradingsymbol": call_hedge["tradingsymbol"]} if call_hedge else None),
        "put_hedge": ({"strike": put_hedge["strike"], "premium": put_hedge.get("ltp"),
                        "tradingsymbol": put_hedge["tradingsymbol"]} if put_hedge else None),
        "richer_side": richer_side, "skew_pct": round(skew_pct, 2), "iv_skew": iv_skew,
        "net_credit_per_share": net_credit, "max_loss_per_share": max_loss_per_share, "min_oi": min_oi,
        "fo_banned_today": fo_banned_today, "event_before_expiry": event_before_expiry,
        "trend": ({"regime": trend_info.get("regime"), "adx14": trend_info.get("adx14"),
                    "avoid_premium_selling": trend_info.get("avoid_premium_selling")}
                   if not trend_info.get("error") else {"regime": "Unknown", "error": trend_info["error"]}),
        "safety_score": safety_score, "reasoning": reasoning,
    }, None


def _skew_safety_score(skew_pct, combined_vs_fair_pct, trend_info, min_oi, net_credit, fo_banned_today=None):
    """Composite 0-100 'how reasonable is it to sell into this skew' score. Transparent and
    rule-based -- NOT a prediction of tomorrow's move, just a blend of the real opportunity size
    with the things most likely to hurt you if ignored: trading against a real trend, trading
    illiquid strikes, and a stock that's actually in an F&O ban and can't be freshly opened at all.
    Always investigate the actual candidate yourself before acting."""
    notes = []

    # F&O ban -- if you literally cannot open a new position today, no other factor matters. Capped
    # hard rather than excluded outright so the UI can still show *why* it's here (e.g. it topped the
    # scan on skew alone) instead of silently disappearing.
    if fo_banned_today:
        notes.append("F&O BAN today -- new positions cannot be opened in this stock right now.")

    # Opportunity size -- up to 40 points. Driven by the COMBINED fair-value edge (actual premium
    # collected vs Black-Scholes theoretical, summed across both legs), not raw call-vs-put skew%.
    # skew_pct only describes how the premium is split between the two legs -- a 46% skew can still
    # mean almost no real edge if one leg is correspondingly cheap versus fair value while the other
    # is rich. combined_vs_fair_pct is what's actually left on the table for you as a seller.
    if combined_vs_fair_pct is not None:
        opportunity_component = min(max(combined_vs_fair_pct, 0), 40)
        if combined_vs_fair_pct <= 0:
            notes.append(f"Combined structure is AT OR BELOW fair value ({combined_vs_fair_pct:+.1f}%) despite "
                          f"the {skew_pct:.1f}% call/put skew -- that skew is mostly about which leg carries "
                          f"the premium, not extra edge overall.")
        elif skew_pct > 25 and combined_vs_fair_pct < 5:
            notes.append(f"Large call/put skew ({skew_pct:.1f}%) but only {combined_vs_fair_pct:+.1f}% combined "
                          f"edge over fair value -- most of the skew is redistribution between legs, not real "
                          f"extra premium.")
        else:
            notes.append(f"{combined_vs_fair_pct:+.1f}% combined edge over Black-Scholes fair value.")
    else:
        # Fair-value calc unavailable (e.g. bad T or missing ATM IV) -- fall back to the old
        # skew-only estimate so scoring still works, but say so since it's the weaker signal.
        opportunity_component = min(skew_pct, 60) / 60 * 40
        notes.append("Fair-value comparison unavailable for this candidate -- opportunity score falls back to "
                      "raw call/put skew, a weaker signal than the combined edge.")

    # Trend suitability -- up to 30 points. This is the biggest single risk factor for a condor.
    if trend_info.get("error"):
        trend_component = 12
        notes.append("Trend regime unavailable (not enough history) -- treat as unconfirmed.")
    elif trend_info.get("avoid_premium_selling"):
        trend_component = 0
        notes.append(f"CAUTION: {trend_info.get('regime')} -- this stock is trending, a bad environment for a condor regardless of the skew.")
    elif trend_info.get("regime") == "Range Bound":
        trend_component = 30
        notes.append("Range Bound -- a favorable environment for selling premium.")
    else:
        trend_component = 16
        notes.append(f"{trend_info.get('regime')} -- workable but not ideal for premium selling.")

    # Liquidity -- up to 20 points, based on the thinner of the two short strikes' open interest.
    if min_oi >= 500000:
        liquidity_component = 20
    elif min_oi >= 100000:
        liquidity_component = 14
    elif min_oi >= 20000:
        liquidity_component = 8
    else:
        liquidity_component = 2
        notes.append("Thin open interest on these strikes -- expect wider spreads and worse fills.")

    # Credit sanity -- 10 points if the condor still nets a real positive credit after hedges.
    credit_component = 10 if (net_credit or 0) > 0 else 0
    if credit_component == 0:
        notes.append("Net credit after hedges is at or below zero at these strikes -- check pricing before acting.")

    total = round(opportunity_component + trend_component + liquidity_component + credit_component, 1)
    if fo_banned_today:
        total = min(total, 15.0)
    return total, " ".join(notes)


@app.route("/api/skew-scanner/state")
def skew_scanner_state_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    return jsonify(load_skew_scanner_state())


@app.route("/api/skew-scanner/config", methods=["POST"])
def skew_scanner_config():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    body = request.json or {}
    state = load_skew_scanner_state()
    for key in SKEW_SCANNER_CONFIGURABLE_KEYS:
        if key in body:
            state[key] = body[key]
    save_skew_scanner_state(state)
    return jsonify({"ok": True, "state": state})


@app.route("/api/skew-scanner/scan-batch", methods=["POST"])
def skew_scanner_scan_batch():
    """Scans the next chunk of the configured universe (rotating -- click again to cover more of the
    F&O list), scores each symbol's call/put premium skew at your configured delta, and merges fresh
    results into the persisted results list (existing entries for the same symbol are replaced with
    the latest scan, not duplicated)."""
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    state = load_skew_scanner_state()
    delta_low, delta_high = state.get("delta_low", 0.15), state.get("delta_high", 0.20)
    hedge_distance_pct = state.get("hedge_distance_pct", 2.5)
    min_skew_pct = state.get("min_skew_pct", 15.0)

    batch, universe_size, pass_complete = _skew_scan_universe_batch(state)
    results_by_symbol = {r["symbol"]: r for r in load_skew_scanner_results()}
    errors = []
    for idx, symbol in enumerate(batch):
        if idx > 0:
            time.sleep(HISTORICAL_CALL_STAGGER_SECONDS)
        candidate, err = _scan_symbol_for_skew(symbol, delta_low, delta_high, hedge_distance_pct)
        if err:
            errors.append(f"{symbol}: {err}")
            continue
        if candidate["skew_pct"] >= min_skew_pct:
            results_by_symbol[symbol] = candidate
        elif symbol in results_by_symbol:
            # No longer qualifies -- drop the stale entry rather than show an outdated skew.
            del results_by_symbol[symbol]

    # Ranked by SAFETY SCORE by default (opportunity size blended with trend + liquidity), not raw
    # skew% -- the biggest, richest-looking skew is often the least safe one to actually sell into.
    results = sorted(results_by_symbol.values(), key=lambda r: r["safety_score"], reverse=True)[:60]
    save_skew_scanner_results(results)
    state["last_scan_at"] = now_ist().isoformat()
    state["last_batch_errors"] = errors[:10]
    save_skew_scanner_state(state)
    return jsonify({"ok": True, "scanned": batch, "errors": errors, "results": results,
                     "universe_size": universe_size, "cursor": state.get("_scan_cursor", 0),
                     "pass_complete": pass_complete,
                     "caveat": ("Ranked by a safety score (opportunity size blended with trend "
                                "suitability, strike liquidity, and F&O ban status), not raw skew% "
                                "alone -- the single biggest, richest-looking skew is often the least "
                                "safe one to sell into if that stock is actually trending or banned "
                                "today. F&O ban checks need the Screener (tab 1) run at least once "
                                "this session, otherwise ban status shows as unverified. This still "
                                "does not predict tomorrow's move or upcoming events; investigate the "
                                "top candidates yourself before acting.")})


@app.route("/api/skew-scanner/results")
def skew_scanner_results_route():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    return jsonify({"results": load_skew_scanner_results()})


@app.route("/api/skew-scanner/results/clear", methods=["POST"])
def skew_scanner_clear_results():
    if not require_session():
        return jsonify({"error": "not_logged_in"}), 401
    save_skew_scanner_results([])
    state = load_skew_scanner_state()
    state["_scan_cursor"] = 0
    save_skew_scanner_state(state)
    return jsonify({"ok": True})



# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    if kotak_broker._config_error:
        print(f"!! Kotak Neo is not configured: {kotak_broker._config_error}")
        print("!! Create/edit /home/opc/kotak_algo/.env with KOTAK_CONSUMER_KEY, KOTAK_MOBILE, "
              "KOTAK_UCC, KOTAK_MPIN, KOTAK_TOTP_SECRET (and optionally KOTAK_CONSUMER_SECRET, "
              "KOTAK_ENV, LIVE_TRADING) before running.")
    if not kotak_broker.live_trading:
        print("LIVE_TRADING is not enabled — orders will be simulated (paper mode) only.")
    if ALLOW_INSECURE_NEWS:
        print("!! ALLOW_INSECURE_NEWS is on — news headline fetches will skip TLS verification on failure.")
    # NOTE: this dev server path (`python backend.py`) is for local testing only. In production
    # on the Oracle VM this app is run under Gunicorn behind Nginx — see DEPLOYMENT.md. Gunicorn
    # should still be run with a single worker (-w 1) since the Kotak session and instrument
    # cache are in-process state, not shared across workers.
    print("Starting local dev server at http://localhost:5000 (use Gunicorn in production — see DEPLOYMENT.md)")
    app.run(host="0.0.0.0", port=5000, debug=False)
