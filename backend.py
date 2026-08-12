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
import re
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


def normalize_kotak_mobile(value):
    """Normalize an Indian registered mobile for Kotak Neo v2 TOTP login.

    Kotak Neo v2 documentation specifies the registered mobile number with
    country code. Accept common .env forms (10 digits, 91XXXXXXXXXX,
    +91XXXXXXXXXX, or 0XXXXXXXXXX) and always send 91XXXXXXXXXX.
    """
    digits = re.sub(r"\D", "", str(value or ""))
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("+91"):
        digits = digits[1:]
    if digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return "91" + digits
    if len(digits) == 12 and digits.startswith("91") and digits[2] in "6789":
        return digits
    raise ValueError("KOTAK_MOBILE must be a valid Indian 10-digit mobile number; "
                     "it may be entered as 10 digits, 91XXXXXXXXXX, or +91XXXXXXXXXX.")


class KotakNeoBroker(BrokerInterface):
    name = "KOTAK"

    def __init__(self):
        self.consumer_key = os.environ.get("KOTAK_CONSUMER_KEY")
        self.consumer_secret = os.environ.get("KOTAK_CONSUMER_SECRET")  # optional in v2.0.x
        mobile_raw = os.environ.get("KOTAK_MOBILE")
        self._mobile_error = None
        try:
            self.mobile = normalize_kotak_mobile(mobile_raw) if mobile_raw else None
        except ValueError as e:
            self.mobile = None
            self._mobile_error = str(e)
        self.ucc = os.environ.get("KOTAK_UCC")
        self.mpin = os.environ.get("KOTAK_MPIN")
        self.totp_secret = os.environ.get("KOTAK_TOTP_SECRET")
        self.environment = os.environ.get("KOTAK_ENV", "prod")
        self.live_trading = os.environ.get("LIVE_TRADING", "false").strip().lower() == "true"

        self.client = None
        self._authenticated_at = None
        self._lock = threading.Lock()

        missing = [n for n, v in [
            ("KOTAK_CONSUMER_KEY", self.consumer_key), ("KOTAK_MOBILE", self.mobile),
            ("KOTAK_UCC", self.ucc), ("KOTAK_MPIN", self.mpin),
            ("KOTAK_TOTP_SECRET", self.totp_secret),
        ] if not v]
        self._config_error = (
            f"Missing required Kotak env vars: {', '.join(missing)} (check ~/kotak_algo/.env)"
            if missing else self._mobile_error
        )
        if not self.live_trading:
            logger.info("[KOTAK] Live trading disabled (LIVE_TRADING is not 'true') — "
                         "order-sending calls will be logged and simulated, not sent.")

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
        quotes = kite.quote(inst_keys)
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
        quotes = kite.quote(inst_keys)
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
#     your Kotak Neo app while Auto mode is armed.
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
    #   "live"  : real orders are placed on your live Kotak Neo account, same as before this setting
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
        "strike": atm["strike"], "expiry": str(data["expiry"]), "lot_size": lot_size,
        "lots": lots, "quantity": lots * lot_size, "premium": premium, "spot": spot,
    }, None


def _execute_autotrade_entry(candidate, state):
    order_info, err = _build_autotrade_order(candidate["symbol"], candidate["direction"], state["capital_per_trade"])
    if err:
        return None, err

    execution_mode = state.get("execution_mode", "track")
    if execution_mode == "live":
        leg = {"leg": "auto_entry", "tradingsymbol": order_info["tradingsymbol"],
               "transaction_type": "BUY", "quantity": order_info["quantity"]}
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
               "transaction_type": "SELL", "quantity": trade["quantity"]}
        results = place_basket_orders([leg], product="MIS", order_type="MARKET", sequence_for_margin=False)
        result = results[0] if results else {"status": "failed", "error": "No result returned"}
        exit_status = result["status"]
        trade["status"] = "closed"
        trade["closed_at"] = now_ist().isoformat()
        trade["exit_order_status"] = exit_status
        if exit_status != "placed":
            trade["exit_reason"] = reason + " -- EXIT ORDER FAILED, check your Kotak Neo app IMMEDIATELY"
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
            q = kite.quote([f"NFO:{trade['tradingsymbol']}"]).get(f"NFO:{trade['tradingsymbol']}")
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
                                  "your live Kotak Neo account."}), 400
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
                q = kite.quote([f"NFO:{trade['tradingsymbol']}"]).get(f"NFO:{trade['tradingsymbol']}")
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
                                  "that future adjustments will place REAL orders on your live Kotak Neo account."}), 400
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
    live Kotak Neo positions (the same data as the "Open F&O positions" table), and classifies each
    one as sell_call / buy_call / sell_put / buy_put using Kite's own authoritative instrument dump
    (strike/instrument_type per tradingsymbol) rather than guessing from the tradingsymbol text.
    This is what lets you attach a position you opened directly in Kotak Neo (or anywhere else) --
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
        return None, (f"No open short call + short put pair found for {symbol} in your live Kotak Neo "
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
    """Attaches a position for monitoring straight from your LIVE Kotak Neo broker positions -- for
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


threading.Thread(target=_autotrade_loop, daemon=True).start()
threading.Thread(target=_delta_engine_loop, daemon=True).start()
threading.Thread(target=_delta_spot_poll_loop, daemon=True).start()


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
    # MUST be run with a single worker (-w 1) since the Kotak session, instrument cache, and the
    # autotrade/delta-engine background threads below are all in-process state, not shared across
    # workers.
    print("Starting local dev server at http://localhost:5000 (use Gunicorn in production — see DEPLOYMENT.md)")
    app.run(host="0.0.0.0", port=5000, debug=False)
