"""
Kotak Neo Options Terminal - automatic F&O screener + delta/hedge strategy builder.

Designed to run behind Gunicorn/Nginx on the Oracle VM.

Main workflow:
    Kotak scrip master -> current F&O stock universe -> live spot/option quotes
    -> Black-Scholes IV/delta -> target-delta short CE/PE -> percentage hedge
    -> credit/max-loss -> ranking.

Safety:
    LIVE_TRADING=false by default. Real orders are blocked unless explicitly enabled
    on the server. The browser never receives Kotak credentials or TOTP material.
"""

import csv
import io
import logging
import math
import os
import re
import secrets
import threading
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from functools import wraps

from flask import Flask, jsonify, request, session, send_from_directory
from dotenv import load_dotenv
import pyotp

try:
    from neo_api_client import NeoAPI
except ImportError as exc:
    raise SystemExit("Install neo-api-client before starting the server.") from exc

ENV_FILE = os.getenv("KOTAK_ENV_FILE", "/home/opc/kotak_algo/.env")
load_dotenv(ENV_FILE)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("PORT", "5000"))
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").strip().lower() == "true"
SESSION_MAX_AGE = int(os.getenv("KOTAK_SESSION_MAX_AGE", "28800"))
MASTER_CACHE_SECONDS = int(os.getenv("KOTAK_MASTER_CACHE_SECONDS", "1800"))
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.06"))
MIN_DAYS_TO_EXPIRY = int(os.getenv("MIN_DAYS_TO_EXPIRY", "1"))
DEFAULT_TARGET_DELTA = float(os.getenv("TARGET_SHORT_DELTA", "0.10"))
DEFAULT_HEDGE_PCT = float(os.getenv("HEDGE_PCT", "0.02"))
MAX_SCAN_SYMBOLS = int(os.getenv("MAX_SCAN_SYMBOLS", "100"))

app = Flask(__name__, static_folder=APP_DIR, static_url_path="")
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("COOKIE_SECURE", "true").lower() == "true"

log = logging.getLogger("kotak_terminal")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_broker = None
_authenticated_at = 0.0
_broker_lock = threading.RLock()
_master_lock = threading.RLock()
_master_cache = {"fetched_at": 0.0, "rows": []}


# ---------------------------------------------------------------------------
# Generic response/auth helpers
# ---------------------------------------------------------------------------
def json_ok(**data):
    return jsonify({"ok": True, **data})


def json_error(message, status=400):
    return jsonify({"ok": False, "error": str(message)}), status


def dashboard_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return json_error("Dashboard login required.", 401)
        return fn(*args, **kwargs)
    return wrapper


def live_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not LIVE_TRADING:
            return json_error(
                "LIVE_TRADING=false. Real orders are disabled on the server.", 409
            )
        return fn(*args, **kwargs)
    return wrapper


def require_env(*names):
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing server configuration: " + ", ".join(missing))


# ---------------------------------------------------------------------------
# Kotak Neo connection
# ---------------------------------------------------------------------------
class KotakService:
    def _new_client(self):
        require_env("KOTAK_CONSUMER_KEY")
        return NeoAPI(
            environment="prod",
            access_token=None,
            neo_fin_key=None,
            consumer_key=os.getenv("KOTAK_CONSUMER_KEY"),
        )

    def authenticate(self, force=False):
        global _broker, _authenticated_at
        with _broker_lock:
            if (
                not force
                and _broker is not None
                and time.time() - _authenticated_at < SESSION_MAX_AGE
            ):
                return True

            require_env(
                "KOTAK_CONSUMER_KEY",
                "KOTAK_MOBILE",
                "KOTAK_UCC",
                "KOTAK_MPIN",
                "KOTAK_TOTP_SECRET",
            )

            client = self._new_client()
            totp = pyotp.TOTP(os.getenv("KOTAK_TOTP_SECRET")).now()
            login = client.totp_login(
                mobile_number="+91" + os.getenv("KOTAK_MOBILE"),
                ucc=os.getenv("KOTAK_UCC"),
                totp=totp,
            )
            if isinstance(login, dict) and login.get("error"):
                raise RuntimeError(login["error"])

            validated = client.totp_validate(mpin=os.getenv("KOTAK_MPIN"))
            if isinstance(validated, dict) and validated.get("error"):
                raise RuntimeError(validated["error"])

            _broker = client
            _authenticated_at = time.time()
            log.info("Kotak Neo authentication successful.")
            return True

    def client(self):
        self.authenticate()
        return _broker

    def call(self, method, **kwargs):
        global _broker
        self.authenticate()
        fn = getattr(self.client(), method)
        try:
            result = fn(**kwargs)
        except Exception as exc:
            log.warning("Kotak %s failed; re-authenticating once: %s", method, exc)
            _broker = None
            self.authenticate(force=True)
            result = getattr(self.client(), method)(**kwargs)

        if isinstance(result, dict):
            err = result.get("error") or result.get("Error")
            if err:
                raise RuntimeError(str(err))
        return result


kotak = KotakService()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def safe_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    n = safe_float(value)
    return int(n) if n is not None else default


def first_value(row, *keys):
    if not isinstance(row, dict):
        return None
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def response_rows(value):
    """Flatten common Neo response shapes to a list of dictionaries."""
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if not isinstance(value, dict):
        return []

    for key in ("data", "result", "response", "scrips", "records"):
        child = value.get(key)
        if isinstance(child, list):
            return [x for x in child if isinstance(x, dict)]
        if isinstance(child, dict):
            # A single quote/search result may itself be the row.
            if any(k in child for k in ("instrument_token", "token", "ltp", "last_price", "LTP", "pSymbol", "pTrdSymbol")):
                return [child]
            rows = response_rows(child)
            if rows:
                return rows

    # The response itself may be one row.
    if any(k in value for k in ("instrument_token", "token", "ltp", "last_price", "LTP", "pSymbol", "pTrdSymbol")):
        return [value]

    # Quote responses are sometimes keyed by token/symbol.
    dict_rows = []
    for key, child in value.items():
        if isinstance(child, dict):
            row = dict(child)
            if not any(k in row for k in ("instrument_token", "token", "ltp", "last_price", "LTP")):
                row["instrument_token"] = key
            dict_rows.append(row)
    return dict_rows


def parse_date(value):
    if value in (None, "", 0, "0"):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y",
        "%d%b%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass

    n = safe_float(value)
    if n is not None:
        # Scrip-master expiry values have historically appeared as Unix seconds.
        try:
            d = datetime.fromtimestamp(n, tz=timezone.utc).date()
            # Some legacy Kotak examples applied a +10-year adjustment. Keep the
            # date sensible if a legacy value is encountered.
            today = datetime.now(timezone.utc).date()
            if d.year > today.year + 7:
                try:
                    d = d.replace(year=d.year - 10)
                except ValueError:
                    d = d.replace(year=d.year - 10, day=28)
            return d
        except (OverflowError, OSError, ValueError):
            return None
    return None


def parse_expiry(value):
    return parse_date(value)


def derive_underlying(row):
    explicit = first_value(
        row,
        "pUnderlying", "pUnderlyingSymbol", "underlying", "underlying_symbol",
        "pSymbolName", "symbol_name", "underlyingName", "pScripName",
    )
    if explicit:
        s = str(explicit).strip().upper()
        # Avoid returning generic instrument labels.
        if s not in {"OPTSTK", "OPTIDX", "OPTIONS", "NSE", "NSE_FO"}:
            return s

    trading = str(first_value(row, "pTrdSymbol", "trading_symbol", "tradingsymbol", "symbol") or "").upper()
    if not trading:
        return None

    # Most NSE option symbols have an expiry marker such as 26AUG or 26AUG26.
    m = re.search(r"\d{2}[A-Z]{3}(?:\d{2,4})?", trading)
    if m:
        return trading[:m.start()].strip()

    # If the row has a numeric strike and CE/PE suffix, strip the suffix and try
    # to identify the prefix. This is only a fallback.
    stripped = re.sub(r"(?:CE|PE)$", "", trading)
    return stripped or None


def normalize_scrip_row(row, default_segment="nse_fo"):
    token = first_value(row, "pSymbol", "instrument_token", "token", "symbol_token")
    segment = str(first_value(row, "pExchSeg", "exchange_segment", "segment") or default_segment).lower()
    trading_symbol = first_value(row, "pTrdSymbol", "trading_symbol", "tradingsymbol")
    inst_type = str(first_value(row, "pInstType", "instrument_type", "inst_type") or "").upper()
    option_type = str(first_value(row, "pOptionType", "option_type", "optionType") or "").upper()
    if option_type in {"CALL", "C"}:
        option_type = "CE"
    elif option_type in {"PUT", "P"}:
        option_type = "PE"

    # Some masters identify options by the trading-symbol suffix even when the
    # option type field is blank.
    ts = str(trading_symbol or "").upper()
    if not option_type and ts.endswith("CE"):
        option_type = "CE"
    elif not option_type and ts.endswith("PE"):
        option_type = "PE"

    expiry = parse_expiry(first_value(row, "pExpiryDate", "expiry", "expiry_date", "expiryDate"))
    strike = safe_float(first_value(row, "pStrikePrice", "strike_price", "strike", "dStrikePrice"))
    lot_size = safe_int(first_value(row, "lLotSize", "llotSize", "lot_size", "lotSize"), 1)
    freeze_qty = safe_int(first_value(row, "lFreezeQty", "freeze_qty", "freezeQty"), 0)

    return {
        "token": str(token) if token not in (None, "") else None,
        "exchange_segment": segment,
        "trading_symbol": str(trading_symbol).strip() if trading_symbol else None,
        "instrument_type": inst_type,
        "option_type": option_type,
        "expiry": expiry,
        "strike": strike,
        "lot_size": lot_size or 1,
        "freeze_qty": freeze_qty,
        "underlying": derive_underlying(row),
        "raw": row,
    }


def download_csv_rows(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "KotakNeoOptionsTerminal/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def extract_master_paths(response):
    if isinstance(response, dict):
        for key in ("filesPaths", "filePaths", "files_paths", "file_paths"):
            paths = response.get(key)
            if isinstance(paths, list):
                return [str(x) for x in paths if x]
            if isinstance(paths, str):
                return [paths]
        # Sometimes the paths are nested under data.
        for key in ("data", "result", "response"):
            child = response.get(key)
            if isinstance(child, dict):
                paths = extract_master_paths(child)
                if paths:
                    return paths
    return []


def load_scrip_master(force=False):
    """Load Kotak's current scrip master and normalize it in memory."""
    now = time.time()
    with _master_lock:
        if _master_cache["rows"] and not force and now - _master_cache["fetched_at"] < MASTER_CACHE_SECONDS:
            return _master_cache["rows"]

        # The v2 SDK documents scrip_master() and the current Kotak API exposes
        # exchange-specific scrip-master CSV paths. We request both segments.
        all_rows = []
        seen = set()
        for segment in ("nse_fo", "nse_cm"):
            try:
                response = kotak.call("scrip_master", exchange_segment=segment)
            except Exception as exc:
                log.warning("Could not fetch %s scrip master: %s", segment, exc)
                continue

            paths = extract_master_paths(response)
            if paths:
                for path in paths:
                    low = path.lower()
                    if segment == "nse_fo" and "nse_fo" not in low:
                        continue
                    if segment == "nse_cm" and "nse_cm" not in low:
                        continue
                    if path in seen:
                        continue
                    seen.add(path)
                    try:
                        raw_rows = download_csv_rows(path)
                        all_rows.extend(
                            normalize_scrip_row(r, segment) for r in raw_rows
                        )
                    except Exception as exc:
                        log.warning("Could not read Kotak scrip master %s: %s", path, exc)
            else:
                # Some SDK versions return the rows directly.
                for raw in response_rows(response):
                    row = normalize_scrip_row(raw, segment)
                    if row["token"] or row["trading_symbol"]:
                        all_rows.append(row)

        # Remove duplicates while retaining the most complete record.
        dedup = {}
        for row in all_rows:
            key = (
                row.get("exchange_segment"),
                row.get("token"),
                row.get("trading_symbol"),
                row.get("expiry"),
                row.get("strike"),
                row.get("option_type"),
            )
            dedup[key] = row

        rows = list(dedup.values())
        if not rows:
            raise RuntimeError(
                "Kotak scrip master returned no usable rows. Check the Neo API session "
                "and the installed neo-api-client version."
            )

        _master_cache["rows"] = rows
        _master_cache["fetched_at"] = time.time()
        log.info("Loaded %s normalized Kotak scrip-master rows.", len(rows))
        return rows


def option_rows_for_symbol(symbol):
    symbol = symbol.upper().strip()
    rows = load_scrip_master()
    result = []
    for row in rows:
        if row["exchange_segment"] != "nse_fo":
            continue
        if row["option_type"] not in {"CE", "PE"}:
            continue
        if not row["expiry"] or row["strike"] is None:
            continue
        underlying = (row.get("underlying") or "").upper()
        trading = (row.get("trading_symbol") or "").upper()
        # Prefer explicit underlying, then exact symbol prefix match.
        if underlying == symbol or trading.startswith(symbol):
            result.append(row)
    return result


def stock_universe_from_master():
    rows = load_scrip_master()
    symbols = set()
    for row in rows:
        if row["exchange_segment"] != "nse_fo":
            continue
        if row["option_type"] not in {"CE", "PE"}:
            continue
        if row["instrument_type"] and "OPTSTK" not in row["instrument_type"]:
            continue
        underlying = row.get("underlying")
        if underlying and re.fullmatch(r"[A-Z0-9&.-]+", underlying):
            symbols.add(underlying)
    # Optional manual supplement for a master that omits underlying names.
    extra = os.getenv("KOTAK_FNO_SYMBOLS", "")
    if extra:
        symbols.update(x.strip().upper() for x in extra.split(",") if x.strip())
    return sorted(symbols)


# ---------------------------------------------------------------------------
# Market quotes / Black-Scholes
# ---------------------------------------------------------------------------
def quote_rows(response):
    return response_rows(response)


def extract_quote(row):
    if not isinstance(row, dict):
        return None
    ltp = safe_float(first_value(row, "ltp", "last_price", "LTP", "lastPrice"))
    if ltp is None:
        # A nested quote can occur in some responses.
        nested = first_value(row, "data", "quote")
        if isinstance(nested, dict):
            ltp = safe_float(first_value(nested, "ltp", "last_price", "LTP"))
            row = nested
    if ltp is None or ltp <= 0:
        return None
    return {
        "ltp": ltp,
        "oi": safe_float(first_value(row, "oi", "open_interest", "openInterest"), 0),
        "volume": safe_float(first_value(row, "volume", "vol_traded", "v", "trade_volume"), 0),
        "previous_close": safe_float(first_value(row, "previous_close", "prev_close", "close", "closePrice")),
        "high": safe_float(first_value(row, "high", "ohlc_high")),
        "low": safe_float(first_value(row, "low", "ohlc_low")),
    }


def get_quotes_for_rows(rows, quote_type="all"):
    tokens = [
        {"instrument_token": str(r["token"]), "exchange_segment": r["exchange_segment"]}
        for r in rows if r.get("token") and r.get("exchange_segment")
    ]
    if not tokens:
        return {}

    # Kotak documents a list of {instrument_token, exchange_segment} dictionaries.
    # Keep requests modest to avoid broker-side throttling.
    out = {}
    for start in range(0, len(tokens), 50):
        batch = tokens[start:start + 50]
        raw = kotak.call("quotes", instrument_tokens=batch, quote_type=quote_type)
        rows_out = quote_rows(raw)
        for qrow in rows_out:
            token = first_value(qrow, "instrument_token", "token", "tk", "pSymbol")
            if token is not None:
                out[str(token)] = qrow
            # A keyed quote response may have the token as a key and is handled by
            # response_rows() as well.
    return out


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t, rate, sigma, call=True):
    if min(spot, strike, t, sigma) <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if call:
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
    return strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def implied_vol(market_price, spot, strike, t, rate, call):
    if not all(v is not None for v in (market_price, spot, strike, t)) or t <= 0:
        return None
    if market_price <= 0 or spot <= 0 or strike <= 0:
        return None

    intrinsic = max(0.0, spot - strike) if call else max(0.0, strike - spot)
    upper = spot if call else strike * math.exp(-rate * t)
    if market_price < intrinsic * 0.999 or market_price > upper * 1.001:
        return None

    lo, hi = 0.01, 5.0
    for _ in range(70):
        mid = (lo + hi) / 2
        value = bs_price(spot, strike, t, rate, mid, call)
        if value is None:
            return None
        if value > market_price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def bs_delta(spot, strike, t, rate, sigma, call=True):
    if not all(v is not None for v in (spot, strike, t, sigma)) or min(spot, strike, t, sigma) <= 0:
        return None
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1.0


def enrich_chain(symbol, expiry_str=None):
    symbol = symbol.upper().strip()
    opts = option_rows_for_symbol(symbol)
    if not opts:
        return None, f"No Kotak NSE F&O option contracts found for {symbol}."

    today = datetime.now(timezone.utc).date()
    expiries = sorted({r["expiry"] for r in opts if r["expiry"] and r["expiry"] >= today})
    if not expiries:
        return None, f"No future option expiry found for {symbol}."

    if expiry_str:
        try:
            expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except ValueError:
            return None, "Expiry must be YYYY-MM-DD."
        if expiry not in expiries:
            return None, f"{expiry_str} is not available for {symbol}."
    else:
        eligible = [e for e in expiries if (e - today).days >= MIN_DAYS_TO_EXPIRY]
        if not eligible:
            eligible = expiries
        expiry = eligible[0]

    chain = [r for r in opts if r["expiry"] == expiry]
    if not chain:
        return None, f"No contracts found for {symbol} {expiry}."

    # Underlying spot is taken from NSE cash segment using the scrip master.
    cash_rows = load_scrip_master()
    cash = [
        r for r in cash_rows
        if r["exchange_segment"] == "nse_cm"
        and (r.get("trading_symbol") or "").upper() == symbol
    ]
    if not cash:
        # Search endpoint is a useful fallback if the cash master is missing a symbol.
        try:
            found = kotak.call("search_scrip", exchange_segment="nse_cm", symbol=symbol)
            for raw in response_rows(found):
                nr = normalize_scrip_row(raw, "nse_cm")
                if nr.get("token"):
                    cash.append(nr)
        except Exception:
            pass
    if not cash:
        return None, f"Could not find NSE cash token for {symbol}."

    q = get_quotes_for_rows([cash[0]], quote_type="all")
    spot_row = q.get(str(cash[0]["token"]))
    spot_quote = extract_quote(spot_row) if spot_row else None
    if not spot_quote:
        # Token-key matching can vary, so use the first quote row as fallback.
        if q:
            spot_quote = extract_quote(next(iter(q.values())))
    if not spot_quote:
        return None, f"Could not obtain live spot quote for {symbol}."
    spot = spot_quote["ltp"]

    # Fetch option quotes in batches.
    quote_map = get_quotes_for_rows(chain, quote_type="all")
    t = max((expiry - today).days / 365.0, 1.0 / 365.0)
    enriched = []
    for contract in chain:
        rawq = quote_map.get(str(contract["token"]))
        qv = extract_quote(rawq) if rawq else None
        if not qv:
            continue
        is_call = contract["option_type"] == "CE"
        iv = implied_vol(qv["ltp"], spot, contract["strike"], t, RISK_FREE_RATE, is_call)
        delta = bs_delta(spot, contract["strike"], t, RISK_FREE_RATE, iv, is_call) if iv else None
        enriched.append({
            **contract,
            "ltp": qv["ltp"],
            "oi": qv["oi"],
            "volume": qv["volume"],
            "delta": delta,
            "iv": iv * 100 if iv else None,
        })

    if not enriched:
        return None, f"Kotak returned no live option quotes for {symbol} {expiry}."

    return {
        "symbol": symbol,
        "spot": spot,
        "expiry": expiry,
        "dte": (expiry - today).days,
        "lot_size": max(1, chain[0].get("lot_size", 1)),
        "contracts": enriched,
        "all_expiries": [str(e) for e in expiries],
    }, None


# ---------------------------------------------------------------------------
# Automatic delta/hedge selection
# ---------------------------------------------------------------------------
def closest_delta(items, target, call):
    candidates = []
    for x in items:
        d = x.get("delta")
        if d is None:
            continue
        diff = abs(d - target) if call else abs(abs(d) - target)
        candidates.append((diff, x))
    return min(candidates, key=lambda z: z[0])[1] if candidates else None


def outward_hedge(items, short_strike, hedge_pct, call):
    """Choose the first available strike at least hedge_pct beyond the short strike.

    For a CE: long strike >= short*(1+hedge_pct).
    For a PE: long strike <= short*(1-hedge_pct).
    If the exact percentage does not exist, choose the nearest available strike
    beyond the threshold. This is deliberate because strikes are discrete.
    """
    if call:
        target = short_strike * (1.0 + hedge_pct)
        candidates = [x for x in items if x["strike"] >= target]
        return min(candidates, key=lambda x: x["strike"]) if candidates else None
    target = short_strike * (1.0 - hedge_pct)
    candidates = [x for x in items if x["strike"] <= target]
    return max(candidates, key=lambda x: x["strike"]) if candidates else None


def leg_view(row, side):
    return {
        "trading_symbol": row["trading_symbol"],
        "side": side,
        "strike": row["strike"],
        "ltp": round(row["ltp"], 2) if row.get("ltp") is not None else None,
        "delta": round(row["delta"], 4) if row.get("delta") is not None else None,
        "iv": round(row["iv"], 2) if row.get("iv") is not None else None,
        "oi": int(row.get("oi") or 0),
        "volume": int(row.get("volume") or 0),
        "token": row["token"],
        "exchange_segment": row["exchange_segment"],
        "lot_size": row.get("lot_size", 1),
        "option_type": row["option_type"],
    }


def build_iron_condor(symbol, target_delta, hedge_pct, lots, expiry=None):
    if not 0.01 <= target_delta <= 0.49:
        return None, "Target delta must be between 0.01 and 0.49."
    if not 0.001 <= hedge_pct <= 0.30:
        return None, "Hedge percentage must be between 0.1% and 30%."
    if lots < 1:
        return None, "Lots must be at least 1."

    data, err = enrich_chain(symbol, expiry)
    if err:
        return None, err

    calls = sorted(
        [x for x in data["contracts"] if x["option_type"] == "CE" and x["strike"] > data["spot"]],
        key=lambda x: x["strike"],
    )
    puts = sorted(
        [x for x in data["contracts"] if x["option_type"] == "PE" and x["strike"] < data["spot"]],
        key=lambda x: x["strike"],
    )
    if not calls or not puts:
        return None, "Not enough quoted OTM CE/PE contracts to build the condor."

    short_call = closest_delta(calls, target_delta, True)
    short_put = closest_delta(puts, target_delta, False)
    if not short_call or not short_put:
        return None, "Could not find short strikes near the requested delta."

    long_call = outward_hedge(calls, short_call["strike"], hedge_pct, True)
    long_put = outward_hedge(puts, short_put["strike"], hedge_pct, False)
    if not long_call or not long_put:
        return None, "Could not find hedge strikes at the requested hedge percentage. Try a smaller hedge %."

    # Ensure the hedge actually lies beyond the short strike.
    if long_call["strike"] <= short_call["strike"] or long_put["strike"] >= short_put["strike"]:
        return None, "Discrete strike spacing is too wide for the requested hedge percentage."

    credit = short_call["ltp"] + short_put["ltp"] - long_call["ltp"] - long_put["ltp"]
    call_width = long_call["strike"] - short_call["strike"]
    put_width = short_put["strike"] - long_put["strike"]
    max_width = max(call_width, put_width)
    max_profit_per_share = max(0.0, credit)
    max_loss_per_share = max(0.0, max_width - credit)
    quantity = data["lot_size"] * lots

    return {
        "symbol": symbol.upper(),
        "spot": round(data["spot"], 2),
        "expiry": str(data["expiry"]),
        "days_to_expiry": data["dte"],
        "lot_size": data["lot_size"],
        "lots": lots,
        "quantity": quantity,
        "target_delta": target_delta,
        "hedge_pct": hedge_pct,
        "strategy_type": "iron_condor",
        "legs": {
            "sell_call": leg_view(short_call, "SELL"),
            "buy_call": leg_view(long_call, "BUY"),
            "sell_put": leg_view(short_put, "SELL"),
            "buy_put": leg_view(long_put, "BUY"),
        },
        "net_credit_per_share": round(credit, 2),
        "max_profit": round(max_profit_per_share * quantity, 2),
        "max_loss": round(max_loss_per_share * quantity, 2),
        "breakeven_upper": round(short_call["strike"] + credit, 2),
        "breakeven_lower": round(short_put["strike"] - credit, 2),
        "short_call_delta": round(abs(short_call["delta"]), 4) if short_call.get("delta") is not None else None,
        "short_put_delta": round(abs(short_put["delta"]), 4) if short_put.get("delta") is not None else None,
        "min_oi": int(min(short_call.get("oi", 0), short_put.get("oi", 0))),
        "all_expiries": data["all_expiries"],
        "method": "Live Kotak quotes + Black-Scholes implied volatility/delta; short legs nearest target delta; hedge legs are the first available strikes at or beyond the requested percentage distance.",
        "warning": "Delta/IV are model estimates from live LTP, not a broker-provided Greeks feed. Verify live bid/ask, margin, results/events, F&O ban status and order quantities before trading.",
    }, None


# ---------------------------------------------------------------------------
# Automatic stock ranking
# ---------------------------------------------------------------------------
def percentile_rank(value, values, reverse=False):
    vals = [v for v in values if v is not None]
    if value is None or not vals:
        return 50.0
    less = sum(v < value for v in vals)
    equal = sum(v == value for v in vals)
    pct = (less + 0.5 * equal) / len(vals) * 100.0
    return 100.0 - pct if reverse else pct


def rank_score(result, all_results):
    # Higher score = better candidate for this specific short-delta condor.
    # Calmness: lower absolute move and lower intraday range are preferred.
    calm = 50.0
    if result.get("day_range_pct") is not None:
        calm = 100.0 - percentile_rank(result["day_range_pct"], [r.get("day_range_pct") for r in all_results])
    if result.get("change_abs_pct") is not None:
        calm2 = 100.0 - percentile_rank(result["change_abs_pct"], [r.get("change_abs_pct") for r in all_results])
        calm = (calm + calm2) / 2

    # Liquidity: OI and volume. The actual short-strike OI is more relevant than
    # equity OI, so it is deliberately part of the score.
    oi_score = percentile_rank(result.get("min_oi"), [r.get("min_oi") for r in all_results])
    vol_score = percentile_rank(result.get("min_volume"), [r.get("min_volume") for r in all_results])
    liquidity = (oi_score + vol_score) / 2

    # Credit relative to the risk width: a direct measure of how much of the
    # defined wing is being monetized.
    rr = result.get("credit_pct_of_wing")
    credit_score = max(0.0, min(100.0, (rr or 0.0) * 2.0))

    # Prefer symmetric deltas near the requested target.
    delta_gap = result.get("delta_gap", 0.0)
    delta_score = max(0.0, 100.0 - delta_gap * 500.0)

    return round(
        calm * 0.30 + liquidity * 0.25 + credit_score * 0.25 + delta_score * 0.20,
        1,
    )


def screen_one(symbol, target_delta, hedge_pct, min_oi, expiry):
    strategy, err = build_iron_condor(symbol, target_delta, hedge_pct, 1, expiry)
    if err:
        raise RuntimeError(err)

    # Reuse the strategy's live data, but we need underlying session stats too.
    # Use the cash token from the master to get OHLC/volume where available.
    rows = load_scrip_master()
    cash = next(
        (r for r in rows if r["exchange_segment"] == "nse_cm" and (r.get("trading_symbol") or "").upper() == symbol.upper()),
        None,
    )
    if cash:
        qmap = get_quotes_for_rows([cash], quote_type="all")
        qrow = qmap.get(str(cash["token"]))
        q = extract_quote(qrow) if qrow else None
    else:
        q = None

    legs = strategy["legs"]
    min_oi_actual = min(legs["sell_call"]["oi"], legs["sell_put"]["oi"])
    min_volume_actual = min(legs["sell_call"]["volume"], legs["sell_put"]["volume"])
    if min_oi_actual < min_oi:
        raise RuntimeError(f"short-leg minimum OI {min_oi_actual} is below filter {min_oi}")

    call_delta = abs(legs["sell_call"]["delta"] or 0)
    put_delta = abs(legs["sell_put"]["delta"] or 0)
    delta_gap = abs(call_delta - target_delta) + abs(put_delta - target_delta)
    wing = max(
        legs["buy_call"]["strike"] - legs["sell_call"]["strike"],
        legs["sell_put"]["strike"] - legs["buy_put"]["strike"],
    )
    credit = strategy["net_credit_per_share"]

    result = {
        "symbol": symbol.upper(),
        "spot": strategy["spot"],
        "call_delta": round(call_delta, 4),
        "put_delta": round(put_delta, 4),
        "call_strike": legs["sell_call"]["strike"],
        "put_strike": legs["sell_put"]["strike"],
        "call_hedge_strike": legs["buy_call"]["strike"],
        "put_hedge_strike": legs["buy_put"]["strike"],
        "credit": credit,
        "max_loss": strategy["max_loss"],
        "min_oi": min_oi_actual,
        "min_volume": min_volume_actual,
        "dte": strategy["days_to_expiry"],
        "credit_pct_of_wing": (credit / wing * 100.0) if wing > 0 else 0.0,
        "delta_gap": delta_gap,
        "change_abs_pct": abs((q["ltp"] - q["previous_close"]) / q["previous_close"] * 100.0) if q and q.get("previous_close") else None,
        "day_range_pct": ((q["high"] - q["low"]) / q["ltp"] * 100.0) if q and q.get("high") is not None and q.get("low") is not None and q.get("ltp") else None,
        "strategy": strategy,
    }
    return result


# ---------------------------------------------------------------------------
# Authentication / account endpoints
# ---------------------------------------------------------------------------
@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    supplied = str(body.get("password", ""))
    configured = os.getenv("DASHBOARD_PASSWORD")
    if not configured:
        return json_error("DASHBOARD_PASSWORD is not configured on the Oracle VM.", 503)
    if not secrets.compare_digest(supplied, configured):
        return json_error("Invalid dashboard password.", 401)
    session["authenticated"] = True
    session["user"] = "admin"
    session.permanent = True
    return json_ok()


@app.post("/api/logout")
def logout():
    session.clear()
    return json_ok()


@app.get("/api/me")
def me():
    return json_ok(authenticated=bool(session.get("authenticated")), live_trading=LIVE_TRADING)


@app.get("/api/status")
@dashboard_required
def status():
    try:
        kotak.authenticate()
        return json_ok(
            connected=True,
            broker="Kotak Neo",
            live_trading=LIVE_TRADING,
            session_age_seconds=round(time.time() - _authenticated_at),
        )
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/reconnect")
@dashboard_required
def reconnect():
    try:
        kotak.authenticate(force=True)
        return json_ok(connected=True, broker="Kotak Neo")
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/positions")
@dashboard_required
def positions():
    try:
        return json_ok(data=kotak.call("positions"))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/holdings")
@dashboard_required
def holdings():
    try:
        return json_ok(data=kotak.call("holdings"))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/limits")
@dashboard_required
def limits():
    try:
        return json_ok(data=kotak.call("limits", segment="ALL", exchange="ALL", product="ALL"))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/orders")
@dashboard_required
def orders():
    try:
        return json_ok(data=kotak.call("order_report"))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/order-history")
@dashboard_required
def order_history():
    try:
        return json_ok(data=kotak.call("order_history"))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/trades")
@dashboard_required
def trades():
    try:
        return json_ok(data=kotak.call("trade_report"))
    except Exception as exc:
        return json_error(str(exc), 502)


# ---------------------------------------------------------------------------
# Instrument endpoints
# ---------------------------------------------------------------------------
@app.get("/api/search")
@dashboard_required
def search():
    symbol = request.args.get("symbol", "").strip()
    segment = request.args.get("segment", "nse_cm")
    if not symbol:
        return json_error("Enter a symbol.")
    try:
        return json_ok(data=kotak.call("search_scrip", exchange_segment=segment, symbol=symbol))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/scrip-master")
@dashboard_required
def scrip_master():
    try:
        segment = request.args.get("segment", "nse_fo")
        return json_ok(data=kotak.call("scrip_master", exchange_segment=segment))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/quotes")
@dashboard_required
def quotes():
    body = request.get_json(silent=True) or {}
    instruments = body.get("instruments")
    if not isinstance(instruments, list) or not instruments:
        return json_error("instruments must be a non-empty list.")
    if len(instruments) > 50:
        return json_error("Maximum 50 instruments per quote request.")
    try:
        return json_ok(data=kotak.call("quotes", instrument_tokens=instruments, quote_type=body.get("quote_type", "all")))
    except Exception as exc:
        return json_error(str(exc), 502)


# ---------------------------------------------------------------------------
# Automatic F&O universe / expiry / chain
# ---------------------------------------------------------------------------
@app.get("/api/fno-universe")
@dashboard_required
def fno_universe():
    try:
        symbols = stock_universe_from_master()
        return json_ok(symbols=symbols, source="Kotak NSE F&O scrip master", count=len(symbols))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/expiries/<symbol>")
@dashboard_required
def expiries(symbol):
    try:
        opts = option_rows_for_symbol(symbol.upper())
        today = datetime.now(timezone.utc).date()
        values = sorted({r["expiry"] for r in opts if r["expiry"] and r["expiry"] >= today})
        return json_ok(symbol=symbol.upper(), expiries=[str(x) for x in values])
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/api/chain/<symbol>")
@dashboard_required
def chain(symbol):
    try:
        data, err = enrich_chain(symbol, request.args.get("expiry"))
        if err:
            return json_error(err, 404)
        return jsonify({
            "ok": True,
            "symbol": data["symbol"],
            "spot": round(data["spot"], 2),
            "expiry": str(data["expiry"]),
            "dte": data["dte"],
            "lot_size": data["lot_size"],
            "contracts": [
                {
                    "option_type": x["option_type"],
                    "strike": x["strike"],
                    "ltp": round(x["ltp"], 2),
                    "delta": round(x["delta"], 4) if x.get("delta") is not None else None,
                    "iv": round(x["iv"], 2) if x.get("iv") is not None else None,
                    "oi": int(x.get("oi") or 0),
                    "volume": int(x.get("volume") or 0),
                    "trading_symbol": x["trading_symbol"],
                    "token": x["token"],
                }
                for x in data["contracts"]
            ],
        })
    except Exception as exc:
        return json_error(str(exc), 502)


# ---------------------------------------------------------------------------
# Automatic screener
# ---------------------------------------------------------------------------
@app.post("/api/screener")
@dashboard_required
def screener():
    body = request.get_json(silent=True) or {}
    symbols = body.get("symbols") or []
    limit = max(1, min(100, safe_int(body.get("limit"), 20)))
    target_delta = safe_float(body.get("target_delta"), DEFAULT_TARGET_DELTA)
    hedge_pct = safe_float(body.get("hedge_pct"), DEFAULT_HEDGE_PCT)
    min_oi = safe_int(body.get("min_oi"), 0)
    expiry = body.get("expiry") or None

    if not symbols:
        try:
            symbols = stock_universe_from_master()
        except Exception as exc:
            return json_error(str(exc), 502)

    # Keep broker load bounded. The UI scan-limit is also used as the number of
    # symbols to evaluate in this pass, so a 20-symbol scan really scans 20.
    scan_limit = max(1, min(MAX_SCAN_SYMBOLS, limit, len(symbols)))
    symbols = [str(s).upper().strip() for s in symbols[:scan_limit] if str(s).strip()]

    results = []
    errors = []
    for idx, symbol in enumerate(symbols, start=1):
        try:
            result = screen_one(symbol, target_delta, hedge_pct, min_oi, expiry)
            results.append(result)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)})

    if results:
        for r in results:
            r["score"] = rank_score(r, results)
        results.sort(key=lambda x: x["score"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        results = results[:limit]

    return jsonify({
        "ok": True,
        "scanned": len(symbols),
        "successful": len(results),
        "results": results,
        "errors": errors,
        "methodology": "Kotak scrip master + live Kotak quotes; Black-Scholes IV/delta; short CE/PE nearest target delta; percentage-based discrete strike hedges; ranking blends calmness, liquidity, credit/wing and delta fit.",
    })


# ---------------------------------------------------------------------------
# Strategy Builder
# ---------------------------------------------------------------------------
@app.get("/api/strategy/<symbol>")
@dashboard_required
def strategy(symbol):
    target_delta = safe_float(request.args.get("target_delta"), DEFAULT_TARGET_DELTA)
    hedge_pct = safe_float(request.args.get("hedge_pct"), DEFAULT_HEDGE_PCT)
    lots = max(1, safe_int(request.args.get("lots"), 1))
    expiry = request.args.get("expiry") or None
    try:
        result, err = build_iron_condor(symbol, target_delta, hedge_pct, lots, expiry)
        if err:
            return json_error(err, 404)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        log.exception("Strategy build failed for %s", symbol)
        return json_error(str(exc), 502)


@app.post("/api/strategy/review")
@dashboard_required
def strategy_review():
    body = request.get_json(silent=True) or {}
    legs = body.get("legs", [])
    if not isinstance(legs, list) or not legs:
        return json_error("legs must be a non-empty list.")
    credit = 0.0
    for leg in legs:
        side = str(leg.get("side", "")).upper()
        qty = safe_int(leg.get("quantity"), 0)
        price = safe_float(leg.get("price"))
        if side not in {"BUY", "SELL"} or qty <= 0 or price is None:
            return json_error("Invalid strategy leg.")
        credit += price * qty if side == "SELL" else -price * qty
    return json_ok(net_credit=round(credit, 2), legs=legs)


# ---------------------------------------------------------------------------
# Orders - paper mode by default
# ---------------------------------------------------------------------------
def validate_order(body):
    required = [
        "exchange_segment", "product", "order_type", "transaction_type",
        "quantity", "trading_symbol",
    ]
    missing = [x for x in required if body.get(x) in (None, "")]
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))

    transaction = str(body["transaction_type"]).upper()
    if transaction not in {"B", "S", "BUY", "SELL"}:
        raise ValueError("transaction_type must be B/S or BUY/SELL.")
    transaction = "B" if transaction in {"B", "BUY"} else "S"

    order_type = str(body["order_type"]).upper()
    if order_type == "MARKET":
        order_type = "MKT"
    if order_type not in {"L", "MKT", "SL", "SL-M"}:
        raise ValueError("order_type must be L, MKT, SL or SL-M.")

    return {
        "exchange_segment": str(body["exchange_segment"]),
        "product": str(body["product"]).upper(),
        "price": str(body.get("price", "0")),
        "order_type": order_type,
        "quantity": str(safe_int(body["quantity"], 0)),
        "validity": body.get("validity", "DAY"),
        "trading_symbol": str(body["trading_symbol"]),
        "transaction_type": transaction,
    }


@app.post("/api/orders/preview")
@dashboard_required
def preview_order():
    try:
        order = validate_order(request.get_json(silent=True) or {})
        return json_ok(order=order, live_trading=LIVE_TRADING)
    except Exception as exc:
        return json_error(str(exc))


@app.post("/api/orders/place")
@dashboard_required
@live_required
def place_order():
    try:
        order = validate_order(request.get_json(silent=True) or {})
        response = kotak.call("place_order", **order)
        return json_ok(mode="LIVE", response=response)
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/orders/modify")
@dashboard_required
@live_required
def modify_order():
    body = request.get_json(silent=True) or {}
    if not body.get("order_id"):
        return json_error("order_id is required.")
    try:
        return json_ok(mode="LIVE", response=kotak.call("modify_order", **body))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/orders/cancel")
@dashboard_required
@live_required
def cancel_order():
    body = request.get_json(silent=True) or {}
    if not body.get("order_id"):
        return json_error("order_id is required.")
    try:
        return json_ok(mode="LIVE", response=kotak.call("cancel_order", **body))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "kotak-neo-options-terminal",
        "time": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


if __name__ == "__main__":
    log.info("Starting Kotak Neo application on port %s; LIVE_TRADING=%s", PORT, LIVE_TRADING)
    app.run(host="0.0.0.0", port=PORT, debug=False)
