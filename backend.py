"""
Kotak Neo Options Terminal - production-oriented Flask backend.

Deployment:
  GitHub -> Oracle VM -> Gunicorn -> Nginx -> algo2.wecon.in

Secrets:
  /home/opc/kotak_algo/.env
  KOTAK_CONSUMER_KEY
  KOTAK_MOBILE
  KOTAK_UCC
  KOTAK_MPIN
  KOTAK_TOTP_SECRET
  FLASK_SECRET_KEY
  DASHBOARD_PASSWORD

Safety:
  LIVE_TRADING=false by default.
  Never expose Kotak credentials to the browser.
"""

import os
import time
import logging
import threading
import secrets
from functools import wraps
from datetime import datetime, timezone

from flask import (
    Flask, jsonify, request, session, send_from_directory,
)
from dotenv import load_dotenv
import pyotp

from neo_api_client import NeoAPI

ENV_FILE = os.getenv("KOTAK_ENV_FILE", "/home/opc/kotak_algo/.env")
load_dotenv(ENV_FILE)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("PORT", "5000"))
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
SESSION_MAX_AGE = int(os.getenv("KOTAK_SESSION_MAX_AGE", "28800"))
BROKER_SEGMENT = os.getenv("KOTAK_EXCHANGE_SEGMENT", "nse_fo")

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
                "LIVE_TRADING=false. Real orders are disabled on the server.",
                409,
            )
        return fn(*args, **kwargs)
    return wrapper


def _require_env(*names):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError("Missing server configuration: " + ", ".join(missing))


class KotakService:
    def _new_client(self):
        _require_env("KOTAK_CONSUMER_KEY")
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

            _require_env(
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
            log.warning("Kotak %s failed; re-authenticating: %s", method, exc)
            _broker = None
            self.authenticate(force=True)
            result = getattr(self.client(), method)(**kwargs)

        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(result["error"])
        return result


kotak = KotakService()


def normalize_rows(value):
    if isinstance(value, dict):
        value = value.get("data", value.get("result", value))
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def pick(row, *keys):
    if not isinstance(row, dict):
        return None
    for key in keys:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def number(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audit(event, payload=None):
    log.info(
        "AUDIT event=%s user=%s payload=%s",
        event,
        session.get("user", "unknown"),
        payload or {},
    )


@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    supplied = str(body.get("password", ""))

    configured = os.getenv("DASHBOARD_PASSWORD")
    if not configured:
        return json_error(
            "DASHBOARD_PASSWORD is not configured on the Oracle VM.", 503
        )

    if not secrets.compare_digest(supplied, configured):
        audit("login_failed")
        return json_error("Invalid dashboard password.", 401)

    session["authenticated"] = True
    session["user"] = "admin"
    session.permanent = True
    audit("login_success")
    return json_ok()


@app.post("/api/logout")
def logout():
    session.clear()
    return json_ok()


@app.get("/api/me")
def me():
    return json_ok(
        authenticated=bool(session.get("authenticated")),
        live_trading=LIVE_TRADING,
    )


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
        audit("broker_reconnect")
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


@app.get("/api/search")
@dashboard_required
def search():
    symbol = request.args.get("symbol", "").strip()
    if not symbol:
        return json_error("Enter a symbol.")
    segment = request.args.get("segment", BROKER_SEGMENT)
    try:
        return json_ok(data=kotak.call(
            "search_scrip",
            exchange_segment=segment,
            symbol=symbol,
        ))
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
        return json_ok(data=kotak.call(
            "quotes",
            instrument_tokens=instruments,
        ))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/screener")
@dashboard_required
def screener():
    """
    Ranks a user-supplied Kotak instrument list.
    It intentionally does not invent historical HV/IV.
    """
    body = request.get_json(silent=True) or {}
    instruments = body.get("instruments", [])
    if not isinstance(instruments, list) or not instruments:
        return json_error("Add Kotak instrument tokens to scan.")

    try:
        raw = kotak.call("quotes", instrument_tokens=instruments[:50])
        rows = normalize_rows(raw)
        ranked = []

        for row in rows:
            ltp = number(pick(row, "ltp", "last_price", "LTP"))
            prev = number(pick(row, "previous_close", "prev_close", "close"))
            high = number(pick(row, "high", "ohlc_high"))
            low = number(pick(row, "low", "ohlc_low"))
            oi = number(pick(row, "oi", "open_interest"), 0)
            volume = number(pick(row, "volume", "vol_traded"), 0)

            change = None if ltp is None or not prev else (ltp - prev) / prev * 100
            intraday_range = None
            if ltp and high is not None and low is not None:
                intraday_range = (high - low) / ltp * 100

            # Current-session suitability score only.
            score = 50.0
            if intraday_range is not None:
                score += max(-35, min(30, 20 - intraday_range * 5))
            if change is not None:
                score += max(-15, min(15, 10 - abs(change) * 2))
            if volume > 0:
                score += 5

            ranked.append({
                "symbol": pick(row, "trading_symbol", "symbol", "display_name"),
                "token": pick(row, "instrument_token", "token"),
                "ltp": ltp,
                "change_pct": round(change, 2) if change is not None else None,
                "range_pct": round(intraday_range, 2) if intraday_range is not None else None,
                "volume": volume,
                "oi": oi,
                "score": round(max(0, min(100, score)), 1),
            })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return json_ok(
            results=ranked,
            methodology="Current quote/range ranking. Historical HV/IV rank requires historical data.",
        )
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/strategy/review")
@dashboard_required
def strategy_review():
    body = request.get_json(silent=True) or {}
    strategy = body.get("strategy")
    legs = body.get("legs", [])

    if strategy not in {"iron_condor", "strangle"}:
        return json_error("Choose Iron Condor or Naked Strangle.")
    if not isinstance(legs, list):
        return json_error("legs must be a list.")

    required = {"side", "quantity", "price", "strike", "option_type", "trading_symbol"}
    normalized = []
    credit = 0.0

    for leg in legs:
        if not required.issubset(leg):
            return json_error("Every leg needs side, quantity, price, strike, option_type and trading_symbol.")

        side = str(leg["side"]).upper()
        if side not in {"BUY", "SELL"}:
            return json_error("Side must be BUY or SELL.")

        qty = int(leg["quantity"])
        price = number(leg["price"])
        strike = number(leg["strike"])
        if qty <= 0 or price is None or strike is None:
            return json_error("Invalid leg quantity/price/strike.")

        credit += price * qty if side == "SELL" else -price * qty
        normalized.append({
            "trading_symbol": leg["trading_symbol"],
            "exchange_segment": leg.get("exchange_segment", BROKER_SEGMENT),
            "product": leg.get("product", "NRML"),
            "transaction_type": "S" if side == "SELL" else "B",
            "side": side,
            "quantity": qty,
            "price": price,
            "strike": strike,
            "option_type": str(leg["option_type"]).upper(),
            "instrument_token": leg.get("instrument_token"),
        })

    if strategy == "iron_condor" and len(normalized) != 4:
        return json_error("Iron Condor requires four legs.")
    if strategy == "strangle" and len(normalized) != 2:
        return json_error("Strangle requires two legs.")

    max_loss = None
    if strategy == "iron_condor":
        calls = sorted([x["strike"] for x in normalized if x["option_type"] == "CE"])
        puts = sorted([x["strike"] for x in normalized if x["option_type"] == "PE"])
        if len(calls) == 2 and len(puts) == 2:
            wing = max(calls[1] - calls[0], puts[1] - puts[0])
            max_loss = max(0, wing * max(x["quantity"] for x in normalized) - credit)

    return json_ok(
        strategy=strategy,
        legs=normalized,
        net_credit=round(credit, 2),
        max_loss=round(max_loss, 2) if max_loss is not None else None,
        note="Review strikes, quantity, premium and margin before any live order.",
    )


def _validate_order(body):
    required = [
        "exchange_segment", "product", "order_type",
        "quantity", "trading_symbol", "transaction_type"
    ]
    missing = [x for x in required if body.get(x) in (None, "")]
    if missing:
        raise ValueError("Missing: " + ", ".join(missing))

    quantity = int(body["quantity"])
    if quantity <= 0:
        raise ValueError("Quantity must be positive.")

    transaction = str(body["transaction_type"]).upper()
    if transaction not in {"B", "S"}:
        raise ValueError("transaction_type must be B or S.")

    order_type = str(body["order_type"]).upper()
    if order_type not in {"MKT", "L", "SL", "SL-M"}:
        raise ValueError("Unsupported order_type.")

    return {
        "exchange_segment": body["exchange_segment"],
        "product": body["product"],
        "price": str(body.get("price", "0")),
        "order_type": order_type,
        "quantity": str(quantity),
        "validity": body.get("validity", "DAY"),
        "trading_symbol": body["trading_symbol"],
        "transaction_type": transaction,
    }


@app.post("/api/orders/preview")
@dashboard_required
def preview_order():
    try:
        order = _validate_order(request.get_json(silent=True) or {})
        return json_ok(order=order, live_trading=LIVE_TRADING)
    except Exception as exc:
        return json_error(str(exc))


@app.post("/api/orders/place")
@dashboard_required
@live_required
def place_order():
    try:
        order = _validate_order(request.get_json(silent=True) or {})
        audit("live_order_attempt", order)
        result = kotak.call("place_order", **order)
        audit("live_order_result", {"trading_symbol": order["trading_symbol"]})
        return json_ok(response=result)
    except Exception as exc:
        audit("live_order_error", {"error": str(exc)})
        return json_error(str(exc), 502)


@app.post("/api/orders/modify")
@dashboard_required
@live_required
def modify_order():
    body = request.get_json(silent=True) or {}
    if not body.get("order_id"):
        return json_error("order_id required.")
    try:
        audit("live_modify_attempt", body)
        return json_ok(response=kotak.call("modify_order", **body))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.post("/api/orders/cancel")
@dashboard_required
@live_required
def cancel_order():
    body = request.get_json(silent=True) or {}
    if not body.get("order_id"):
        return json_error("order_id required.")
    try:
        audit("live_cancel_attempt", body)
        return json_ok(response=kotak.call("cancel_order", **body))
    except Exception as exc:
        return json_error(str(exc), 502)


@app.get("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "kotak-neo-options-terminal",
        "time": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
