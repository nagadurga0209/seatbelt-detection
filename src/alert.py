"""SMS alerting via Twilio.

Credentials are read from environment variables (see .env.example):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_FROM_NUMBER

Nothing is hard-coded, so secrets never end up in version control.
If credentials are missing, the alert is logged to the console instead of
sent, so the detection pipeline still runs end-to-end during development.
"""
from __future__ import annotations

import time

from src import config

# Tracks the last alert time per vehicle to avoid spamming.
_last_alert_at: dict[str, float] = {}


def _cooldown_active(vehicle_number: str) -> bool:
    now = time.time()
    last = _last_alert_at.get(vehicle_number, 0.0)
    if now - last < config.ALERT_COOLDOWN_SECONDS:
        return True
    _last_alert_at[vehicle_number] = now
    return False


def send_alert_sms(
    to_number: str | None = None,
    vehicle_number: str = "UNKNOWN",
    message: str | None = None,
) -> str | None:
    """Send a seat-belt-violation SMS. Returns the Twilio message SID, or None.

    Falls back to a console log when credentials or the recipient are missing,
    or while the per-vehicle cooldown is active.
    """
    to_number = to_number or config.ALERT_TO_NUMBER
    body = message or (
        f"⚠️ Alert: Seat belt not detected for vehicle "
        f"{vehicle_number}. Please ensure driver safety."
    )

    if _cooldown_active(vehicle_number):
        return None

    if not (
        config.TWILIO_ACCOUNT_SID
        and config.TWILIO_AUTH_TOKEN
        and config.TWILIO_FROM_NUMBER
        and to_number
    ):
        print(
            "[alert] Twilio not fully configured; logging instead of sending.\n"
            f"        -> {body}"
        )
        return None

    try:
        from twilio.rest import Client
    except ImportError:
        print("[alert] twilio not installed (`pip install twilio`). Logging:")
        print(f"        -> {body}")
        return None

    try:
        client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        msg = client.messages.create(
            body=body,
            from_=config.TWILIO_FROM_NUMBER,
            to=to_number,
        )
        print(f"[alert] SMS sent (sid={msg.sid})")
        return msg.sid
    except Exception as exc:  # network / auth / number errors
        print(f"[alert] failed to send SMS: {exc}")
        return None


if __name__ == "__main__":
    # Quick manual test: reads recipient/vehicle from the environment.
    send_alert_sms(
        to_number=config.ALERT_TO_NUMBER,
        vehicle_number=config.DEFAULT_VEHICLE_NUMBER,
    )
