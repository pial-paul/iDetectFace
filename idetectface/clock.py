"""Attendance dates and times are defined by the calendar in Bangladesh."""

from datetime import datetime
from zoneinfo import ZoneInfo

DHAKA = ZoneInfo("Asia/Dhaka")


def bangladesh_time(instant: datetime | None = None) -> datetime:
    """Convert an instant to Dhaka time. Naive inputs represent Bangladesh wall time."""
    if instant is None:
        return datetime.now(DHAKA)
    if instant.tzinfo is None:
        return instant.replace(tzinfo=DHAKA)
    return instant.astimezone(DHAKA)
