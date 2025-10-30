from datetime import datetime


def current_timestamp() -> str:
    """년월일시분초(YYYYMMDDHHMMSS) 포맷의 타임스탬프를 반환한다."""
    return datetime.now().strftime("%Y%m%d%H%M%S")
