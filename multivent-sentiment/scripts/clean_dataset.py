import json

import _bootstrap  # noqa: F401
from app.config.settings import get_settings
from app.dataset import clean_download

if __name__ == "__main__":
    settings = get_settings()
    report = clean_download(settings.root / "data/raw/hilisenti", settings.processed,
                            settings.enable_pii_sanitization)
    print(json.dumps(report, indent=2))
