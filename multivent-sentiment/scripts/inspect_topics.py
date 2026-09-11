import _bootstrap  # noqa: F401
from app.config.settings import get_settings
from app.topics.lda_service import LDAService

if __name__ == "__main__":
    settings = get_settings()
    service = LDAService.load(settings.lda_dir, settings.enable_pii_sanitization)
    for topic in service.all_topics():
        print(f"Topic {topic['id']} ({topic['name']}):\n{', '.join(topic['keywords'])}\n")
