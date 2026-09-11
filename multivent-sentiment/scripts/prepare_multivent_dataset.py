"""Download reviewed sources and prepare the binary Multivent corpus."""
import _bootstrap  # noqa: F401
from app.config.settings import get_settings
from app.corpus.prepare import prepare_multivent

if __name__ == "__main__":
    manifest = prepare_multivent(get_settings())
    print("Corpus sizes:", manifest["sizes"])
    print("Neutral rows excluded:", manifest["neutral_excluded"])
