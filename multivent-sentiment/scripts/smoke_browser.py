"""Optional real-browser check against a running local API; no synthetic NLP results."""
import argparse
import json

import _bootstrap  # noqa: F401
from playwright.sync_api import sync_playwright

from app.config.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--browser", default="chrome", choices=["chrome", "msedge"])
    args = parser.parse_args()
    report_dir = get_settings().root / "reports"
    report_dir.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1200, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_function("document.getElementById('health').textContent.includes('Pipeline ready')")
        page.get_by_label("Enter Multivent customer feedback").fill("Tuman kadugay gid ya sang caterer pero nami ang food.")
        page.get_by_role("button", name="Analyze feedback").click()
        page.locator("#results").wait_for(state="visible", timeout=60000)
        result = json.loads(page.locator("#json").text_content())
        assert result["provider"] == "rag_weighted_fallback", result["provider"]
        assert result["rag"]["retrieved_examples"]
        assert page.locator(".reference").count() == len(result["rag"]["retrieved_examples"])
        assert not errors, errors
        page.screenshot(path=str(report_dir / "ui-desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(report_dir / "ui-mobile.png"), full_page=True)
        # Dataset/user text must render as text, never as executable markup.
        page.get_by_label("Enter Multivent customer feedback").fill('<img src=x onerror="window.injected=true"> Nami gid.')
        page.get_by_role("button", name="Analyze feedback").click()
        page.locator("#results").wait_for(state="visible", timeout=60000)
        assert page.evaluate("window.injected === undefined")
        assert not errors, errors
        browser.close()
    print("Real Chrome/Edge feedback submission, references, responsive layout, and safe text rendering passed.")


if __name__ == "__main__":
    main()
