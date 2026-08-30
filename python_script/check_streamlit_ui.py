"""Run real-browser desktop and mobile checks against the Streamlit app."""

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_URL = os.getenv("STREAMLIT_URL", "http://127.0.0.1:8510")
IMAGE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "EuroSAT_RGB" / "2750" / "River" / "River_100.jpg"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "reports" / "ui"
VIEWPORTS = {
    "desktop": {"width": 1440, "height": 900},
    "mobile": {"width": 390, "height": 844},
}


def check_ui(app_url=APP_URL):
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        for name, viewport in VIEWPORTS.items():
            page = browser.new_page(viewport=viewport)
            browser_errors = []
            page.on(
                "pageerror",
                lambda error, errors=browser_errors: errors.append(str(error)),
            )
            page.goto(app_url, wait_until="networkidle", timeout=60_000)
            page.get_by_role("heading", name="EuroSAT RGB Classifier").wait_for()
            page.locator('input[type="file"]').set_input_files(IMAGE_PATH)
            page.get_by_role("button", name="Classify image").click()
            confidence_metric = page.locator('[data-testid="stMetricLabel"]').filter(
                has_text="Confidence"
            )
            confidence_metric.wait_for(timeout=60_000)
            page.wait_for_timeout(1500)

            horizontal_overflow = page.evaluate(
                "document.documentElement.scrollWidth > window.innerWidth + 1"
            )
            screenshot_path = OUTPUT_DIRECTORY / f"streamlit-{name}.png"
            page.screenshot(path=screenshot_path, full_page=True)
            confidence_metric.scroll_into_view_if_needed()
            prediction_screenshot_path = (
                OUTPUT_DIRECTORY / f"streamlit-{name}-prediction.png"
            )
            page.screenshot(path=prediction_screenshot_path)
            result = {
                "viewport": name,
                "size": viewport,
                "prediction_visible": confidence_metric.is_visible(),
                "horizontal_overflow": horizontal_overflow,
                "browser_errors": browser_errors,
                "screenshot": str(screenshot_path),
                "prediction_screenshot": str(prediction_screenshot_path),
            }
            if browser_errors or horizontal_overflow:
                raise RuntimeError(f"Browser check failed: {result}")
            results.append(result)
            page.close()
        browser.close()

    return results


def main():
    print(json.dumps(check_ui(), indent=2))


if __name__ == "__main__":
    main()
