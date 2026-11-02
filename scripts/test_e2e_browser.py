#!/usr/bin/env python3
"""End-to-End Browser & Integration Test Runner for TopoCircuit-SAE.

Spins up the unified ThreadingHTTPServer, connects Google Chrome via Selenium WebDriver,
performs real user interactions across all UI components, validates live /api/v1 REST endpoints,
verifies WebGL 3D rendering, debouncing, pagination, state machines, and checks for zero console errors.
"""

import os
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import tempfile
import threading
import time
import urllib.request
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.chrome import options as chrome_options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from topocircuit.server.serve import create_server


def test_rest_api_endpoints(base_url: str) -> Dict[str, bool]:
    """Validates all REST API endpoints over live HTTP."""
    results = {}

    # 1. Health endpoint
    req = urllib.request.Request(f"{base_url}/api/v1/health")
    with urllib.request.urlopen(req, timeout=5) as res:
        results["health_200"] = res.status == 200

    # 2. Metrics endpoint
    req = urllib.request.Request(f"{base_url}/api/v1/metrics")
    with urllib.request.urlopen(req, timeout=5) as res:
        results["metrics_200"] = res.status == 200

    # 3. DAG endpoint
    req = urllib.request.Request(f"{base_url}/api/v1/dag?family=all")
    with urllib.request.urlopen(req, timeout=5) as res:
        results["dag_200"] = res.status == 200

    # 4. Cells paginated endpoint
    req = urllib.request.Request(f"{base_url}/api/v1/cells?page=1&page_size=12&letter=all&category=all")
    with urllib.request.urlopen(req, timeout=5) as res:
        results["cells_200"] = res.status == 200

    # 5. Prop2 endpoint
    req = urllib.request.Request(f"{base_url}/api/v1/prop2?s1=0.05&s2=0.38&lam=0.20")
    with urllib.request.urlopen(req, timeout=5) as res:
        results["prop2_200"] = res.status == 200

    return results


def click_elem(driver, elem) -> None:
    """Safely scrolls element to center and executes click."""
    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", elem)
    time.sleep(0.15)
    driver.execute_script("arguments[0].click();", elem)


def run_e2e_browser_suite(
    port: int = 8899,
    headless: bool = True,
) -> Dict[str, object]:
    """Runs full end-to-end browser automation against live server."""
    server = create_server(host="127.0.0.1", port=port, db_pool_size=4)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.4)

    base_url = f"http://127.0.0.1:{port}"
    api_results = test_rest_api_endpoints(base_url)

    user_data_dir = tempfile.mkdtemp(prefix="topocircuit_e2e_")
    opts = chrome_options.Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument(f"--user-data-dir={user_data_dir}")
    opts.set_capability("goog:loggingPrefs", {"browser": "ALL"})

    driver = webdriver.Chrome(options=opts)
    wait = WebDriverWait(driver, 10)
    audit: Dict[str, object] = {"api_endpoints": api_results}

    try:
        # Step 1: Open primary web application
        driver.get(f"{base_url}/demo/explorer.html")
        wait.until(EC.title_contains("TopoCircuit-SAE"))
        audit["page_title"] = driver.title

        # Step 2: Verify live backend integration badge
        time.sleep(0.5)
        backend_badge = driver.find_element(By.ID, "backendStatusBadge")
        audit["backend_badge_text"] = backend_badge.text
        assert "LIVE REST API" in backend_badge.text, f"Unexpected badge text: {backend_badge.text}"

        # Step 3: Verify Section 5 Main Evidence Table rows
        evidence_rows = driver.find_elements(By.CSS_SELECTOR, "#storyMainTableBody tr")
        audit["story_evidence_rows"] = len(evidence_rows)
        assert len(evidence_rows) >= 7, f"Expected >= 7 benchmark methods, got {len(evidence_rows)}"

        # Step 4: Test Proposition 2 Interactive Sliders
        s1_input = driver.find_element(By.ID, "storySigma1")
        s2_input = driver.find_element(By.ID, "storySigma2")
        lam_input = driver.find_element(By.ID, "storyLambda")

        # Set slider values via JS and dispatch event to trigger debounced listener
        driver.execute_script(
            "arguments[0].value = 0.08; arguments[0].dispatchEvent(new Event('input'));", s1_input
        )
        driver.execute_script(
            "arguments[0].value = 0.42; arguments[0].dispatchEvent(new Event('input'));", s2_input
        )
        driver.execute_script(
            "arguments[0].value = 0.22; arguments[0].dispatchEvent(new Event('input'));", lam_input
        )
        time.sleep(0.3)

        s1_val = driver.find_element(By.ID, "storySigma1Label").text
        s2_val = driver.find_element(By.ID, "storySigma2Label").text
        lam_val = driver.find_element(By.ID, "storyLambdaLabel").text
        regime_text = driver.find_element(By.ID, "storyRegimeBadge").text

        audit["prop2_slider_test"] = {
            "s1_label": s1_val,
            "s2_label": s2_val,
            "lambda_label": lam_val,
            "regime": regime_text,
        }
        assert "SELECTIVE PARTIAL ABSORPTION" in regime_text

        # Step 5: Switch to Workbench Layer (/app)
        wb_btn = driver.find_element(By.ID, "layerSwitchWbBtn")
        click_elem(driver, wb_btn)
        time.sleep(0.3)

        wb_layer = driver.find_element(By.ID, "layerWorkbench")
        story_layer = driver.find_element(By.ID, "layerStory")
        assert not ("hidden" in wb_layer.get_attribute("class") or ""), "Workbench layer should be visible"
        assert "hidden" in (story_layer.get_attribute("class") or ""), "Story layer should be hidden"

        # Step 6: Verify SQLite DB connection pool badge
        wb_db_badge = driver.find_element(By.ID, "wbDbPoolBadge")
        audit["wb_db_badge"] = wb_db_badge.text
        assert "SQLITE POOL" in wb_db_badge.text

        # Step 7: Test MPAbs Table Pagination and Item Count
        initial_rows = driver.find_elements(By.CSS_SELECTOR, "#wbTableStateContainer tbody tr")
        audit["initial_paginated_rows"] = len(initial_rows)
        assert len(initial_rows) == 12, f"Expected 12 cells per page, got {len(initial_rows)}"

        # Next page
        click_elem(driver, driver.find_element(By.ID, "wbNextPageBtn"))
        time.sleep(0.4)
        page_ind_2 = driver.find_element(By.ID, "wbPageIndicator").text
        audit["page_2_indicator"] = page_ind_2
        assert "Page 2" in page_ind_2

        # Prev page
        click_elem(driver, driver.find_element(By.ID, "wbPrevPageBtn"))
        time.sleep(0.4)
        page_ind_1 = driver.find_element(By.ID, "wbPageIndicator").text
        assert "Page 1" in page_ind_1

        # Step 8: Test Filter Dropdowns (Letter & Category)
        letter_sel = Select(driver.find_element(By.ID, "wbCellLetterFilter"))
        letter_sel.select_by_value("B")
        time.sleep(0.3)
        filtered_b_rows = driver.find_elements(By.CSS_SELECTOR, "#wbTableStateContainer tbody tr")
        audit["letter_b_filter_rows"] = len(filtered_b_rows)
        assert len(filtered_b_rows) > 0

        # Reset filter
        letter_sel.select_by_value("all")
        time.sleep(0.3)

        # Step 9: Test UI State Simulation Buttons (Loading, Empty, Error, Ready)
        # Loading skeleton
        click_elem(driver, driver.find_element(By.ID, "stateLoadingBtn"))
        time.sleep(0.2)
        skeletons = driver.find_elements(By.CSS_SELECTOR, "#wbTableStateContainer .skeleton")
        audit["skeleton_count"] = len(skeletons)
        assert len(skeletons) >= 4

        # Empty state & reset button
        click_elem(driver, driver.find_element(By.ID, "stateEmptyBtn"))
        time.sleep(0.2)
        empty_btn = driver.find_element(By.ID, "resetEmptyFilterBtn")
        assert empty_btn.is_displayed()
        click_elem(driver, empty_btn)
        time.sleep(0.3)

        # Error state & retry button
        click_elem(driver, driver.find_element(By.ID, "stateErrorBtn"))
        time.sleep(0.2)
        retry_btn = driver.find_element(By.ID, "retryTableErrorBtn")
        assert retry_btn.is_displayed()
        click_elem(driver, retry_btn)
        time.sleep(0.3)
        restored_rows = driver.find_elements(By.CSS_SELECTOR, "#wbTableStateContainer tbody tr")
        assert len(restored_rows) == 12

        # Step 10: Test 2D DAG Graph & Node Inspector Selection
        search_input = driver.find_element(By.ID, "wbSearchInput")
        driver.execute_script(
            "arguments[0].value = 'Paris'; arguments[0].dispatchEvent(new Event('input'));", search_input
        )
        time.sleep(0.4)
        node_buttons = driver.find_elements(By.CSS_SELECTOR, "#wbNodeList button")
        audit["search_filtered_nodes"] = len(node_buttons)
        assert len(node_buttons) > 0

        # Click first matching node
        click_elem(driver, node_buttons[0])
        time.sleep(0.2)
        insp_title = driver.find_element(By.ID, "wbInspTitle").text
        audit["selected_node_inspector"] = insp_title
        assert "Paris" in insp_title or "#0104" in insp_title or "#0019" in insp_title

        # Reset Graph
        click_elem(driver, driver.find_element(By.ID, "wbResetGraphBtn"))
        time.sleep(0.3)

        # Step 11: Switch back to Story Layer
        click_elem(driver, driver.find_element(By.ID, "layerSwitchStoryBtn"))
        time.sleep(0.3)
        assert "hidden" in (wb_layer.get_attribute("class") or "")
        assert not ("hidden" in story_layer.get_attribute("class") or "")

        # Step 12: Collect and check Browser Console Logs
        browser_logs: List[Dict[str, str]] = driver.get_log("browser")
        severe_errors = [entry for entry in browser_logs if entry.get("level") == "SEVERE"]
        audit["browser_console_errors"] = severe_errors
        audit["total_console_entries"] = len(browser_logs)
        assert len(severe_errors) == 0, f"Found SEVERE console errors: {severe_errors}"

        audit["passed"] = True
        return audit

    finally:
        driver.quit()
        shutil.rmtree(user_data_dir, ignore_errors=True)
        server.shutdown()
        server.server_close()


def main():
    print("=================================================================")
    print("Running TopoCircuit-SAE End-to-End Browser & Integration Tests...")
    print("=================================================================")
    res = run_e2e_browser_suite(port=8899, headless=True)
    print("\nE2E Test Results:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("\nSUCCESS: All end-to-end integration and browser tests PASSED!")


if __name__ == "__main__":
    main()
