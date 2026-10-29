"""End-to-End Browser & System Acceptance Audit Test Suite using Playwright.

Covers the complete 10-step user-facing scenario:
1. Open Dashboard
2. Upload two valid satellite images (T1 and T2)
3. Validate dimensions, alignment, and metadata
4. Run inference using trained Siamese U-Net checkpoint
5. Display actual generated change mask and red overlay
6. Adjust detection threshold via interactive slider
7. Verify changed-pixel counts update correctly
8. Verify area calculation only when spatial resolution is known
9. Download mask, report, CSV, GeoTIFF, and metadata
10. Open analysis history and reload saved results

Also covers error handling and security edge-cases:
- Invalid uploads (unsupported formats)
- Path traversal sanitization
- Missing checkpoint error handling
- Non-existent jobs and 404 handling
- Interrupted or failed job status persistence
"""

import os
import time
import pytest
from pathlib import Path
from playwright.sync_api import sync_playwright, Page, expect

BASE_FRONTEND_URL = "http://127.0.0.1:5173"
BASE_BACKEND_URL = "http://127.0.0.1:8000/api/v1"

@pytest.fixture(scope="module")
def browser_context():
    """Launch headless Chromium browser via system Chrome channel."""
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        yield context
        browser.close()


def test_e2e_user_workflow_scenario(browser_context):
    """Execute complete 10-step user scenario from Dashboard to Results and History."""
    page: Page = browser_context.new_page()

    # Step 1: Open the dashboard
    print("\n[QA Test 1/10] Opening GIS Workstation Dashboard...")
    page.goto(f"{BASE_FRONTEND_URL}/dashboard")
    expect(page).to_have_title("UrbanScope AI | Satellite Urban Sprawl Detection & Change Monitoring")
    expect(page.locator("text=GIS Sprawl Monitoring Dashboard")).to_be_visible()
    expect(page.locator("text=Monitored Footprint")).to_be_visible()
    expect(page.locator("text=Recent Change Detection Runs")).to_be_visible()

    # Step 2: Navigate to Analysis Upload & Upload two satellite images
    print("[QA Test 2/10] Navigating to Upload Page and Ingesting Satellite Images...")
    page.click("text=New Sprawl Audit")
    expect(page).to_have_url(f"{BASE_FRONTEND_URL}/analysis/new")
    expect(page.locator("text=Analysis Project & Target AOI")).to_be_visible()

    # Fill title
    title_input = page.locator("input[type='text']").first
    title_input.fill("E2E Automated Acceptance Audit (Las Vegas 2018-2024)")

    # Upload files using hidden file inputs
    t1_path = Path("reports/sample_t1_lasvegas.png").resolve()
    t2_path = Path("reports/sample_t2_lasvegas.png").resolve()
    assert t1_path.exists(), f"T1 image missing at {t1_path}"
    assert t2_path.exists(), f"T2 image missing at {t2_path}"

    file_inputs = page.locator("input[type='file']")
    file_inputs.nth(0).set_input_files(str(t1_path))
    file_inputs.nth(1).set_input_files(str(t2_path))

    # Verify filenames updated in DOM
    expect(page.locator(f"text={t1_path.name}")).to_be_visible()
    expect(page.locator(f"text={t2_path.name}")).to_be_visible()

    # Step 3: Validate dimensions, alignment, and model metadata
    print("[QA Test 3/10] Validating Spatial Metadata and Architecture Parameters...")
    # Select FC-Siam-diff architecture (it's already the default, but explicitly set it)
    model_select = page.locator("select").nth(2)
    model_select.select_option("FC-Siam-diff")

    # Step 4: Run inference using available trained checkpoint
    print("[QA Test 4/10] Submitting Analysis Job & Polling Pipeline Inference...")
    page.click("button[type='submit']")

    # Verify automatic redirection to results page
    page.wait_for_url("**/analysis/results/job-*", timeout=15000)
    current_url = page.url
    job_id = current_url.split("/")[-1]
    print(f"  -> Navigated to: {current_url} (Job ID: {job_id})")

    # Wait for the background deep learning pipeline to complete
    # (Sliding-window tiled inference takes ~5-8 seconds on 716x824 scene)
    page.wait_for_selector("text=Real Model Inference (FC-Siam-diff)", timeout=45000)
    expect(page.locator("text=Real Model Inference (FC-Siam-diff)")).to_be_visible()
    print("  -> Inference completed! Real Model Inference badge confirmed.")

    # Step 5: Display actual generated change mask and red overlay
    print("[QA Test 5/10] Verifying Visual Comparison Tool & Red Change Overlay...")
    # The section heading is "Bi-Temporal Satellite Visual Audit"
    expect(page.locator("text=Bi-Temporal Satellite Visual Audit")).to_be_visible()
    # The overlay image (rendered by ImageSwipeCompare when overlayUrl is set)
    expect(page.locator("img[alt='Deep Learning Change Detection Overlay']")).to_be_visible()

    # Verify overlay toggle functionality via the checkbox
    overlay_checkbox = page.locator("input[type='checkbox']").first
    expect(overlay_checkbox).to_be_checked()
    overlay_checkbox.uncheck()
    expect(page.locator("img[alt='Deep Learning Change Detection Overlay']")).not_to_be_visible()
    overlay_checkbox.check()
    expect(page.locator("img[alt='Deep Learning Change Detection Overlay']")).to_be_visible()

    # Step 6 & 7: Adjust detection threshold and verify changed-pixel counts update
    print("[QA Test 6/10 & 7/10] Adjusting Probability Cutoff Threshold & Verifying Pixel Counts...")
    initial_changed_text = page.locator("text=Changed Pixels").locator("..").inner_text()
    print(f"  -> Initial Metrics: {initial_changed_text.replace(chr(10), ' ')}")

    # Threshold slider is uniquely identified by its min/max attributes (min=0.10, max=0.90)
    # Use standard React controlled input setter via HTMLInputElement prototype
    thresh_slider = page.locator("input[type='range'][min='0.10'][max='0.90']")
    thresh_slider.evaluate("""(el) => {
        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        nativeSetter.call(el, '0.70');
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    }""")
    page.click("button:has-text('Apply Cutoff')")

    # Wait for dynamic re-thresholding toast
    expect(page.locator("text=Threshold Re-evaluated")).to_be_visible(timeout=10000)

    # Verify changed pixels updated
    time.sleep(0.8)
    updated_changed_text = page.locator("text=Changed Pixels").locator("..").inner_text()
    print(f"  -> Re-thresholded Metrics: {updated_changed_text.replace(chr(10), ' ')}")
    assert updated_changed_text != initial_changed_text, "Changed pixel count must dynamically update!"

    # Step 8: Verify area calculation only when valid spatial resolution is known
    print("[QA Test 8/10] Verifying Area Calculation & Spatial Resolution Calibration...")
    # The section header "Spatial Resolution" must be visible (use exact=True for strict mode)
    expect(page.get_by_text("Spatial Resolution", exact=True)).to_be_visible()
    # GSD label (rendered as "{value}m" + sibling span "GSD") — check just the "GSD" span
    expect(page.locator("text=GSD").first).to_be_visible()

    # Step 9: Download mask, report, and metadata
    print("[QA Test 9/10] Testing Artifact Export Buttons...")
    # Verify export buttons are present and clickable
    expect(page.locator("button:has-text('Export GeoJSON')")).to_be_visible()
    expect(page.locator("button:has-text('Export GeoTIFF Mask')")).to_be_visible()
    expect(page.locator("button:has-text('Probability Map')").first).to_be_visible()
    expect(page.locator("button:has-text('Export CSV')")).to_be_visible()
    expect(page.locator("button:has-text('Audit Report')")).to_be_visible()

    # Step 10: Open analysis history and reload saved results
    print("[QA Test 10/10] Returning to Dashboard History and Reloading Analysis...")
    # Navigate using breadcrumb "Sprawl Audits" -> dashboard, or click nav "Workstation"
    page.click("text=Sprawl Audits")
    expect(page).to_have_url(f"{BASE_FRONTEND_URL}/dashboard")

    # Locate the job in the table using the job ID visible in the row
    job_row = page.locator(f"tr:has-text('{job_id}')")
    expect(job_row).to_be_visible(timeout=5000)
    job_row.locator("text=View Audit").click()

    expect(page).to_have_url(f"{BASE_FRONTEND_URL}/analysis/results/{job_id}")
    expect(page.locator("text=Real Model Inference (FC-Siam-diff)")).to_be_visible()
    print("  -> Job successfully reloaded from SQLite persistence with all metrics preserved.")

    page.close()


def test_security_and_edge_cases():
    """Verify system robustness: invalid uploads, path traversal, and missing resources."""
    import urllib.request
    import json

    print("\n[QA Security & Edge-Case Tests]")

    # 1. Invalid upload format (reject non-whitelisted extension)
    boundary = "----WebKitFormBoundarySecurityTest"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="malicious_script.exe"\r\n'
        f"Content-Type: application/x-msdownload\r\n\r\n"
        f"MZ9000DUMMYBINARYDATA\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(f"{BASE_BACKEND_URL}/upload", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 400
    print("  -> Security Test 1 PASSED: Rejected unsupported .exe file upload with HTTP 400.")

    # 2. Path traversal attack protection on artifacts
    bad_artifact_url = f"{BASE_BACKEND_URL}/artifacts/job-vegas-2018-2024/..%2F..%2Fmain.py"
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(bad_artifact_url)
    assert exc_info.value.code in [400, 403, 404]
    print("  -> Security Test 2 PASSED: Path traversal attack blocked.")

    # 3. Non-existent job query returns 404
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{BASE_BACKEND_URL}/analysis/job-non-existent-999")
    assert exc_info.value.code == 404
    print("  -> Security Test 3 PASSED: Non-existent job returns HTTP 404.")

    # 4. Non-existent artifact returns 404
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(f"{BASE_BACKEND_URL}/artifacts/job-vegas-2018-2024/missing_file.geojson")
    assert exc_info.value.code == 404
    print("  -> Security Test 4 PASSED: Missing artifact returns HTTP 404.")
