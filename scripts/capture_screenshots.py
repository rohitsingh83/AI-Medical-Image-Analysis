"""
capture_screenshots.py - Automated high-resolution screenshots generator
"""
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

SCREENSHOTS_DIR = Path(__file__).resolve().parents[1] / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

opts = Options()
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1440,1080")
opts.add_argument("--disable-gpu")
opts.add_argument("--no-sandbox")

driver = webdriver.Chrome(options=opts)

try:
    # 1. GitHub Pages Website Hero & Metrics
    print("[1/7] Capturing Website Hero & Metrics...")
    driver.get("http://127.0.0.1:8080/")
    time.sleep(2)
    driver.set_window_size(1440, 960)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "01_website_hero_and_metrics.png"))

    # 2. Interactive Diagnostic Studio - Normal Case
    print("[2/7] Capturing Diagnostic Studio - Case 1 Normal...")
    driver.execute_script("document.getElementById('studio').scrollIntoView();")
    time.sleep(1)
    driver.find_element(By.ID, "caseBtn1").click()
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "02_diagnostic_studio_normal_case.png"))

    # 3. Interactive Diagnostic Studio - Bacterial Pneumonia with Grad-CAM
    print("[3/7] Capturing Diagnostic Studio - Case 2 Pneumonia with Grad-CAM...")
    driver.find_element(By.ID, "caseBtn2").click()
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "03_diagnostic_studio_pneumonia_gradcam.png"))

    # 4. Automated Radiology Report
    print("[4/7] Capturing Automated Radiology Report...")
    driver.execute_script("document.getElementById('report').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "04_automated_radiology_report.png"))

    # 5. Clinical Confusion Matrix & Benchmarks
    print("[5/7] Capturing Clinical Benchmarks...")
    driver.execute_script("document.getElementById('metrics').scrollIntoView();")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "05_clinical_confusion_matrix_and_benchmarks.png"))

    # 6. Flask Doctor App Upload Page
    print("[6/7] Capturing Flask App Upload Interface...")
    driver.get("http://127.0.0.1:5000/")
    time.sleep(1)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "06_flask_app_upload_interface.png"))

    # 7. Flask App Prediction Result Page
    print("[7/7] Submitting sample image to Flask app and capturing result...")
    sample_file = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "chest_xray" / "test" / "PNEUMONIA" / "pneumonia_0000.png"
    file_input = driver.find_element(By.NAME, "image")
    file_input.send_keys(str(sample_file))
    time.sleep(0.5)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(2)
    driver.save_screenshot(str(SCREENSHOTS_DIR / "07_flask_app_diagnosis_result.png"))

    print("All screenshots captured successfully in:", SCREENSHOTS_DIR)

finally:
    driver.quit()
