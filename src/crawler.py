import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

INPUT_PATH = "data/Gen_AI Dataset.xlsx"  # or .csv
OUTPUT_PATH = "data/shl_catalog_clean.csv"
SLEEP_TIME = 2                           # Try 2 seconds (reduce for speed, increase if failures)

if INPUT_PATH.endswith(".xlsx"):
    df = pd.read_excel(INPUT_PATH)
else:
    df = pd.read_csv(INPUT_PATH)

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

results = []

def safe_find(driver, by, value):
    try:
        return driver.find_element(by, value).text.strip()
    except Exception:
        return ""

print("Starting crawl...")
for idx, row in df.iterrows():
    url = row.get('Assessment_url') or row.get('Assessment URL') or row.get('url', '')
    if not isinstance(url, str) or not url.strip():
        print(f"Row {idx} — Missing URL, skipped.")
        continue

    print(f"Crawling {idx+1}/{len(df)}: {url}")
    try:
        driver.get(url)
        time.sleep(SLEEP_TIME)

        blocks = driver.find_elements(By.CSS_SELECTOR, "div.product-catalogue-training-calendar__row")
        description = ""
        job_level = ""
        for block in blocks:
            h4_elems = block.find_elements(By.TAG_NAME, "h4")
            if not h4_elems:
                continue
            h4 = h4_elems[0].text.strip().lower()
            if "description" in h4:
                description = block.find_element(By.TAG_NAME, "p").text.strip()
            if "job levels" in h4:
                job_level = block.find_element(By.TAG_NAME, "p").text.strip()
        name = safe_find(driver, By.TAG_NAME, "h1")

        results.append({
            "Assessment Name": name,
            "Assessment_url": url,
            "Description": description,
            "Category": "",          # Fill this if needed with more selectors
            "Test Type": job_level
        })

        print(f"✓ [{idx+1}] {name[:30]}... | Desc: {description[:30]}... | Job: {job_level}")

    except Exception as e:
        print(f"Failed {url[:60]}: {e}")
        continue

driver.quit()
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"\nDone: {len(results)} rows written to {OUTPUT_PATH}")
