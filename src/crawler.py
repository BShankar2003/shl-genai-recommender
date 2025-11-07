"""
crawler.py
------------------------------------------------
Crawls SHL product catalog to build dataset.
------------------------------------------------
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
print("🕷️ Crawling SHL Catalog...")

headers = {"User-Agent": "Mozilla/5.0"}
resp = requests.get(BASE_URL, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")

data = []
for link in soup.select("a[href*='/view/']"):
    name = link.text.strip()
    url = link["href"]
    if "product-catalog/view/" in url:
        data.append({"Assessment Name": name, "Assessment_url": url})

df = pd.DataFrame(data).drop_duplicates()
df.to_csv("data/shl_catalog.csv", index=False)
print(f"✅ Saved {len(df)} assessments to data/shl_catalog.csv")
