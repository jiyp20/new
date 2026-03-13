import requests
from bs4 import BeautifulSoup
import pandas as pd


# --- CASE A: webscraper.io laptops (known, paginated) ---
# BASE_URL = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page="
# PAGES = 20  # how many pages to scrape

# --- CASE B: unknown site ---
# BASE_URL = "https://YOUR-SITE-HERE?page="
# PAGES = 1   # start with 1 page, increase once it works

# response = requests.get("https://YOUR-SITE-HERE").text
# soup = BeautifulSoup(response, 'html.parser')
#
# all_items = soup.find_all('div', class_='YOUR-CARD-CLASS')   
#
# print(f"Found {len(all_items)} items") 
# print(all_items[0])

results = []

# --- CASE A: webscraper.io laptops ---
# BASE_URL = "https://webscraper.io/test-sites/e-commerce/static/computers/laptops?page="
# PAGES = 20
# for i in range(1, PAGES + 1):
#     response = requests.get(BASE_URL + str(i)).text
#     soup = BeautifulSoup(response, 'html.parser')
#     all_items = soup.find_all('div', class_='col-md-4 col-xl-4 col-lg-4')
#     for item in all_items:
#         results.append({
#             "product_name": item.find('a', class_='title').text.strip(),
#             "price":        item.find('h4', class_='price').text.strip(),
#             "description":  item.find('p', class_='description card-text').text.strip(),
#             "review":       item.find('p', class_='review-count float-end').text.strip(),
#         })

# BASE_URL = "https://YOUR-SITE-HERE?page="
# PAGES = 5
# for i in range(1, PAGES + 1):
#     response = requests.get(BASE_URL + str(i)).text
#     soup = BeautifulSoup(response, 'html.parser')
#     all_items = soup.find_all('div', class_='YOUR-CARD-CLASS')
#     for item in all_items:
#         results.append({
#             "title":  item.find('YOUR-TAG', class_='YOUR-CLASS').text.strip(),
#             "price":  item.find('YOUR-TAG', class_='YOUR-CLASS').text.strip(),
#             # add more fields as needed
#         })

# --- SAFE SCRAPING TIP ---
# If a field might be missing on some pages, use:
# name_tag = item.find('a', class_='title')
# name = name_tag.text.strip() if name_tag else "Unknown"

df = pd.DataFrame(results)
print(df.head())
print(df.info())

# ============================================================
# STEP 4: CLEANING — fully automatic, handles everything
# ============================================================

before = len(df)
print(f"\nTotal rows before cleaning: {before}")

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Strip whitespace from all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 3. Fill missing text columns with "Unknown"
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna("Unknown")

# 4. Clean price — remove $, £, € symbols and convert to float
if 'price' in df.columns:
    df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

# 5. Fill missing numeric columns with 0
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(0)

# 6. Remove rows where price is missing or <= 0
if 'price' in df.columns:
    df = df[df['price'].notna()]
    df = df[df['price'] > 0]

if 'price' in df.columns:
    p95 = df['price'].quantile(0.95)
    df = df[df['price'] <= p95]

print(f"Total rows after cleaning: {len(df)}")
print(f"Rows removed: {before - len(df)}")

df.to_csv("scraped_output.csv", index=False)
print("Saved to scraped_output.csv")




import requests
import pandas as pd

# ============================================================
# STEP 1: FETCH DATA
# ============================================================

# --- CASE A: PRODUCT API (dummyjson style — returns a list) ---
# API_URL = "https://dummyjson.com/products?limit=100"
# response = requests.get(API_URL, timeout=60)
# raw_data = response.json()
# print(type(raw_data))         
# print(raw_data.keys())        
# items = raw_data['products'] 
# print(f"Total products: {len(items)}")
# print("First 3 products:", [i['title'] for i in items[:3]])

# --- CASE B: COVID API (nested dict — state -> dates -> metrics) ---
# API_URL = "https://data.covid19india.org/v4/timeseries.json"
# response = requests.get(API_URL, timeout=60)
# raw_data = response.json()
# print(type(raw_data))         # dict
# print(list(raw_data.keys())[:5])  # ['AN', 'AP', 'AR', ...]  state codes
# STATE = "MH"
# items = raw_data[STATE]['dates']  # dict of date -> metrics

# --- CASE C: UNKNOWN API — run this first to explore ---
API_URL = "https://YOUR-API-URL-HERE"
response = requests.get(API_URL, timeout=60)
raw_data = response.json()

print(type(raw_data))             
if isinstance(raw_data, dict):
    print(raw_data.keys())        
if isinstance(raw_data, list):
    print(raw_data[0])            
if isinstance(raw_data, dict):
    first_key = list(raw_data.keys())[0]
    print(raw_data[first_key])    

# After printing above, you'll know the structure
# Then build rows like CASE A or CASE B below

# ============================================================
# STEP 2: BUILD ROWS — uncomment the one that matches
# ============================================================

rows = []

# --- CASE A: list of items (product style) ---
# for item in items:
#     rows.append({
#         "id":                   item.get("id", ""),
#         "title":                item.get("title", ""),
#         "category":             item.get("category", ""),
#         "brand":                item.get("brand", "Unknown"),
#         "price":                item.get("price", 0),
#         "discountPercentage":   item.get("discountPercentage", 0),
#         "rating":               item.get("rating", 0),
#         "stock":                item.get("stock", 0),
#     })

# --- CASE B: nested dict (covid style) ---
# for date_str, metrics in items.items():
#     total = metrics.get("total", {})
#     rows.append({
#         "date":             date_str,
#         "confirmed_total":  total.get("confirmed", 0),
#         "recovered_total":  total.get("recovered", 0),
#         "deceased":         total.get("deceased", 0),
#         "tested_total":     total.get("tested", 0),
#         "vaccinated1":      total.get("vaccinated1", 0),
#         "vaccinated2":      total.get("vaccinated2", 0),
#     })

# --- CASE C: unknown — once you know the structure, copy CASE A or B above ---

df = pd.DataFrame(rows)
print(df.head())
print(df.info())

# ============================================================
# STEP 3: CLEANING — fully automatic, handles everything
# ============================================================

before = len(df)
print(f"\nTotal rows before cleaning: {before}")

# 1. Remove duplicate rows
df = df.drop_duplicates()

# 2. Strip whitespace from all string columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# 3. Fill missing text columns with "Unknown"
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna("Unknown")

# 4. Fill missing numeric columns with 0
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(0)

# 5. Remove rows where price is missing or <= 0
if 'price' in df.columns:
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # handle "$23.99" style
    df = df[df['price'].notna()]
    df = df[df['price'] > 0]

# 6. Remove price outliers above 95th percentile
if 'price' in df.columns:
    p95 = df['price'].quantile(0.95)
    df = df[df['price'] <= p95]

# 7. Add final price after discount (uncomment if needed)
# if 'discountPercentage' in df.columns:
#     df['final_price'] = round(df['price'] * (1 - df['discountPercentage'] / 100), 2)

# 8. Add rating bucket (uncomment if needed)
# if 'rating' in df.columns:
#     def rating_bucket(r):
#         if r >= 4.5:   return "High"
#         elif r >= 3.5: return "Medium"
#         else:          return "Low"
#     df['rating_bucket'] = df['rating'].apply(rating_bucket)

print(f"Total rows after cleaning: {len(df)}")
print(f"Rows removed: {before - len(df)}")

# ============================================================
# STEP 4: SAVE CSV
# ============================================================

df.to_csv("output.csv", index=False)
print("Saved to output.csv")