import requests
from src.config import USER_AGENT

def lookup(barcode: str) -> dict:
    if not barcode: return {"status": "NOT_ATTEMPTED"}
    try:
        response = requests.get(f"https://world.openfoodfacts.org/api/v2/product/{barcode}.json", headers={"User-Agent": USER_AGENT}, timeout=(3, 8))
        if response.status_code == 429: return {"status": "RATE_LIMITED"}
        if response.status_code >= 500: return {"status": "NETWORK_ERROR"}
        data = response.json()
        if data.get("status") != 1: return {"status": "NOT_FOUND", "raw": data}
        product = data.get("product", {})
        return {"status": "FOUND", "product_name": product.get("product_name") or product.get("product_name_en", ""), "brand": product.get("brands", ""), "category": product.get("categories", ""), "quantity": product.get("quantity", ""), "image_url": product.get("image_url", ""), "raw": data}
    except requests.RequestException: return {"status": "NETWORK_ERROR"}
    except (ValueError, TypeError): return {"status": "PROVIDER_ERROR"}
