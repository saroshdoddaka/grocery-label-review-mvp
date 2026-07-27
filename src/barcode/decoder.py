def decode_image(path: str, image_order: int = 1) -> tuple[list[dict], list[str]]:
    supported = {"UPC-A", "UPC-E", "EAN-8", "EAN-13"}
    try:
        from pyzbar.pyzbar import decode
        from PIL import Image
        raw = decode(Image.open(path)); out = []; warnings = []
        for item in raw:
            fmt = item.type.decode() if isinstance(item.type, bytes) else str(item.type)
            value = item.data.decode(errors="replace")
            out.append({"value": value, "format": fmt, "image_order": image_order, "confidence": None, "supported": fmt in supported, "raw": {"type": fmt, "data": value, "rect": str(item.rect), "polygon": str(item.polygon)}})
        return out, warnings
    except (ImportError, OSError):
        try:
            import zxingcpp
            from PIL import Image
            out = []
            for item in zxingcpp.read_barcodes(Image.open(path)):
                fmt = str(item.format).replace("BarcodeFormat.", "").replace("_", "-")
                fmt = {"UPCA": "UPC-A", "UPCE": "UPC-E", "EAN8": "EAN-8", "EAN13": "EAN-13"}.get(fmt.replace("-", ""), fmt)
                value = item.text or ""
                out.append({"value": value, "format": fmt, "image_order": image_order, "confidence": None, "supported": fmt in supported, "raw": {"format": fmt, "text": value}})
            return out, []
        except Exception as exc:
            return [], [f"Barcode decoder unavailable: {exc}"]
    except Exception as exc:
        return [], [f"Barcode decoding failed: {exc}"]
