import requests
import os
import sys

def main():
    print("=== STARTING API ENDPOINT TESTS ON PORT 8080 ===")
    
    base_url = "http://127.0.0.1:8080"
    
    # 1. Test GET / (HTML Dashboard page)
    print("1. Testing GET / (Root HTML)...")
    try:
        r = requests.get(base_url)
        if r.status_code == 200 and "ProdVision" in r.text:
            print("   [PASS] Root page loaded successfully and contains 'ProdVision'.")
        else:
            print(f"   [FAIL] Root page returned status {r.status_code}.")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] Failed to request root page: {e}")
        sys.exit(1)
        
    # 2. Test POST /upload (Upload Image)
    print("2. Testing POST /upload...")
    image_path = os.path.join("uploads", "milk_label.png")
    if not os.path.exists(image_path):
        print(f"   [FAIL] Test image not found at {image_path}")
        sys.exit(1)
        
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/png")}
            r = requests.post(f"{base_url}/upload", files=files)
            
        if r.status_code == 200:
            data = r.json()
            print("   [PASS] Upload completed successfully.")
            print(f"   Returned Product ID: {data.get('id')}")
            print(f"   Attributes Dict: {data.get('attributes')}")
            print(f"   Confidence Score: {data.get('confidence_score')}%")
            product_id = data.get("id")
        else:
            print(f"   [FAIL] Upload returned status {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] Failed to upload image: {e}")
        sys.exit(1)
        
    # 3. Test GET /products (Retrieve History)
    print("3. Testing GET /products (History)...")
    try:
        r = requests.get(f"{base_url}/products")
        if r.status_code == 200:
            products = r.json()
            print(f"   [PASS] Retrieved history containing {len(products)} records.")
            # Check if our uploaded product is in the history list
            found = False
            for p in products:
                if p["id"] == product_id:
                    found = True
                    break
            if found:
                print("   [PASS] Uploaded product found in history list.")
            else:
                print(f"   [FAIL] Uploaded product ID {product_id} not found in history list.")
                sys.exit(1)
        else:
            print(f"   [FAIL] History returned status {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] Failed to retrieve history: {e}")
        sys.exit(1)
        
    # 4. Test GET /products/{id}/json (Download JSON)
    print(f"4. Testing GET /products/{product_id}/json...")
    try:
        r = requests.get(f"{base_url}/products/{product_id}/json")
        if r.status_code == 200:
            data = r.json()
            print("   [PASS] JSON downloaded successfully.")
            print(f"   JSON Content: {data}")
        else:
            print(f"   [FAIL] JSON download returned status {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] Failed to download JSON: {e}")
        sys.exit(1)
        
    # 5. Test GET /products/{id}/pdf (Download PDF)
    print(f"5. Testing GET /products/{product_id}/pdf...")
    try:
        r = requests.get(f"{base_url}/products/{product_id}/pdf")
        if r.status_code == 200:
            content_type = r.headers.get("content-type", "")
            if "pdf" in content_type.lower():
                print("   [PASS] PDF downloaded successfully (Content-Type verified).")
                # Save locally to verify file size
                temp_pdf_path = f"test_output_{product_id}.pdf"
                with open(temp_pdf_path, "wb") as pdf_file:
                    pdf_file.write(r.content)
                size = os.path.getsize(temp_pdf_path)
                print(f"   [PASS] PDF saved to {temp_pdf_path} (Size: {size} bytes).")
                os.remove(temp_pdf_path) # Cleanup
            else:
                print(f"   [FAIL] PDF returned invalid content type: {content_type}")
                sys.exit(1)
        else:
            print(f"   [FAIL] PDF download returned status {r.status_code}: {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"   [FAIL] Failed to download PDF: {e}")
        sys.exit(1)
        
    print("\n=== ALL API ENDPOINT TESTS PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
