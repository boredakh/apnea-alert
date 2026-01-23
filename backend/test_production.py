"""
Test the API in production-like mode
"""
import requests
import time

def test_production():
    print("🧪 Production-like testing...")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    tests = [
        ("GET /", "API root"),
        ("GET /health", "Health check"),
        ("GET /model/info", "Model info"),
        ("POST /predict", "Prediction")
    ]
    
    for endpoint, description in tests:
        print(f"\n🔍 Testing {description} ({endpoint})...")
        
        try:
            if endpoint == "POST /predict":
                # Test prediction
                test_data = {
                    "mean": -0.0003,
                    "std": 0.35,
                    "min": -1.2,
                    "max": 2.1,
                    "range": 3.3,
                    "rms": 0.32,
                    "skewness": 3.5,
                    "kurtosis": 18.5,
                    "hr_mean": 68.5,
                    "hr_std": 8.2,
                    "rr_mean": 0.88,
                    "rr_std": 0.12,
                    "lf_power": 0.00015,
                    "hf_power": 0.00008,
                    "lf_hf_ratio": 1.87
                }
                
                start = time.time()
                response = requests.post(
                    f"{base_url}/predict",
                    json=test_data,
                    timeout=10
                )
                elapsed = time.time() - start
                
            else:
                # Test GET endpoints
                start = time.time()
                response = requests.get(
                    f"{base_url}{endpoint.split(' ')[1]}",
                    timeout=5
                )
                elapsed = time.time() - start
            
            print(f"   Status: {response.status_code}")
            print(f"   Time: {elapsed:.3f}s")
            
            if response.status_code == 200:
                if endpoint == "POST /predict":
                    data = response.json()
                    print(f"   ✅ Prediction: {data.get('prediction_label')}")
                    print(f"   📊 Probability: {data.get('apnea_probability'):.3f}")
                print("   ✅ Success")
            else:
                print(f"   ❌ Error: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            print("   ❌ Timeout - too slow")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Production test complete!")
    
    # Performance summary
    print("\n📊 Performance expectations:")
    print("   Health check: < 100ms")
    print("   Prediction: < 500ms")
    print("   Model loading: < 5s on cold start")

if __name__ == "__main__":
    # First check if API is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            test_production()
        else:
            print("❌ API not responding properly")
    except:
        print("❌ API is not running. Please start it first:")
        print("   python main.py")