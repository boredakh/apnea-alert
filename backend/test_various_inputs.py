"""
Test various input values to see different predictions
"""
import requests

print("🔬 Testing various input scenarios...")

# Different test cases
test_cases = [
    {
        "name": "High Risk Example",
        "data": {
            "mean": -0.0003, "std": 0.35, "min": -1.2, "max": 2.1, "range": 3.3,
            "rms": 0.32, "skewness": 3.5, "kurtosis": 18.5,
            "hr_mean": 68.5, "hr_std": 8.2, "rr_mean": 0.88, "rr_std": 0.12,
            "lf_power": 0.00015, "hf_power": 0.00008, "lf_hf_ratio": 1.87
        }
    },
    {
        "name": "Lower Risk Example",
        "data": {
            "mean": 0.001, "std": 0.25, "min": -0.8, "max": 1.5, "range": 2.3,
            "rms": 0.25, "skewness": 2.0, "kurtosis": 12.0,
            "hr_mean": 65.0, "hr_std": 5.0, "rr_mean": 0.92, "rr_std": 0.08,
            "lf_power": 0.00008, "hf_power": 0.00012, "lf_hf_ratio": 0.67
        }
    },
    {
        "name": "Very Low Risk Example", 
        "data": {
            "mean": 0.002, "std": 0.15, "min": -0.5, "max": 1.0, "range": 1.5,
            "rms": 0.18, "skewness": 1.2, "kurtosis": 8.0,
            "hr_mean": 62.0, "hr_std": 3.0, "rr_mean": 0.97, "rr_std": 0.05,
            "lf_power": 0.00005, "hf_power": 0.00015, "lf_hf_ratio": 0.33
        }
    }
]

for test in test_cases:
    print(f"\n🧪 Testing: {test['name']}")
    print("-" * 40)
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test["data"],
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            prob = result.get('apnea_probability', 0)
            label = result.get('prediction_label', 'Unknown')
            
            # Create a visual bar
            bar_length = 30
            filled = int(prob * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            print(f"   Prediction: {label}")
            print(f"   Probability: {prob:.3f}")
            print(f"   Visual: [{bar}]")
            print(f"   Risk: {result.get('risk_level')}")
        else:
            print(f"   ❌ Error: Status {response.status_code}")
            print(f"   Message: {response.text[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "=" * 50)
print("✅ All tests completed!")
print("=" * 50)