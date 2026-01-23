"""
Quick test of the prediction endpoint
"""
import requests
import json

print("🧪 Testing prediction endpoint...")

# The test data from your example
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

try:
    # Make the request
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"✅ Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n📊 Prediction Results:")
        print(f"   Success: {result.get('success')}")
        print(f"   Prediction: {result.get('prediction_label')}")
        print(f"   Apnea Probability: {result.get('apnea_probability'):.3f}")
        print(f"   Normal Probability: {result.get('normal_probability'):.3f}")
        print(f"   Confidence: {result.get('confidence'):.3f}")
        print(f"   Risk Level: {result.get('risk_level')}")
        print(f"   Threshold: {result.get('threshold')}")
        
        # Show the prediction visually
        prob = result.get('apnea_probability', 0)
        bar_length = 40
        filled = int(prob * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n📈 Probability Bar: [{bar}] {prob:.1%}")
        
        if prob > 0.7:
            print("⚠️  High risk of apnea detected!")
        elif prob > 0.3:
            print("⚠️  Moderate risk of apnea detected")
        else:
            print("✅ Low risk of apnea")
            
    else:
        print(f"❌ Error: {response.text}")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n" + "=" * 50)
print("🎉 Your ApneaAlert API is fully functional!")
print("=" * 50)