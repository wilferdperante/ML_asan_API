import requests
import json

def test_api():
    url = 'http://127.0.0.1:5000/predict'
    
    # Single prediction data
    data = {
        "Avg. Session Length": 34.49726773,
        "Time on App": 12.65565115,
        "Time on Website": 39.57766802,
        "Length of Membership": 4.082620633
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            url,
            data=json.dumps(data),
            headers=headers
        )
        response.raise_for_status()
        
        result = response.json()
        print("✅ Prediction successful")
        print("Result:", result)
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        if hasattr(e, 'response') and e.response:
            print("Response:", e.response.text)
        return None

if __name__ == '__main__':
    test_api()