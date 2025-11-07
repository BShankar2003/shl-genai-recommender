"""
Quick API Response Tester
Tests if your API returns the correct format with assessment URLs
"""

import requests
import json

API_URL = "http://127.0.0.1:8000/recommend"

def test_api():
    """Test API with sample queries."""
    
    test_queries = [
        "Hiring a software engineer skilled in Python",
        "Looking for a data analyst with SQL skills",
        "Need a project manager with leadership experience"
    ]
    
    print("🧪 Testing API Response Format\n")
    print("="*70)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-"*70)
        
        try:
            response = requests.post(
                API_URL,
                json={"query": query, "top_k": 3},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Status: {response.status_code}")
                print(f"Query Type: {data.get('query_type', 'N/A')}")
                print(f"Results Count: {len(data.get('results', []))}")
                
                print("\n📋 Results:")
                for i, result in enumerate(data.get('results', [])[:3], 1):
                    print(f"\n  {i}. {result.get('assessment_name', 'N/A')}")
                    print(f"     Category: {result.get('category', 'N/A')}")
                    print(f"     Score: {result.get('similarity_score', 'N/A'):.3f}")
                    
                    # Check if URL is present
                    url = result.get('assessment_url', '')
                    if url:
                        print(f"     ✅ URL: {url}")
                    else:
                        print(f"     ⚠️  WARNING: No 'assessment_url' field found!")
                        print(f"     Available fields: {list(result.keys())}")
                
            else:
                print(f"❌ Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print("\n" + "="*70)
    print("\n💡 IMPORTANT:")
    print("   Make sure your API returns 'assessment_url' in each result!")
    print("   Check your api.py file to ensure URLs are included in the response.")

if __name__ == "__main__":
    test_api()