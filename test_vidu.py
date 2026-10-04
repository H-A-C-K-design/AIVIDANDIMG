import requests

API_KEY = "vda_1004229886702784512_Jk3nEjtcFwt8c1DlnVYfkNJLzJhOw86N"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

endpoints = [
    "https://api.vidu.studio/api/v1/video/text-to-video",
    "https://api.vidu.studio/v1/video/text-to-video",
    "https://api.vidu.com/api/v1/video/text-to-video",
    "https://api.vidu.com/v1/video/text-to-video",
    "https://api.vidu.com/ent/v2/text-to-video"
]

payload = {
    "prompt": "A dog walking",
    "style": "general"
}

for url in endpoints:
    try:
        print(f"Testing {url}...")
        resp = requests.post(url, headers=HEADERS, json=payload, timeout=5)
        print(resp.status_code, resp.text)
    except Exception as e:
        print("Error:", e)
