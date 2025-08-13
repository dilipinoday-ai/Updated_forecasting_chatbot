import requests

API_KEY = "AIzaSyBucS0pyC3gfNTUemlauMlcVQfNTl8Yhqc"
url = f"https://generativelanguage.googleapis.com/v1/models?key={API_KEY}"

response = requests.get(url)
print(response.json())
