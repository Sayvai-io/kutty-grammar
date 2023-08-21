# whisper audio 
import requests

with open("keys.txt", "r") as f:
    key = f.read()

API_URL = "https://p0oi0gtn7iyhoioj.us-east-1.aws.endpoints.huggingface.cloud/" #--> english
# API_URL = "https://z4dl3t7upduo9inw.us-east-1.aws.endpoints.huggingface.cloud/" #--> tamil
API_TOKEN = key
headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "audio/mpeg"}

def get_text(file: str) -> str:
    return requests.post(API_URL, headers=headers, data = file).json()