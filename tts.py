"""
curl --request POST \
     --url https://play.ht/api/v1/convert \
     --header 'accept: application/json' \
     --header 'content-type: application/json' \
     --data '
{
  "content": [
    "Hey you!"
  ],
  "voice": "en-US-JennyNeural"
}
'"""
import requests

with open("tts-key.txt", "r") as f:
    key = f.read()
    
user_id = key.split("\n")[0]
key = key.split("\n")[1]

API_URL = "https://play.ht/api/v1/convert"
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "AUTHORIZATION": f"{key}",
    "X-USER-ID": f"{user_id}"
}


#get audio from transcriptionId
def get_audio_url(transcript_id : str):
    url = f"https://play.ht/api/v1/articleStatus?transcriptionId={transcript_id}"
    headers_transcript = {
        "accept": "application/json",
        "AUTHORIZATION": f"{key}",
        "X-USER-ID": f"{user_id}"
    }
    response = requests.get(url, headers=headers_transcript).json()
    return response['audioUrl']


def convert_to_audio(text : str):
    data = {
        "content": [
            text
        ],
        "voice": "ta-IN-PallaviNeural"
    }
    print()
    # return file as response
    print(requests.post(API_URL, headers=headers, json=data).json()['transcriptionId'])
    audiourl = get_audio_url(requests.post(API_URL, headers=headers, json=data).json()['transcriptionId'])
    print(audiourl)
    # return requests.get(url= audiourl)
    return audiourl