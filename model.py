import requests

API_URL = "https://api-inference.huggingface.co/models/phmnm/finetunning_phobert_base_sentiment"
headers = {"Authorization": "Bearer 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"}

def predict(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	result = response.json()[0][0]
	
	return result['label']

# output = predict({
# 	"inputs": input,
#     "wait_for_model": True,
# })
