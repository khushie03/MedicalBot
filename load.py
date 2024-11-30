import requests
API_URL = "https://api-inference.huggingface.co/models/blaze999/Medical-NER"
headers = {"Authorization": "Bearer hf_key"}  

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        data = response.json()
        if 'error' in data:
            print(f"Error: {data['error']}")
            return None
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

out = query("""Symptoms of gliomas depend on which part of the central nervous system is affected. A brain glioma can cause headaches, vomiting, seizures, and cranial nerve disorders as a result of increased intracranial pressure. Also, different cognitive impairments can arise as a sign of tumor growth. A glioma of the optic nerve can cause vision loss. Spinal cord gliomas can cause pain, weakness, or numbness in the extremities. Gliomas do not usually metastasize by the bloodstream, but they ca…""")
print(out)
