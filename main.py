import requests
import google.generativeai as genai
genai.configure(api_key="AIzaSyDM9xdKD9JDW_wu6Lp1gnCraUK3Ds-DPNc")
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image
import os
import time

model = load_model("my_model.h5")
print("Model loaded successfully.")

API_URL = "https://api-inference.huggingface.co/models/blaze999/Medical-NER"
headers = {"Authorization": "Bearer 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"}  

def query(payload, retries=5, wait_time=2):
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            data = response.json()
            if 'error' in data:
                print(f"Error: {data['error']}")
                if 'currently loading' in data['error'].lower():
                    print("Model is loading, retrying...")
                    time.sleep(wait_time)
                    continue  # Retry the request
                return None
            return data
        except Exception as e:
            print(f"An error occurred during the API request: {e}")
            return None
    print("Max retries reached. Could not query the model.")
    return None

def create_entity_table(entities):
    if entities is None or not isinstance(entities, list):
        print("No entities to display or invalid format.")
        return None
    
    entity_data = {
        'Entity Type': [],
        'Entity Text': []
    }
    
    for entity in entities:
        entity_data['Entity Type'].append(entity.get('entity_group', 'Unknown'))
        entity_data['Entity Text'].append(entity.get('word', 'Unknown'))

    entity_table = pd.DataFrame(entity_data)
    print(entity_data)
    print("Entity table created successfully.")
    return entity_table

import time

def summarize_text(text, retries=5, wait_time=10):
    try:
        model_name = "Khushiee/pegasus-samsum-summarization"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        summary_ids = model.generate(inputs["input_ids"])

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        print("Text summarized successfully.")
        return summary
    except Exception as e:
        print(f"An error occurred during text summarization: {e}")
        return None

base_prompt = """
You have given the prescription of the patient you just you need summarize it on the medical basis :
Just provide a paragraph summarized .
"""

def summarize_text(text):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(base_prompt + text)
    return response.text
label_mapping = {
    0: "glioma",
    1: "meningioma",
    2: "notumor",
    3: "pituitary"
}

def get_label_from_prediction(predicted_index):
    label = label_mapping.get(predicted_index, "Unknown")
    print(f"Predicted label: {label}")
    return label

def visualize_prediction_from_image(img_path=None, img_url=None):
    try:
        if img_url:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
        elif img_path:
            img = Image.open(img_path)
        else:
            print("No image provided.")
            return None

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_label_index = np.argmax(prediction, axis=1)[0]
        predicted_label = get_label_from_prediction(predicted_label_index)

        save_path = os.path.join("static", "uploads", f"prediction_result_{predicted_label}.png")
        print(f"Image saved at: {save_path}")
        return predicted_label, save_path
    except Exception as e:
        print(f"An error occurred during image prediction: {e}")
        return None

import os
def output_report(image_path, custom_text):
    predicted_label = "Prediction functionality removed"
    
    entities_output = query({"inputs": custom_text})
    entity_table = create_entity_table(entities_output)

    summarized_text = None
    summarized_text = summarize_text(custom_text)
    if summarized_text:
            print(summarize_text)
    report = {
        "Predicted Tumor Type": predicted_label,
        "Extracted Entities": entity_table.to_dict(orient='records') if entity_table is not None else [],
        "Summarized Text": summarized_text if summarized_text else "No summary available.",
        "Image Path": None  
    }

    return report
