from flask import Flask, render_template, request, redirect, url_for, jsonify
from main import output_report
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    print("Accessed index page.")
    return render_template('index.html')

@app.route('/main')
def main():
    print("Accessed main page.")
    return render_template('main.html')

@app.route('/report', methods=['POST'])
def report():
    if 'image' not in request.files:
        print("No image uploaded in the request.")
        return "No image uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        print("No selected file.")
        return "No selected file.", 400

    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(f"Attempting to save image at: {img_path}")

    # Save the file to the upload folder
    try:
        file.save(img_path)
        print(f"Image successfully saved at: {img_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return "Error saving image.", 500

    if os.path.exists(img_path):
        print(f"File exists at: {img_path}")
    else:
        print(f"File not found at: {img_path}")
        return "File not found after saving.", 500

    custom_text = request.form.get("custom_text")
    if not custom_text:
        print("No custom text provided.")
        return "Please provide prescription text.", 400
    print(f"Custom text received: {custom_text}")

    report_data = output_report(img_path, custom_text)
    if report_data is None:
        print("Report generation failed.")
        return "Report generation failed. Please check your inputs.", 400
    print("Report successfully generated.")

    return render_template("result.html", report_data=report_data)

if __name__ == "__main__":
    print("Starting Flask app on port 5001...")
    app.run(debug=True, port=5001)
