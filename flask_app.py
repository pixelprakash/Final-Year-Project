from flask import Flask, render_template, request, send_file
import cv2
from model_inference import KidneyStoneDetectionModel
import matplotlib.pyplot as plt

app = Flask(__name__)

original_image_path = None
processed_image_path = None
processed_image = None




@app.route('/result')
def result():
   return render_template("result.html")


@app.route('/', methods=["GET", "POST"])
def index():
    count=0
    global original_image_path, processed_image_path, processed_image
    if request.method == 'POST':
        original_image = request.files["file"]

        if original_image.filename == "":
            return "No file selected."

        # Save the uploaded image
        original_image_path = "./static/uploaded_image.jpg"
        original_image.save(original_image_path)

        original_image_np = cv2.imread(original_image_path) # Convert image into nparray
        original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_BGR2RGB)
        
        # Kidney stone detection
        model = KidneyStoneDetectionModel(model_path="./ks_detection.pt") # Load model
        model.run_inference(image=original_image_np) # Run inference
        processed_image,count = model.annotate_image(image=original_image_np) # Annotate image

        processed_image_path = "static/processed_image.jpg"
        plt.imsave(processed_image_path, processed_image)

        return render_template("result.html", img_paths=[original_image_path, processed_image_path], count=count)

    return render_template("index.html")




if __name__ == '__main__':
    app.run(debug=True)
