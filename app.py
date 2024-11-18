from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("retina.pt")

# Define path to the image file
source = "./demoImages/image1.jpg"

# Run inference on the source
results = model.predict(source,save=True,project="./result")  # list of Results objects

print(model.names)

for result in results:
    result.save(filename="result.jpg")
    result.show() 

    for box in result.boxes:  # Iterate through detected boxes
        class_id = int(box.cls)  # Get the class ID
        class_label = model.names[class_id]  # Get the class label using model.names
        confidence = float(box.conf)  # Get the confidence score
        print(f"Detected class: {class_label}, Confidence: {confidence:.2f}")