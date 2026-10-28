
from ultralytics import YOLO

# STEP 1: Load the AI model
model = YOLO("yolo11n.pt")

# STEP 2: Give the AI an image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True
)

# STEP 3: Read the detected objects
for result in results:
    for box in result.boxes:

        # Get the object name
        class_id = int(box.cls[0])
        object_name = result.names[class_id]

        # Get the confidence score
        confidence = float(box.conf[0])

        print("Detected Object:", object_name)
        print("Confidence:", round(confidence * 100, 2), "%")
        print("----------------------------")

print("Object Detection Completed!")
