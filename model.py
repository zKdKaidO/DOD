from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    cfg = "config.yaml",  # Path to model configuration file
    data="coco8.yaml", # Path to dataset configuration file
    pretrained = "yolov8n-p2.pt",  # Use pretrained weights
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("../DOD/horse (1).jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
#path = model.export(format="onnx")  # Returns the path to the exported model
