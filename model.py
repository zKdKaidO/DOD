from ultralytics import YOLO
model = YOLO("ultralytics/cfg/models/v8/yolov8-p2.yaml")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(cfg = "config.yaml")

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
#results = model("../DOD/horse (1).jpg")  # Predict on an image
#results[0].show()  # Display results

# Export the model to ONNX format for deployment
#path = model.export(format="onnx")  # Returns the path to the exported model
