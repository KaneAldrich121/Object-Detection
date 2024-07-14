import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

# Load Instance of Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

# Features
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load Pretrained Model Data
model.load_state_dict(torch.load('./V3_fasterrcnn_resnet50_fpn.pth'))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')
model.to('cpu')

# Define Image Transformations
transform = T.Compose([
    T.ToTensor(),
])

# OD Function
def detect_objects(frame):
    # Transform
    img = transform(frame)
    img = img.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img)

    return predictions

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    predictions = detect_objects(frame)

    # Extract Results
    prediction = predictions[0]
    boxes = prediction['boxes'].numpy()
    labels = prediction['labels'].numpy()
    scores = prediction['scores'].numpy()

    # Draw Bounding Boxes
    for i in range(len(scores)):
        if scores[i] > 0.5:
            box = boxes[i]
            class_id = labels[i]
            score = scores[i]

            # Convert Coordinates
            (startX, startY, endX, endY) = box.astype(int)

            # Draw Box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Draw Label
            label = f"Class {class_id}: {score:.2f}"
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display Frame
    cv2.imshow('Object Detection', frame)

    # Break Loop on "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close
cap.release()
cv2.destroyAllWindows()
