
import cv2
import numpy as np
import onnxruntime as ort


class ONNXDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        # Standard COCO object names used by YOLO11n
        self.names = [
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass",
            "cup", "fork", "knife", "spoon", "bowl", "banana",
            "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet",
            "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster",
            "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

    def predict(self, image, conf=0.25):
        original_h, original_w = image.shape[:2]

        # Resize while preserving aspect ratio
        scale = min(320 / original_w, 320 / original_h)
        new_w = round(original_w * scale)
        new_h = round(original_h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.full((320, 320, 3), 114, dtype=np.uint8)
        pad_x = (320 - new_w) // 2
        pad_y = (320 - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        output = self.session.run(
            None, {self.input_name: tensor}
        )[0]

        # Standard YOLO11 detection export:
        # (1, 84, number_of_predictions)
        predictions = output[0].T

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            if score < conf:
                continue

            cx, cy, w, h = pred[:4]

            x1 = (cx - w / 2 - pad_x) / scale
            y1 = (cy - h / 2 - pad_y) / scale
            x2 = (cx + w / 2 - pad_x) / scale
            y2 = (cy + h / 2 - pad_y) / scale

            x1 = max(0, min(float(x1), original_w))
            y1 = max(0, min(float(y1), original_h))
            x2 = max(0, min(float(x2), original_w))
            y2 = max(0, min(float(y2), original_h))

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        # Remove overlapping detections, separately per class
        selected = []
        for class_id in set(class_ids):
            indices = [
                i for i, value in enumerate(class_ids)
                if value == class_id
            ]

            keep = cv2.dnn.NMSBoxes(
                [boxes[i] for i in indices],
                [scores[i] for i in indices],
                conf,
                0.45
            )

            for index in np.array(keep).reshape(-1):
                selected.append(indices[int(index)])

        selected.sort(key=lambda i: scores[i], reverse=True)

        evidence = []

        for number, i in enumerate(selected, start=1):
            x, y, w, h = boxes[i]

            evidence.append({
                "evidence_id": f"E-{number:03d}",
                "object": self.names[class_ids[i]],
                "confidence": round(scores[i] * 100, 2),
                "bounding_box": [
                    round(x, 1),
                    round(y, 1),
                    round(x + w, 1),
                    round(y + h, 1)
                ]
            })

        return evidence
