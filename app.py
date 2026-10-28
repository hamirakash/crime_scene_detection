
from flask import Flask, request, jsonify, render_template, send_from_directory
from database import initialize_database, get_connection
from ultralytics import YOLO
import cv2
import numpy as np
import os
import uuid
import json

app = Flask(__name__)
initialize_database()
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Load the AI model
MODEL_PATH = os.path.join(app.root_path, "yolo11n.pt")
model = YOLO(MODEL_PATH)

# Create output folder
OUTPUT_FOLDER = os.path.join(app.root_path, "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/app")
def home():
    return render_template("index.html")


@app.route("/video")
def video_page():
    return render_template("video.html")


@app.route("/outputs/<filename>")
def output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "Please upload an image"}), 400

    image_file = request.files["image"]

    if not image_file.filename:
        return jsonify({"error": "No image selected"}), 400

    allowed = (".jpg", ".jpeg", ".png", ".webp")

    if not image_file.filename.lower().endswith(allowed):
        return jsonify({"error": "Unsupported image format"}), 400

    # Read the uploaded image
    image_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # Run YOLO detection
    results = model.predict(
    source=image,
    conf=0.25,
    imgsz=320,
    device="cpu",
    verbose=False
)

    evidence = []
    evidence_number = 1

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            object_name = result.names[class_id]
            confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            evidence.append({
                "evidence_id": f"E-{evidence_number:03d}",
                "object": object_name,
                "confidence": round(confidence * 100, 2),
                "bounding_box": [
                    round(value, 1)
                    for value in coordinates
                ]
            })

            evidence_number += 1

    # Create a copy of the original image
    annotated_image = image.copy()

    # Draw evidence IDs and bounding boxes
    for item in evidence:

        x1, y1, x2, y2 = map(int, item["bounding_box"])

        label = (
            f'{item["evidence_id"]} | '
            f'{item["object"]} | '
            f'{item["confidence"]}%'
        )

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (30, 30, 30),
            2
        )

        # Draw a filled label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        label_y = max(y1, text_height + 12)

        cv2.rectangle(
            annotated_image,
            (x1, label_y - text_height - 10),
            (x1 + text_width + 10, label_y + baseline),
            (30, 30, 30),
            -1
        )

        # Draw white text on dark background
        cv2.putText(
            annotated_image,
            label,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Save the annotated image
    filename = f"detection_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    cv2.imwrite(output_path, annotated_image)

    return jsonify({
        "message": "Analysis completed",
        "total_objects": len(evidence),
        "evidence": evidence,
        "image_url": f"/outputs/{filename}"
    })

@app.route("/cases", methods=["POST"])
def save_case():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Please provide valid JSON data"}), 400

    case_name = data.get("case_name")
    image_url = data.get("image_url")
    evidence_items = data.get("evidence")
    selected_ids = data.get("selected_evidence_ids", [])

    if not isinstance(case_name, str) or not case_name.strip():
        return jsonify({"error": "Case name is required"}), 400

    if len(case_name.strip()) > 150:
        return jsonify({"error": "Case name is too long"}), 400

    if not isinstance(image_url, str):
        return jsonify({"error": "Image URL is required"}), 400

    image_filename = image_url.removeprefix("/outputs/")

    if (
        not image_url.startswith("/outputs/detection_")
        or not image_filename.endswith(".jpg")
        or os.path.basename(image_filename) != image_filename
        or not os.path.isfile(os.path.join(OUTPUT_FOLDER, image_filename))
    ):
        return jsonify({"error": "Invalid detection image"}), 400

    if not isinstance(evidence_items, list):
        return jsonify({"error": "Evidence must be a list"}), 400

    if not isinstance(selected_ids, list):
        return jsonify({"error": "Selected evidence IDs must be a list"}), 400

    if len(evidence_items) > 1000:
        return jsonify({"error": "Too many evidence items"}), 400

    if not all(isinstance(item, str) for item in selected_ids):
        return jsonify({"error": "Invalid selected evidence IDs"}), 400

    selected_set = set(selected_ids)
    validated_evidence = []
    evidence_ids = set()

    for item in evidence_items:
        if not isinstance(item, dict):
            return jsonify({"error": "Invalid evidence record"}), 400

        evidence_id = item.get("evidence_id")
        object_name = item.get("object")
        confidence = item.get("confidence")
        bounding_box = item.get("bounding_box")

        if (
            not isinstance(evidence_id, str)
            or not evidence_id
            or evidence_id in evidence_ids
            or not isinstance(object_name, str)
            or not object_name
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 100
            or not isinstance(bounding_box, list)
            or len(bounding_box) != 4
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in bounding_box
            )
        ):
            return jsonify({"error": "Invalid evidence details"}), 400

        evidence_ids.add(evidence_id)

        validated_evidence.append((
            evidence_id,
            object_name,
            float(confidence),
            json.dumps(bounding_box),
            int(evidence_id in selected_set)
        ))

    if not selected_set.issubset(evidence_ids):
        return jsonify({"error": "Unknown selected evidence ID"}), 400

    status = "Reviewed" if selected_set else "Pending"

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO cases (
                case_name,
                image_filename,
                total_objects,
                status
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                case_name.strip(),
                image_filename,
                len(validated_evidence),
                status
            )
        )

        case_id = cursor.lastrowid

        connection.executemany(
            """
            INSERT INTO evidence (
                case_id,
                evidence_id,
                object_name,
                confidence,
                bounding_box,
                selected
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (case_id, *item)
                for item in validated_evidence
            ]
        )

        connection.commit()

    return jsonify({
        "message": "Case saved successfully",
        "case_id": case_id,
        "case_name": case_name.strip(),
        "total_objects": len(validated_evidence),
        "selected_evidence": len(selected_set),
        "status": status
    }), 201
@app.route("/cases", methods=["GET"])
def get_cases():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                c.id,
                c.case_name,
                c.image_filename,
                c.total_objects,
                c.status,
                c.created_at,
                COUNT(e.id) AS selected_evidence
            FROM cases c
            LEFT JOIN evidence e
                ON e.case_id = c.id AND e.selected = 1
            GROUP BY c.id
            ORDER BY c.id DESC
        """).fetchall()

    cases = []

    for row in rows:
        case = dict(row)
        case["image_url"] = "/outputs/" + case["image_filename"]
        cases.append(case)

    return jsonify({"cases": cases})

@app.route("/detect-video", methods=["POST"])
def detect_video():
    import tempfile

    if "video" not in request.files:
        return jsonify({"error": "Please upload a video"}), 400

    video_file = request.files["video"]

    if not video_file.filename:
        return jsonify({"error": "No video selected"}), 400

    if not video_file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        return jsonify({"error": "Use an MP4, AVI or MOV video"}), 400

    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp_path = temp.name
            while True:
                chunk = video_file.stream.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            return jsonify({"error": "Unable to read video"}), 400

        frame_results = []

        # Sample up to 10 frames, one second apart
        for second in range(10):
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()

            if not ok:
                break

            predictions = model.predict(
                source=frame, conf=0.25, verbose=False
            )

            evidence = []

            for result in predictions:
                for box in result.boxes:
                    evidence.append({
                        "evidence_id": f"F{second + 1:02d}-E{len(evidence) + 1:03d}",
                        "object": result.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]) * 100, 2),
                        "bounding_box": [
                            round(value, 1)
                            for value in box.xyxy[0].tolist()
                        ]
                    })

            annotated = frame.copy()

            for item in evidence:
                x1, y1, x2, y2 = map(int, item["bounding_box"])

                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2),
                    (30, 30, 30), 2
                )

                cv2.putText(
                    annotated,
                    item["evidence_id"] + " " + item["object"],
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

            filename = f"video_frame_{uuid.uuid4().hex}.jpg"

            if not cv2.imwrite(
                os.path.join(OUTPUT_FOLDER, filename), annotated
            ):
                raise RuntimeError("Could not save analyzed frame")

            frame_results.append({
                "frame_number": second + 1,
                "timestamp_seconds": second,
                "image_url": f"/outputs/{filename}",
                "total_objects": len(evidence),
                "evidence": evidence
            })

        if not frame_results:
            return jsonify({
                "error": "No readable video frames found"
            }), 400

        return jsonify({
            "message": "Video-frame analysis completed",
            "frames_analyzed": len(frame_results),
            "total_objects": sum(
                item["total_objects"] for item in frame_results
            ),
            "frames": frame_results
        })

    except Exception:
        app.logger.exception("Video analysis failed")
        return jsonify({"error": "Video analysis failed"}), 500

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
