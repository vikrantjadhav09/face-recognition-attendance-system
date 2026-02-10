import os
import json
import base64
import datetime
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO

app = Flask(__name__)

# ---------------- CONFIG ----------------
STUDENTS_DB_FILE = "students.json"
ATTENDANCE_DB_FILE = "attendance_history.json"
FACE_SIMILARITY_THRESHOLD = 0.85  # Very strict threshold to ensure only exact matches
FACE_SIMILARITY_MIN_THRESHOLD = 0.75  # Secondary threshold - if close but not matching, reject

# Simple face detection: extract center region of image as face
# (assumes user positions face in center of frame)
FACE_REGION_RATIO = 0.6  # Extract center 60% of image

# ---------------- GLOBAL DATA ----------------
known_face_features = []
known_face_names = []
student_data_map = {}
attendance_history = []

# ---------------- UTILITIES ----------------
def decode_image_from_base64(base64_string):
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        raise


def extract_face_region(image):
    """Extract center region of image as face"""
    width, height = image.size
    face_size = int(min(width, height) * FACE_REGION_RATIO)
    left = (width - face_size) // 2
    top = (height - face_size) // 2
    return image.crop((left, top, left + face_size, top + face_size))


def extract_face_features(face_img):
    try:
        # Resize to consistent size
        face_img = face_img.resize((100, 100), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = ImageOps.grayscale(face_img)
        gray_array = np.array(gray, dtype=np.uint8)
        
        # Extract multi-region features for better discrimination
        h, w = gray_array.shape
        h_third = h // 3
        w_half = w // 2
        
        # Divide face into regions: forehead, eyes, mouth, left, right
        forehead = gray_array[:h_third, :]  # Top third
        middle = gray_array[h_third:2*h_third, :]  # Middle third (eyes)
        lower = gray_array[2*h_third:, :]  # Lower third (mouth)
        left_half = gray_array[:, :w_half]  # Left side
        right_half = gray_array[:, w_half:]  # Right side
        
        # Extract histograms from each region with different bin sizes for detail
        features = []
        
        # Global histogram (32 bins for overall pattern)
        hist_global, _ = np.histogram(gray_array, bins=32, range=(0, 256))
        features.extend(hist_global)
        
        # Regional histograms (16 bins each for regional patterns)
        for region in [forehead, middle, lower, left_half, right_half]:
            hist, _ = np.histogram(region, bins=16, range=(0, 256))
            features.extend(hist)
        
        # Add statistical features for better discrimination
        stats = [
            np.mean(gray_array),  # Overall brightness
            np.std(gray_array),   # Contrast
            np.mean(forehead),    # Forehead brightness
            np.mean(lower),       # Lower face brightness
            np.std(left_half),    # Left side variation
            np.std(right_half),   # Right side variation
            np.mean(middle),      # Eye region brightness
        ]
        features.extend(stats)
        
        # Convert to numpy array and normalize
        feature_vector = np.array(features, dtype=np.float32)
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
        
        return feature_vector
    except Exception as e:
        print(f"Error extracting face features: {e}")
        raise


def detect_sentiment(face_img):
    """
    Detect sentiment/emotion from face image
    Returns sentiment label and confidence score
    """
    try:
        # Resize for analysis
        face_img_resized = face_img.resize((100, 100), Image.Resampling.LANCZOS)
        gray = ImageOps.grayscale(face_img_resized)
        gray_array = np.array(gray, dtype=np.uint8)
        
        # Analyze facial features for sentiment
        # Divide face into regions and analyze brightness/contrast
        height, width = gray_array.shape
        mid_h, mid_w = height // 2, width // 2
        
        # Get brightness levels in different regions
        forehead = gray_array[:mid_h, :]  # Upper region
        lower_face = gray_array[mid_h:, :]  # Lower region
        left_side = gray_array[:, :mid_w]  # Left region
        right_side = gray_array[:, mid_w:]  # Right region
        
        # Calculate metrics
        forehead_brightness = np.mean(forehead)
        lower_brightness = np.mean(lower_face)
        brightness_diff = forehead_brightness - lower_brightness
        
        # Edge detection for expression intensity (apply filter before converting to array)
        edges_img = gray.filter(ImageFilter.FIND_EDGES)
        edges_array = np.array(edges_img, dtype=np.uint8)
        edge_intensity = np.mean(edges_array)
        
        # Face symmetry analysis
        left_mean = np.mean(left_side)
        right_mean = np.mean(right_side)
        symmetry = 1.0 - (abs(left_mean - right_mean) / 255.0)
        
        # Overall brightness and contrast
        overall_brightness = np.mean(gray_array)
        contrast = np.std(gray_array)
        
        # Determine sentiment based on features (more lenient thresholds)
        sentiment = "Neutral"
        confidence = 0.5
        
        # Happy: brighter lower face (smile creates shadow above), moderate to high edges
        if brightness_diff < -3 and edge_intensity > 12 and symmetry > 0.65:
            sentiment = "Happy"
            confidence = min(0.95, 0.6 + (edge_intensity / 120))
        # Sad: darker lower face, lower edge intensity, more contrast
        elif brightness_diff > 5 and edge_intensity < 12 and contrast > 20:
            sentiment = "Sad"
            confidence = min(0.95, 0.6 + abs(brightness_diff) / 40)
        # Angry: high contrast, asymmetric, moderate edge intensity, darker overall
        elif abs(brightness_diff) > 15 and edge_intensity > 18 and symmetry < 0.65 and overall_brightness < 130:
            sentiment = "Angry"
            confidence = min(0.95, 0.6 + edge_intensity / 60)
        # Surprised: very high edge intensity, good symmetry, bright overall
        elif edge_intensity > 22 and symmetry > 0.7 and overall_brightness > 115:
            sentiment = "Surprised"
            confidence = min(0.95, 0.6 + edge_intensity / 70)
        # Confused: mixed signals - moderate edges, asymmetric, varied brightness
        elif edge_intensity > 15 and symmetry < 0.6 and abs(brightness_diff) > 8 and abs(brightness_diff) < 15:
            sentiment = "Confused"
            confidence = min(0.95, 0.55 + contrast / 100)
        else:
            sentiment = "Neutral"
            confidence = 0.5 + (symmetry * 0.25) + (contrast / 200)
        
        return sentiment, round(float(confidence), 2)
    except Exception as e:
        print(f"Error detecting sentiment: {e}")
        return "Neutral", 0.5


def calculate_similarity(f1, f2):
    """Calculate chi-square distance (lower is more similar)"""
    # Use chi-square distance which is better for histograms
    # Convert to similarity (1 - distance for normalized chi-square)
    distance = 0.5 * np.sum((f1 - f2) ** 2 / (f1 + f2 + 1e-8))
    return float(1.0 - distance)


def detect_faces(image):
    """
    Simple face detection: assumes user centering.
    Returns list of face coordinates in format: [(x, y, w, h)]
    """
    width, height = image.size
    face_size = int(min(width, height) * FACE_REGION_RATIO)
    x = (width - face_size) // 2
    y = (height - face_size) // 2
    # Return in (x, y, w, h) format for compatibility
    return [(x, y, face_size, face_size)]


# ---------------- DATA LOAD/SAVE ----------------
def load_data():
    global known_face_features, known_face_names, student_data_map, attendance_history

    known_face_features.clear()
    known_face_names.clear()
    student_data_map.clear()
    attendance_history.clear()

    if os.path.exists(STUDENTS_DB_FILE):
        try:
            with open(STUDENTS_DB_FILE, "r") as f:
                data = json.load(f)
                for sid, s in data.items():
                    if "name" in s and "features" in s:
                        # Convert features to numpy array
                        features = np.array(s["features"], dtype=np.float32)
                        # Normalize as histogram (sum to 1)
                        features = features / (np.sum(features) + 1e-8)
                        
                        student_data_map[sid] = {
                            "name": s["name"],
                            "features": features,
                        }
                        known_face_names.append(s["name"])
                        known_face_features.append(features)
        except Exception as e:
            print(f"Error loading students: {e}")

    if os.path.exists(ATTENDANCE_DB_FILE):
        try:
            with open(ATTENDANCE_DB_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    attendance_history.extend(data)
        except Exception as e:
            print(f"Error loading attendance: {e}")


def save_students():
    data = {}
    for sid, s in student_data_map.items():
        features = s["features"]
        # Convert numpy array to list if needed
        if hasattr(features, 'tolist'):
            features_list = features.tolist()
        else:
            features_list = list(features)
        data[sid] = {"name": s["name"], "features": features_list}
    with open(STUDENTS_DB_FILE, "w") as f:
        json.dump(data, f, indent=4)


def save_attendance():
    with open(ATTENDANCE_DB_FILE, "w") as f:
        json.dump(attendance_history, f, indent=4)


load_data()

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/register_student", methods=["POST"])
def register_student():
    try:
        data = request.json
        sid = data.get("studentId")
        name = data.get("studentName")
        img64 = data.get("image")

        if not sid or not name or not img64:
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        image = decode_image_from_base64(img64)
        faces = detect_faces(image)

        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face region detected"}), 400

        # Extract face region
        x, y, w, h = faces[0]
        face_img = image.crop((x, y, x + w, y + h))
        features = extract_face_features(face_img)

        student_data_map[sid] = {"name": name, "features": features}
        known_face_names.append(name)
        known_face_features.append(features)

        save_students()
        return jsonify({"status": "success", "message": "Student registered"})
    except Exception as e:
        print(f"Error in register_student: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/process_frame", methods=["POST"])
def process_frame():
    try:
        data = request.json
        image = decode_image_from_base64(data.get("image"))
        faces = detect_faces(image)

        detections = []
        today = datetime.date.today().isoformat()

        for (x, y, w, h) in faces:
            # Extract face region
            face_img = image.crop((x, y, x + w, y + h))
            features = extract_face_features(face_img)
            
            # Detect sentiment from face
            sentiment, sentiment_conf = detect_sentiment(face_img)

            name = "Unknown"
            marked = False

            if known_face_features:
                sims = [calculate_similarity(features, f) for f in known_face_features]
                idx = int(np.argmax(sims))
                max_similarity = sims[idx]
                
                # Debug logging with detailed info
                all_sims = ", ".join([f"{s:.3f}" for s in sims])
                print(f"Max: {max_similarity:.3f} | All: [{all_sims}] | Threshold: {FACE_SIMILARITY_THRESHOLD} | Student: {known_face_names[idx]} | Sentiment: {sentiment}")
                
                # Only match if similarity is well above threshold (strict matching)
                if max_similarity > FACE_SIMILARITY_THRESHOLD:
                    name = known_face_names[idx]

                    sid = next(
                        (k for k, v in student_data_map.items() if v["name"] == name),
                        None,
                    )

                    if sid:
                        now = datetime.datetime.now()
                        
                        # Check if student was marked today
                        marked_today = any(
                            a["studentId"] == sid and a["date"] == today
                            for a in attendance_history
                        )
                        
                        # Mark attendance only if NOT marked today (one mark per day, no proxy)
                        if not marked_today:
                            attendance_history.append({
                                "logId": str(len(attendance_history) + 1),
                                "studentId": sid,
                                "name": name,
                                "time": now.strftime("%H:%M:%S"),
                                "date": today,
                                "sentiment": sentiment,
                                "sentiment_confidence": sentiment_conf,
                            })
                            save_attendance()
                            marked = True

            detections.append({
                "name": name,
                "box": {
                    "top": int(y),
                    "right": int(x + w),
                    "bottom": int(y + h),
                    "left": int(x),
                },
                "attendance_marked": marked,
                "sentiment": sentiment,
                "sentiment_confidence": sentiment_conf,
            })

        return jsonify({
            "status": "success",
            "detections": detections,
            "attendanceLogs": [
                a for a in attendance_history if a["date"] == today
            ],
        })
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/get_students", methods=["GET"])
def get_students():
    """Return list of all registered students"""
    try:
        students = [
            {"id": sid, "name": s["name"]}
            for sid, s in student_data_map.items()
        ]
        return jsonify({"status": "success", "students": students})
    except Exception as e:
        print(f"Error in get_students: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/get_today_attendance", methods=["GET"])
def get_today_attendance():
    """Return today's attendance records"""
    try:
        today = datetime.date.today().isoformat()
        today_attendance = [
            a for a in attendance_history if a["date"] == today
        ]
        return jsonify({"status": "success", "attendance": today_attendance})
    except Exception as e:
        print(f"Error in get_today_attendance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/clear_today_attendance", methods=["POST"])
def clear_today_attendance():
    """Clear all attendance records for today"""
    global attendance_history
    try:
        today = datetime.date.today().isoformat()
        # Remove all records for today
        attendance_history = [a for a in attendance_history if a["date"] != today]
        save_attendance()
        return jsonify({"status": "success", "message": "Today's attendance cleared"})
    except Exception as e:
        print(f"Error in clear_today_attendance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/clear_all_students", methods=["POST"])
def clear_all_students():
    """Clear all registered students"""
    global student_data_map, known_face_features, known_face_names
    try:
        student_data_map.clear()
        known_face_features.clear()
        known_face_names.clear()
        save_students()
        return jsonify({"status": "success", "message": "All students cleared"})
    except Exception as e:
        print(f"Error in clear_all_students: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Running Face Recognition Attendance System")
    app.run(debug=True)
