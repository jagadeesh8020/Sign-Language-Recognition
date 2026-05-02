import base64
import csv
import threading
from pathlib import Path

import cv2 as cv
import mediapipe as mp

from slr.model.classifier import KeyPointClassifier
from slr.utils.draw_debug import draw_bounding_rect, draw_hand_label
from slr.utils.landmarks import draw_landmarks
from slr.utils.pre_process import calc_bounding_rect, calc_landmark_list, pre_process_landmark


class SignLanguagePredictor:
    def __init__(
        self,
        model_path="slr/model/slr_model.tflite",
        labels_path="slr/model/label.csv",
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ):
        self.labels = self._load_labels(labels_path)
        self.classifier = KeyPointClassifier(model_path=model_path)
        self.lock = threading.Lock()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def predict_image(self, image):
        image = self._resize_for_inference(image)
        annotated_image = image.copy()

        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False

        with self.lock:
            results = self.hands.process(rgb_image)
            predictions = self._collect_predictions(annotated_image, results)

        encoded_image = self._encode_image(annotated_image)
        return {
            "predictions": predictions,
            "annotated_image": encoded_image,
            "hand_detected": bool(predictions),
        }

    def _collect_predictions(self, image, results):
        if results.multi_hand_landmarks is None:
            return []

        predictions = []
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmarks = pre_process_landmark(landmark_list)
            sign_id, confidence = self.classifier.predict(pre_processed_landmarks)
            label = self._label_for(sign_id)

            bounding_rect = calc_bounding_rect(image, hand_landmarks)
            hand = handedness.classification[0].label

            draw_bounding_rect(image, True, bounding_rect, outline_color=(30, 130, 76), pad=8)
            draw_landmarks(image, landmark_list)
            draw_hand_label(image, bounding_rect, handedness)
            self._draw_prediction_label(image, bounding_rect, label, confidence)

            predictions.append(
                {
                    "label": label,
                    "confidence": round(confidence, 4),
                    "hand": hand,
                    "bounding_box": bounding_rect,
                }
            )

        return predictions

    def _label_for(self, sign_id):
        if 0 <= sign_id < len(self.labels):
            return self.labels[sign_id]
        return "Unknown"

    @staticmethod
    def _draw_prediction_label(image, bounding_rect, label, confidence):
        x1, y1, x2, y2 = bounding_rect
        text = f"{label}  {confidence:.0%}"
        label_y = min(y2 + 32, image.shape[0] - 8)
        cv.rectangle(image, (x1, label_y - 24), (min(x1 + 220, image.shape[1] - 1), label_y + 6), (30, 130, 76), -1)
        cv.putText(image, text, (x1 + 8, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

    @staticmethod
    def _resize_for_inference(image, max_width=960):
        height, width = image.shape[:2]
        if width <= max_width:
            return image

        scale = max_width / width
        return cv.resize(image, (max_width, int(height * scale)), interpolation=cv.INTER_AREA)

    @staticmethod
    def _encode_image(image):
        success, buffer = cv.imencode(".jpg", image, [cv.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            return None
        return base64.b64encode(buffer).decode("ascii")

    @staticmethod
    def _load_labels(labels_path):
        path = Path(labels_path)
        with path.open(encoding="utf-8-sig") as label_file:
            return [row[0] for row in csv.reader(label_file) if row]
