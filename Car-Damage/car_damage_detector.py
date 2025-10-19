import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Union
import os
import urllib.request
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarDamageDetector:
    """
    Car damage detection system using YOLOv8 model.
    
    This class handles the detection and classification of various types
    of car damage including scratches, dents, paint damage, and broken parts.
    """
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path or "yolov8n.pt"
        self.device = self._get_device()
        self.model = None
        self.class_names = {
            0: "scratch",
            1: "dent", 
            2: "broken_part",
            3: "paint_damage"
        }
        
        self.severity_mapping = {
            "scratch": {"light": (0, 5), "moderate": (5, 15), "severe": (15, 100)},
            "dent": {"light": (0, 3), "moderate": (3, 10), "severe": (10, 100)},
            "broken_part": {"light": (0, 2), "moderate": (2, 8), "severe": (8, 100)},
            "paint_damage": {"light": (0, 4), "moderate": (4, 12), "severe": (12, 100)}
        }
        
        self.cost_estimates = {
            "scratch": {"light": 100, "moderate": 300, "severe": 800},
            "dent": {"light": 200, "moderate": 500, "severe": 1200},
            "broken_part": {"light": 300, "moderate": 800, "severe": 2000},
            "paint_damage": {"light": 150, "moderate": 400, "severe": 900}
        }
        
        self._load_model()
    
    def _get_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_model(self) -> None:
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading custom model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                logger.info("Loading pre-trained YOLOv8 model")
                self.model = YOLO("yolov8n.pt")
                logger.warning("Using base YOLOv8 model. For production, use a model trained on car damage data.")
            
            if self.device != "cpu":
                self.model.to(self.device)
                
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, np.ndarray):
            img = image.copy()
        else:
            raise TypeError("Image must be a file path, PIL Image, or numpy array")
        
        return img
    
    def _calculate_damage_area(self, bbox: List[int], image_shape: Tuple[int, int]) -> float:
        x1, y1, x2, y2 = bbox
        damage_area = (x2 - x1) * (y2 - y1)
        total_area = image_shape[0] * image_shape[1]
        return (damage_area / total_area) * 100
    
    def _classify_severity(self, damage_type: str, area_percentage: float) -> str:
        if damage_type not in self.severity_mapping:
            return "moderate"
        
        thresholds = self.severity_mapping[damage_type]
        
        if area_percentage <= thresholds["light"][1]:
            return "light"
        elif area_percentage <= thresholds["moderate"][1]:
            return "moderate"
        else:
            return "severe"
    
    def _estimate_repair_cost(self, damage_type: str, severity: str, area_percentage: float) -> int:
        if damage_type not in self.cost_estimates:
            return 0
        
        base_cost = self.cost_estimates[damage_type][severity]
        
        area_multiplier = max(1.0, area_percentage / 10.0)
        estimated_cost = int(base_cost * area_multiplier)
        
        return estimated_cost
    
    def detect_damage(self, image: Union[str, np.ndarray, Image.Image]) -> Dict:
        try:
            img_array = self._preprocess_image(image)
            original_shape = img_array.shape[:2]
            
            results = self.model(img_array, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    damage_type = self._simulate_damage_classification(bbox, img_array)
                    
                    area_percentage = self._calculate_damage_area(bbox, original_shape)
                    
                    severity = self._classify_severity(damage_type, area_percentage)
                    
                    estimated_cost = self._estimate_repair_cost(damage_type, severity, area_percentage)
                    
                    detection = {
                        "type": damage_type,
                        "severity": severity,
                        "confidence": confidence,
                        "bbox": bbox,
                        "area_percentage": round(area_percentage, 2),
                        "estimated_cost": estimated_cost,
                        "location": self._describe_location(bbox, original_shape)
                    }
                    
                    detections.append(detection)
            
            result = {
                "image_shape": original_shape,
                "total_damages": len(detections),
                "damages": detections,
                "total_estimated_cost": sum([d["estimated_cost"] for d in detections]),
                "highest_severity": self._get_highest_severity(detections),
                "processing_info": {
                    "model_used": "YOLOv8",
                    "device": self.device,
                    "confidence_threshold": self.confidence_threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during damage detection: {str(e)}")
            raise RuntimeError(f"Damage detection failed: {str(e)}")
    
    def _simulate_damage_classification(self, bbox: List[int], image: np.ndarray) -> str:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        
        if aspect_ratio > 2.0:
            return "scratch"
        elif width * height < 5000:
            return "paint_damage"
        elif aspect_ratio < 0.8:
            return "dent"
        else:
            return "broken_part"
    
    def _describe_location(self, bbox: List[int], image_shape: Tuple[int, int]) -> str:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        height, width = image_shape
        
        if center_x < width / 3:
            horizontal = "left"
        elif center_x > 2 * width / 3:
            horizontal = "right"
        else:
            horizontal = "center"
        
        if center_y < height / 3:
            vertical = "upper"
        elif center_y > 2 * height / 3:
            vertical = "lower"
        else:
            vertical = "middle"
        
        return f"{vertical} {horizontal}"
    
    def _get_highest_severity(self, detections: List[Dict]) -> str:
        if not detections:
            return "none"
        
        severity_levels = {"light": 1, "moderate": 2, "severe": 3}
        max_severity_value = max([severity_levels.get(d["severity"], 0) for d in detections])
        
        for severity, value in severity_levels.items():
            if value == max_severity_value:
                return severity
        
        return "light"
    
    def annotate_image(self, image: Union[str, np.ndarray, Image.Image], 
                       detections: List[Dict]) -> np.ndarray:
        img_array = self._preprocess_image(image)
        annotated_img = img_array.copy()
        
        colors = {
            "scratch": (0, 255, 0),
            "dent": (255, 165, 0),
            "broken_part": (255, 0, 0),
            "paint_damage": (255, 0, 255)
        }
        
        for detection in detections:
            bbox = detection["bbox"]
            damage_type = detection["type"]
            confidence = detection["confidence"]
            severity = detection["severity"]
            
            x1, y1, x2, y2 = bbox
            color = colors.get(damage_type, (255, 255, 0))
            
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            
            label = f"{damage_type.replace('_', ' ').title()}"
            sublabel = f"{severity.capitalize()} ({confidence:.2f})"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (sub_width, sub_height), _ = cv2.getTextSize(sublabel, font, font_scale-0.1, thickness-1)
            
            cv2.rectangle(annotated_img, 
                          (x1, y1 - text_height - sub_height - 10), 
                          (x1 + max(text_width, sub_width), y1), 
                          color, -1)
            
            cv2.putText(annotated_img, label, (x1, y1 - sub_height - 5), 
                        font, font_scale, (255, 255, 255), thickness)
            cv2.putText(annotated_img, sublabel, (x1, y1 - 2), 
                        font, font_scale-0.1, (255, 255, 255), thickness-1)
        
        return annotated_img
    
    def batch_detect(self, image_paths: List[str]) -> List[Dict]:
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.detect_damage(image_path)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "total_damages": 0,
                    "damages": []
                })
        
        return results
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        if 0.0 <= new_threshold <= 1.0:
            self.confidence_threshold = new_threshold
            logger.info(f"Confidence threshold updated to {new_threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

def main():
    detector = CarDamageDetector(confidence_threshold=0.5)
    
    print("CarDamageDetector initialized successfully")
    print(f"Device: {detector.device}")
    print(f"Model: {detector.model_path}")
    print(f"Confidence threshold: {detector.confidence_threshold}")

if __name__ == "__main__":
    main()