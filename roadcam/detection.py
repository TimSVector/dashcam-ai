"""CPU-only vehicle detector. Heavy dependencies load only when constructed."""
import logging
import cv2
from .models import Box, Detection


class TorchvisionVehicleDetector:
    def __init__(self, confidence: float = 0.5, threads: int = 4):
        if not 0 <= confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        import torch
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )
        torch.set_num_threads(threads)
        self.torch = torch
        self.confidence = confidence
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self.categories = weights.meta["categories"]
        self.transform = weights.transforms()
        logging.getLogger(__name__).info("Loading SSDLite COCO weights on CPU (first run may download weights)")
        self.model = ssdlite320_mobilenet_v3_large(weights=weights).to("cpu").eval()

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1)
        with self.torch.inference_mode():
            result = self.model([self.transform(tensor)])[0]
        detections = []
        for box, label, score in zip(result["boxes"].tolist(), result["labels"].tolist(), result["scores"].tolist()):
            name = self.categories[label]
            if name in {"car", "truck", "bus", "motorcycle"} and score >= self.confidence:
                b = Box(*(round(v) for v in box)).clipped(image.shape[1], image.shape[0])
                if b.x2 > b.x1 and b.y2 > b.y1:
                    detections.append(Detection(b, name, score))
        return detections
