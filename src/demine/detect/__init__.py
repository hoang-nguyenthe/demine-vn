"""Tầng hai — phát hiện hố bom trên ảnh vệ tinh lịch sử."""

from .interface import BaseDetector, DetectionOutput, build_detector, select_backend

__all__ = ["BaseDetector", "DetectionOutput", "build_detector", "select_backend"]
