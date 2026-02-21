package com.pragyan.kernelguardians.detection

import android.graphics.RectF

/**
 * Unified detection result used throughout the pipeline.
 * Replaces ML Kit's [DetectedObject] so the rest of the code is
 * model-agnostic.
 *
 * @param trackingId  Stable ID assigned by [IoUTracker] across frames
 * @param label       Human-readable COCO class name (e.g. "person", "clock")
 * @param confidence  Detection score 0.0–1.0
 * @param boundingBox Box in the coordinate space of the image passed to the detector
 */
data class DetectionResult(
    val trackingId: Int,
    val label: String,
    val confidence: Float,
    val boundingBox: RectF
)
