package com.pragyan.kernelguardians.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.vision.detector.ObjectDetector

/**
 * On-device object detector using EfficientDet-Lite2 via the TFLite Task Library.
 *
 * The Task Library handles:
 *  - Model metadata reading (labels, input normalization)
 *  - Input resizing to 448×448
 *  - NMS and output tensor parsing automatically
 *
 * This replaces the previous raw-tensor approach which broke because the
 * MediaPipe model outputs pre-NMS anchor tensors ([1, 37629, 90]) rather
 * than the post-NMS 4-tensor format we assumed.
 *
 * Detects 90 COCO categories. Stable tracking IDs assigned by [IoUTracker].
 */
class EfficientDetDetector(context: Context) {

    private val tag = "EfficientDetDetector"

    companion object {
        const val MODEL_FILE      = "efficientdet_lite2.tflite"
        const val MAX_RESULTS     = 15
        const val SCORE_THRESHOLD = 0.35f
    }

    private val detector: ObjectDetector
    private val iouTracker = IoUTracker()

    init {
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(MAX_RESULTS)
            .setScoreThreshold(SCORE_THRESHOLD)
            .build()

        detector = ObjectDetector.createFromFileAndOptions(context, MODEL_FILE, options)
        Log.d(tag, "EfficientDet-Lite2 (Task Library) loaded")
    }

    /**
     * Run detection on [bitmap].
     * Returns results mapped into the original [imageWidth]×[imageHeight] space
     * with stable tracking IDs.
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): List<DetectionResult> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val detections  = detector.detect(tensorImage)

        val rawBoxes = detections.mapNotNull { det ->
            val cat   = det.categories.firstOrNull() ?: return@mapNotNull null
            val label = cat.label  ?: "unknown"
            val score = cat.score

            // Task Library returns boxes in the input bitmap's pixel coords
            val box = det.boundingBox   // RectF in bitmap pixels

            Pair(box, Pair(label, score))
        }

        return iouTracker.update(rawBoxes)
    }

    fun resetTracking() = iouTracker.reset()

    fun close() = detector.close()
}
