package com.pragyan.kernelguardians.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector

/**
 * On-device object detector using EfficientDet-Lite2 via the TFLite Task Library.
 *
 * Acceleration strategy (tried in order):
 *   1. NNAPI  — routes to Qualcomm Hexagon DSP / Adreno GPU via Android's
 *               Neural Networks API. Best real-world speedup on modern Snapdragon.
 *   2. CPU (XNNPACK, 4 threads) — reliable fallback on any device.
 *
 * Frame skipping to hit 30 fps is handled upstream in CameraAnalyzer;
 * IoUTracker's per-track Kalman filters bridge the in-between frames.
 */
class EfficientDetDetector(context: Context) {

    private val tag = "EfficientDetDetector"

    companion object {
        const val MODEL_FILE      = "efficientdet_lite0.tflite"
        const val MAX_RESULTS     = 10          // fewer results → less post-processing overhead
        const val SCORE_THRESHOLD = 0.40f       // slightly higher → fewer low-confidence junk boxes
    }

    private val detector: ObjectDetector
    private val iouTracker = IoUTracker()

    init {
        detector = buildDetector(context, useNnapi = true)
            ?: buildDetector(context, useNnapi = false)
            ?: error("Failed to create ObjectDetector on CPU — should never happen")
    }

    private fun buildDetector(context: Context, useNnapi: Boolean): ObjectDetector? {
        return try {
            val baseOptions = if (useNnapi) {
                BaseOptions.builder()
                    .useNnapi()                  // Hexagon DSP / Adreno GPU via Android NNAPI
                    .setNumThreads(2)            // NNAPI handles parallelism internally
                    .build()
            } else {
                BaseOptions.builder()
                    .setNumThreads(4)            // XNNPACK on 4 CPU threads
                    .build()
            }

            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(MAX_RESULTS)
                .setScoreThreshold(SCORE_THRESHOLD)
                .setBaseOptions(baseOptions)
                .build()

            val det = ObjectDetector.createFromFileAndOptions(context, MODEL_FILE, options)
            Log.d(tag, "EfficientDet-Lite2 loaded — NNAPI=$useNnapi")
            det
        } catch (e: Exception) {
            Log.w(tag, "Could not create detector (NNAPI=$useNnapi): ${e.message}")
            null
        }
    }

    /**
     * Run detection on [bitmap].
     * Returns Kalman-smoothed results with stable tracking IDs and appearance embeddings.
     * Embeddings are computed from raw EfficientDet boxes before IoU tracking,
     * so they reflect the detector's exact crop — not the Kalman-smoothed position.
     */
    fun detect(bitmap: Bitmap, imageWidth: Int, imageHeight: Int): List<DetectionResult> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val detections  = detector.detect(tensorImage)

        val rawBoxes = detections.mapNotNull { det ->
            val cat = det.categories.firstOrNull() ?: return@mapNotNull null
            Pair(det.boundingBox, Pair(cat.label ?: "unknown", cat.score))
        }

        // Compute appearance embeddings from raw bounding boxes BEFORE IoU tracking
        val embeddings: List<FloatArray?> = rawBoxes.map { (box, _) ->
            runCatching { AppearanceEmbedder.embed(bitmap, box) }.getOrNull()
        }

        return iouTracker.update(rawBoxes, embeddings)
    }

    fun resetTracking() = iouTracker.reset()

    fun close() = detector.close()
}
