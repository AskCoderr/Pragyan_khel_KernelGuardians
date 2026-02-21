package com.pragyan.kernelguardians.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.pragyan.kernelguardians.detection.DetectionResult
import com.pragyan.kernelguardians.detection.EfficientDetDetector
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import com.pragyan.kernelguardians.utils.CoordinateUtils
import com.pragyan.kernelguardians.utils.LowLightEnhancer
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * Analyses camera frames.
 *
 * FPS strategy — two-layer approach:
 *
 *   Layer 1 — ML inference skip:
 *     Run EfficientDet every [INFERENCE_EVERY_N_FRAMES] frames.
 *     In-between frames are forwarded instantly with the PREVIOUS detection
 *     result, while each track's Kalman filter predicts the updated box.
 *     This keeps the preview and overlay smooth at 30 fps even if inference
 *     only runs at 10 fps.
 *
 *   Layer 2 — Throttle gate:
 *     Hard-caps at one analysis per [MIN_FRAME_GAP_MS] to avoid queuing
 *     on slow devices.
 *
 * Pipeline per frame:
 *   ImageProxy → Bitmap → (optional CLAHE low-light) → EfficientDet →
 *   IoUTracker (SORT, Kalman inside) → CoordinateUtils → ObjectTracker → UI
 */
class CameraAnalyzer(
    private val context: Context,
    private val tracker: ObjectTracker,
    private val onResult: (
        detections: List<Pair<DetectionResult, RectF>>,
        state: TrackingState,
        box: RectF?,
        label: String,
        confidence: Float,
        fps: Float,
        frameBitmap: Bitmap,
        rotDeg: Int
    ) -> Unit
) : ImageAnalysis.Analyzer {

    companion object {
        /** Throttle: skip frames if previous inference hasn't finished */
        private const val MIN_FRAME_GAP_MS = 0L  // 0 = no artificial cap, NNAPI drives the rate
    }

    private val detector     = EfficientDetDetector(context)
    private val isProcessing = AtomicBoolean(false)
    private val lastAnalyzed = AtomicLong(0L)

    // FPS counter
    private var fpsFrameCount  = 0
    private var fpsWindowStart = System.currentTimeMillis()
    private var currentFps     = 0f

    var viewW: Int = 1
    var viewH: Int = 1

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()

        if (isProcessing.get() || (now - lastAnalyzed.get()) < MIN_FRAME_GAP_MS) {
            imageProxy.close()
            return
        }

        isProcessing.set(true)
        lastAnalyzed.set(now)

        val rotDeg = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap() ?: run {
            imageProxy.close()
            isProcessing.set(false)
            return
        }

        val imageWidth  = imageProxy.width
        val imageHeight = imageProxy.height
        imageProxy.close()

        // Full ML pipeline every frame — NNAPI/GPU drives the detection rate
        val enhanced         = LowLightEnhancer.enhance(bitmap)
        val detectionResults = detector.detect(enhanced, imageWidth, imageHeight)

        val mapped = detectionResults.map { result ->
            val viewBox = CoordinateUtils.mapBoundingBoxToView(
                box         = result.boundingBox,
                imageWidth  = imageWidth,
                imageHeight = imageHeight,
                viewWidth   = viewW,
                viewHeight  = viewH,
                rotDeg      = rotDeg
            )
            Pair(result, viewBox)
        }

        tracker.processDetections(mapped)

        // FPS accounting (counts every frame, not just inference frames)
        fpsFrameCount++
        val elapsed = System.currentTimeMillis() - fpsWindowStart
        if (elapsed >= 1000L) {
            currentFps     = fpsFrameCount * 1000f / elapsed
            fpsFrameCount  = 0
            fpsWindowStart = System.currentTimeMillis()
        }

        onResult(
            mapped,
            tracker.currentState,
            tracker.currentBox,
            tracker.currentLabel,
            tracker.currentConfidence,
            currentFps,
            bitmap,
            rotDeg
        )

        isProcessing.set(false)
    }

    fun shutdown() {
        detector.close()
        LowLightEnhancer.release()
    }
}

// ── ImageProxy → Bitmap ───────────────────────────────────────────────────────

@SuppressLint("UnsafeOptInUsageError")
private fun ImageProxy.toBitmap(): Bitmap? = runCatching {
    // Use CameraX's built-in YUV→Bitmap converter which avoids the JPEG round-trip
    toBitmapInternal()
}.getOrElse {
    // Fallback: manual NV21 → JPEG → Bitmap
    yuvToJpegBitmap()
}

@SuppressLint("UnsafeOptInUsageError")
private fun ImageProxy.toBitmapInternal(): Bitmap {
    // CameraX 1.3+ exposes this directly
    return this.toBitmap()
}

@SuppressLint("UnsafeOptInUsageError")
private fun ImageProxy.yuvToJpegBitmap(): Bitmap? {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)
    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
        nv21, android.graphics.ImageFormat.NV21, width, height, null
    )
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 80, out)
    val bytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
