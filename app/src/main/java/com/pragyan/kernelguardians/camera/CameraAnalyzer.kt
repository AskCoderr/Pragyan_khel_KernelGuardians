package com.pragyan.kernelguardians.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.RectF
import android.graphics.YuvImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.pragyan.kernelguardians.detection.DetectionResult
import com.pragyan.kernelguardians.detection.EfficientDetDetector
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import com.pragyan.kernelguardians.utils.CoordinateUtils
import com.pragyan.kernelguardians.utils.LowLightEnhancer
import java.io.ByteArrayOutputStream
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

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
        private const val MIN_FRAME_GAP_MS = 33L
    }

    // ── EfficientDet-Lite2 replaces ML Kit Object Detection ──────────────────
    private val detector = EfficientDetDetector(context)

    private val isProcessing    = AtomicBoolean(false)
    private val lastProcessedAt = AtomicLong(0L)

    private var fpsFrameCount   = 0
    private var fpsWindowStart  = System.currentTimeMillis()
    private var currentFps      = 0f

    var viewW: Int = 1
    var viewH: Int = 1

    @SuppressLint("UnsafeOptInUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()

        if (isProcessing.get() || (now - lastProcessedAt.get()) < MIN_FRAME_GAP_MS) {
            imageProxy.close()
            return
        }

        isProcessing.set(true)
        lastProcessedAt.set(now)

        val rotDeg = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap() ?: run {
            imageProxy.close()
            isProcessing.set(false)
            return
        }

        val imageWidth  = imageProxy.width
        val imageHeight = imageProxy.height
        imageProxy.close()

        // Apply low-light enhancement to both the ML input and the display frame
        val enhanced = LowLightEnhancer.enhance(bitmap)

        // Run EfficientDet-Lite2 — returns DetectionResult list with stable IDs
        val detectionResults = detector.detect(enhanced, imageWidth, imageHeight)

        // Map boxes from image space → view/overlay space
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

@SuppressLint("UnsafeOptInUsageError")
private fun ImageProxy.toBitmap(): Bitmap? {
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

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out      = ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 85, out)
    val bytes    = out.toByteArray()
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
}
