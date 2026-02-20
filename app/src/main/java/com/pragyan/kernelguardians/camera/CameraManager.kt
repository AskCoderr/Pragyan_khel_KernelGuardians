package com.pragyan.kernelguardians.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.PointF
import android.util.Log
import android.view.MotionEvent
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import com.pragyan.kernelguardians.utils.CoordinateUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Encapsulates all CameraX lifecycle logic:
 *  - Binds [Preview] + [ImageAnalysis] use cases.
 *  - Handles "tap-to-focus" via [MeteringAction].
 *  - Owns the [CameraAnalyzer] and [ObjectTracker].
 *  - Passes results back via [onAnalysisResult].
 */
class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val tracker: ObjectTracker,
    private val onAnalysisResult: (
        state: TrackingState,
        box: android.graphics.RectF?,
        label: String,
        confidence: Float,
        fps: Float
    ) -> Unit,
    private val onTapLocked: (x: Float, y: Float, success: Boolean) -> Unit
) {

    private val TAG = "CameraManager"

    private var camera: Camera? = null
    private var analyzer: CameraAnalyzer? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    // Last detections for tap-to-lock
    private var lastDetections: List<Pair<com.google.mlkit.vision.objects.DetectedObject, android.graphics.RectF>> = emptyList()
    private var imageWidth  = 1
    private var imageHeight = 1

    @SuppressLint("ClickableViewAccessibility", "UnsafeOptInUsageError")
    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Image Analysis — RGBA_8888 for easier Bitmap conversion
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            val cameraAnalyzer = CameraAnalyzer(tracker) { detections, state, box, label, conf, fps ->
                lastDetections  = detections
                imageWidth      = detections.firstOrNull()?.first?.boundingBox?.width() ?: imageWidth
                onAnalysisResult(state, box, label, conf, fps)
            }.also {
                it.viewW = previewView.width.coerceAtLeast(1)
                it.viewH = previewView.height.coerceAtLeast(1)
                analyzer = it
            }

            imageAnalysis.setAnalyzer(cameraExecutor, cameraAnalyzer)

            val selector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    lifecycleOwner, selector, preview, imageAnalysis
                )
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }

            // Update analyzer view dims when layout is ready
            previewView.post {
                cameraAnalyzer.viewW = previewView.width.coerceAtLeast(1)
                cameraAnalyzer.viewH = previewView.height.coerceAtLeast(1)
            }

            // ── Tap-to-focus ──────────────────────────────────────────────
            previewView.setOnTouchListener { view, event ->
                if (event.action == MotionEvent.ACTION_UP) {
                    val tapX = event.x
                    val tapY = event.y

                    // 1. Trigger CameraX hardware autofocus
                    val factory       = previewView.meteringPointFactory
                    val meteringPoint = CoordinateUtils.toMeteringPoint(factory, tapX, tapY)
                    val action        = FocusMeteringAction.Builder(meteringPoint).build()
                    camera?.cameraControl?.startFocusAndMetering(action)

                    // 2. Attempt to lock onto nearest ML Kit object
                    val success = tracker.onUserTap(
                        touchPoint   = PointF(tapX, tapY),
                        detections   = lastDetections,
                        imageWidth   = cameraAnalyzer.viewW,
                        imageHeight  = cameraAnalyzer.viewH,
                        viewWidth    = previewView.width,
                        viewHeight   = previewView.height
                    )

                    onTapLocked(tapX, tapY, success)
                    view.performClick()
                }
                true
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun shutdown() {
        analyzer?.shutdown()
        cameraExecutor.shutdown()
    }
}
