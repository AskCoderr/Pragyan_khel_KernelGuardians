package com.pragyan.kernelguardians.camera

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import android.view.MotionEvent
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.pragyan.kernelguardians.detection.DetectionResult
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import com.pragyan.kernelguardians.utils.CoordinateUtils
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val tracker: ObjectTracker,
    private val onAnalysisResult: (
        state: TrackingState,
        box: RectF?,
        label: String,
        confidence: Float,
        fps: Float,
        frameBitmap: Bitmap,
        rotDeg: Int
    ) -> Unit,
    private val onTapLocked: (x: Float, y: Float, success: Boolean) -> Unit
) {

    private val TAG = "CameraManager"

    private var camera: Camera? = null
    private var analyzer: CameraAnalyzer? = null
    private val cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    private var lastDetections: List<Pair<DetectionResult, RectF>> = emptyList()
    private var imageWidth  = 1

    @SuppressLint("ClickableViewAccessibility", "UnsafeOptInUsageError")
    fun start() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()

            val cameraAnalyzer = CameraAnalyzer(context, tracker) { detections, state, box, label, conf, fps, frameBitmap, rotDeg ->
                lastDetections = detections
                imageWidth = frameBitmap.width
                onAnalysisResult(state, box, label, conf, fps, frameBitmap, rotDeg)
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

            previewView.post {
                cameraAnalyzer.viewW = previewView.width.coerceAtLeast(1)
                cameraAnalyzer.viewH = previewView.height.coerceAtLeast(1)
            }

            previewView.setOnTouchListener { view, event ->
                if (event.action == MotionEvent.ACTION_UP) {
                    val tapX = event.x
                    val tapY = event.y

                    val factory       = previewView.meteringPointFactory
                    val meteringPoint = CoordinateUtils.toMeteringPoint(factory, tapX, tapY)
                    val action        = FocusMeteringAction.Builder(meteringPoint).build()
                    camera?.cameraControl?.startFocusAndMetering(action)

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
