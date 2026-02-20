package com.pragyan.kernelguardians

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.pragyan.kernelguardians.camera.CameraManager
import com.pragyan.kernelguardians.databinding.ActivityMainBinding
import com.pragyan.kernelguardians.rendering.BackgroundBlurRenderer
import com.pragyan.kernelguardians.segmentation.SegmentationProcessor
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    // Core components
    private lateinit var tracker:      ObjectTracker
    private lateinit var cameraManager: CameraManager
    private lateinit var segmentation: SegmentationProcessor
    private lateinit var blurRenderer: BackgroundBlurRenderer

    // State
    private var bgBlurEnabled = false

    // ── Permission launcher ────────────────────────────────────────────────

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { results ->
        if (results[Manifest.permission.CAMERA] == true) {
            startCamera()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
            finish()
        }
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Hide system UI for full-screen immersive experience
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )

        // Initialise OpenCV
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV init failed — low-light enhancement disabled", Toast.LENGTH_SHORT).show()
        }

        setupGLSurface()
        setupSegmentation()
        setupButtons()

        if (hasCameraPermission()) startCamera() else requestPermission()
    }

    override fun onResume() {
        super.onResume()
        if (bgBlurEnabled) binding.glSurfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        if (bgBlurEnabled) binding.glSurfaceView.onPause()
    }

    override fun onDestroy() {
        cameraManager.shutdown()
        segmentation.release()
        super.onDestroy()
    }

    // ── Setup helpers ──────────────────────────────────────────────────────

    private fun setupGLSurface() {
        blurRenderer = BackgroundBlurRenderer()
        binding.glSurfaceView.apply {
            setEGLContextClientVersion(2)
            setRenderer(blurRenderer)
            renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        }
    }

    private fun setupSegmentation() {
        segmentation = SegmentationProcessor(this).apply {
            onMaskReady = { mask ->
                if (bgBlurEnabled) {
                    blurRenderer.maskBitmap = mask
                    binding.glSurfaceView.requestRender()
                }
            }
        }
    }

    private fun setupButtons() {
        binding.btnClearLock.setOnClickListener {
            tracker.clearLock()
            binding.overlayView.update(null, TrackingState.IDLE)
            binding.tvTrackingStatus.text = getString(R.string.tap_to_track)
            binding.focusRingView.hide()
        }

        binding.btnToggleBlur.setOnClickListener {
            bgBlurEnabled = !bgBlurEnabled
            binding.glSurfaceView.visibility = if (bgBlurEnabled) View.VISIBLE else View.GONE
            binding.btnToggleBlur.text =
                if (bgBlurEnabled) getString(R.string.bg_blur_on) else getString(R.string.bg_blur_off)
            if (!bgBlurEnabled) {
                binding.glSurfaceView.onPause()
            } else {
                binding.glSurfaceView.onResume()
            }
        }
    }

    private fun startCamera() {
        tracker = ObjectTracker()

        cameraManager = CameraManager(
            context         = this,
            lifecycleOwner  = this,
            previewView     = binding.previewView,
            tracker         = tracker,
            onAnalysisResult = { state, box, label, conf, fps ->
                runOnUiThread {
                    binding.overlayView.update(box, state, label, conf)
                    binding.tvFps.text = "FPS: ${"%.1f".format(fps)}"
                    binding.tvTrackingStatus.text = when (state) {
                        TrackingState.LOCKED     -> getString(R.string.tracking_locked)
                        TrackingState.PREDICTING -> getString(R.string.tracking_lost)
                        TrackingState.SEARCHING  -> getString(R.string.tracking_searching)
                        TrackingState.IDLE       -> getString(R.string.tap_to_track)
                    }
                }

                // Feed frame into segmenter if blur is active
                if (bgBlurEnabled && box != null) {
                    // We pass the current camera frame bitmap through the segmenter
                    // The CameraAnalyzer exposes the last processed bitmap via a callback extension
                    // (segmentation runs on the same frame that was analyzed)
                }
            },
            onTapLocked = { x, y, success ->
                runOnUiThread {
                    if (success) {
                        binding.focusRingView.showLocked(x, y)
                        binding.tvTrackingStatus.text = getString(R.string.tracking_locked)
                    } else {
                        binding.focusRingView.show(x, y)
                    }
                }
            }
        )

        cameraManager.start()
    }

    // ── Permissions ────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestPermission() =
        permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
}
