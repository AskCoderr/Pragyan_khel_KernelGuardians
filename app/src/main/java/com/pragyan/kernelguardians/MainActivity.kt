package com.pragyan.kernelguardians

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.pragyan.kernelguardians.camera.CameraManager
import com.pragyan.kernelguardians.databinding.ActivityMainBinding
import com.pragyan.kernelguardians.rendering.BlurProcessor
import com.pragyan.kernelguardians.tracking.ObjectTracker
import com.pragyan.kernelguardians.tracking.TrackingState
import org.opencv.android.OpenCVLoader

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    // Core components
    private lateinit var tracker:       ObjectTracker
    private lateinit var cameraManager: CameraManager

    // State
    private var bgBlurEnabled = false

    // Blur worker thread — keeps bitmap processing off the main thread
    private val blurThread = HandlerThread("BlurWorker").also { it.start() }
    private val blurHandler = Handler(blurThread.looper)

    // Last known tracking box (in view coords) — updated every frame
    @Volatile private var lastTrackedBox: RectF? = null

    // Camera rotation — read from ImageAnalysis once available
    // Not needed for visual rotation since we use ImageView.rotation
    @Volatile private var frameRotationDegrees: Float = 0f

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

        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
        )

        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV init failed — low-light enhancement disabled", Toast.LENGTH_SHORT).show()
        }

        setupButtons()

        if (hasCameraPermission()) startCamera() else requestPermission()
    }

    override fun onDestroy() {
        blurThread.quitSafely()
        if (::cameraManager.isInitialized) cameraManager.shutdown()
        super.onDestroy()
    }

    // ── Buttons ────────────────────────────────────────────────────────────

    private fun setupButtons() {
        binding.btnClearLock.setOnClickListener {
            tracker.clearLock()
            lastTrackedBox = null
            binding.overlayView.update(null, TrackingState.IDLE)
            binding.tvTrackingStatus.text = getString(R.string.tap_to_track)
            binding.focusRingView.hide()
            // Clear blur overlay when lock is cleared
            binding.blurOverlayView.setImageBitmap(null)
        }

        binding.btnToggleBlur.setOnClickListener {
            bgBlurEnabled = !bgBlurEnabled
            val label = if (bgBlurEnabled) getString(R.string.bg_blur_on) else getString(R.string.bg_blur_off)
            binding.btnToggleBlur.text = label
            if (!bgBlurEnabled) {
                binding.blurOverlayView.visibility = View.GONE
                binding.blurOverlayView.setImageBitmap(null)
            }
        }
    }

    // ── Camera ─────────────────────────────────────────────────────────────

    private fun startCamera() {
        tracker = ObjectTracker()

        cameraManager = CameraManager(
            context          = this,
            lifecycleOwner   = this,
            previewView      = binding.previewView,
            tracker          = tracker,
            onAnalysisResult = { state, box, label, conf, fps, frameBitmap, rotDeg ->

                // Store latest box and rotation
                lastTrackedBox = box
                frameRotationDegrees = rotDeg.toFloat()


                // Update UI on main thread
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

                // Run CPU blur on worker thread if enabled and box is available
                if (bgBlurEnabled && box != null) {
                    blurHandler.post {
                        processBlurFrame(frameBitmap, box)
                    }
                } else if (bgBlurEnabled) {
                    // No tracked object: still show camera sharp (just reveal the overlay with null box)
                    blurHandler.post {
                        processBlurFrame(frameBitmap, null)
                    }
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

    // ── Blur processing ────────────────────────────────────────────────────

    /**
     * Called on [blurThread].
     * 1. Rotate camera bitmap to portrait
     * 2. Map bounding box from view coords to rotated bitmap coords
     * 3. Run [BlurProcessor.process]
     * 4. Post result to [blurOverlayView] on main thread
     */
    private fun processBlurFrame(raw: Bitmap, viewBox: RectF?) {
        try {
            // Map bounding box from view space → raw bitmap space
            // The raw bitmap is landscape (sensor native), so we map box using
            // view dims to bitmap dims directly (both may differ in aspect)
            val bitmapBox: RectF? = if (viewBox != null) {
                val vw = binding.previewView.width.coerceAtLeast(1).toFloat()
                val vh = binding.previewView.height.coerceAtLeast(1).toFloat()
                val bw = raw.width.toFloat()
                val bh = raw.height.toFloat()

                // The preview maps sensor landscape to portrait via rotation.
                // After 90° rotation: preview-Y corresponds to bitmap-X, preview-X to bitmap-Y
                val rot = frameRotationDegrees
                if (rot == 90f || rot == 270f) {
                    // Swap X/Y axes
                    RectF(
                        viewBox.top    / vh * bw,
                        viewBox.left   / vw * bh,
                        viewBox.bottom / vh * bw,
                        viewBox.right  / vw * bh
                    )
                } else {
                    RectF(
                        viewBox.left   / vw * bw,
                        viewBox.top    / vh * bh,
                        viewBox.right  / vw * bw,
                        viewBox.bottom / vh * bh
                    )
                }
            } else null

            // Produce composited blurred frame on raw bitmap
            val composited = BlurProcessor.process(raw, bitmapBox)

            // Show result — visually rotate the ImageView to match sensor orientation
            runOnUiThread {
                if (bgBlurEnabled) {
                    binding.blurOverlayView.rotation = frameRotationDegrees
                    binding.blurOverlayView.visibility = View.VISIBLE
                    binding.blurOverlayView.setImageBitmap(composited)
                }
            }
        } catch (e: Exception) {
            // Silently ignore — next frame will retry
        }
    }

    // ── Permissions ────────────────────────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun requestPermission() =
        permissionLauncher.launch(arrayOf(Manifest.permission.CAMERA))
}
