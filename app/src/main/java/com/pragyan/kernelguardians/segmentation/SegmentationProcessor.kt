package com.pragyan.kernelguardians.segmentation

import android.content.Context
import android.graphics.Bitmap
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenter
import com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
import java.nio.ByteBuffer

/**
 * Wraps MediaPipe Image Segmenter for real-time subject/background separation.
 *
 * Uses the bundled "selfie_multiclass_256x256" model (ships with the MediaPipe
 * tasks-vision AAR — no external download required).
 *
 * Runs in LIVE_STREAM mode to match CameraX's continuous frame pipeline.
 */
class SegmentationProcessor(context: Context) {

    companion object {
        // Model bundled in MediaPipe tasks-vision AAR
        private const val MODEL_NAME = "selfie_multiclass_256x256.tflite"
        /** Mask byte value that corresponds to the foreground/subject class */
        private const val SUBJECT_CATEGORY = 1
    }

    /** Callback carrying the latest mask bitmap to the renderer. */
    var onMaskReady: ((maskBitmap: Bitmap) -> Unit)? = null

    private var segmenter: ImageSegmenter? = null
    private var frameTs = 0L

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_NAME)
            .build()

        val options = ImageSegmenter.ImageSegmenterOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setOutputCategoryMask(true)
            .setOutputConfidenceMasks(false)
            .setResultListener { result, _ -> handleResult(result) }
            .build()

        try {
            segmenter = ImageSegmenter.createFromOptions(context, options)
        } catch (e: Exception) {
            // Model may not be bundled in older AAR versions — degrade gracefully
            e.printStackTrace()
        }
    }

    /**
     * Submit a frame for asynchronous segmentation.
     * Result arrives via [onMaskReady].
     */
    fun process(bitmap: Bitmap) {
        val seg = segmenter ?: return
        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()
        frameTs += 33L   // Monotonic synthetic timestamp
        try {
            seg.segmentAsync(mpImage, frameTs)
        } catch (_: Exception) {}
    }

    private fun handleResult(result: ImageSegmenterResult) {
        val masks = result.categoryMask() ?: return
        if (!masks.isPresent) return

        val mask: MPImage = masks.get()
        val buffer: ByteBuffer = ByteBufferExtractor.extract(mask)
        val width  = mask.width
        val height = mask.height

        // Convert category mask to RGBA bitmap (white = subject, black = bg)
        val pixels = IntArray(width * height)
        buffer.rewind()
        for (i in pixels.indices) {
            val category = buffer.get().toInt() and 0xFF
            pixels[i] = if (category == SUBJECT_CATEGORY)
                0xFFFFFFFF.toInt()  // foreground
            else
                0xFF000000.toInt()  // background
        }

        val maskBitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        onMaskReady?.invoke(maskBitmap)
    }

    fun release() {
        segmenter?.close()
        segmenter = null
    }
}
