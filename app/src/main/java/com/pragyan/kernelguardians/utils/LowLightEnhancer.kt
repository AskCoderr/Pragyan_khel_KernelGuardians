package com.pragyan.kernelguardians.utils

import android.graphics.Bitmap
import android.graphics.Color
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc

/**
 * Low-light enhancement using OpenCV CLAHE.
 *
 * Performance strategy:
 *   1. Fast luminance check using Android's Bitmap pixel sampling (no OpenCV Mat allocation).
 *      Samples a 16×16 grid of pixels → ~0.1ms, skips CLAHE if frame is bright.
 *   2. Only if actually dark: full CLAHE pipeline in LAB colour space (~15ms).
 *
 * Previously the code still ran 2× cvtColor + split + mean even on bright frames.
 * Now bright frames are rejected before touching OpenCV at all.
 */
object LowLightEnhancer {

    private const val CLIP_LIMIT          = 2.0
    private const val TILE_GRID_W         = 8
    private const val TILE_GRID_H         = 8
    /** Mean luminance (0–255) below which we apply CLAHE */
    private const val LOW_LIGHT_THRESHOLD = 80
    /** Sample grid for fast luminance check (samples = GRID × GRID) */
    private const val SAMPLE_GRID         = 16

    private var clahe: CLAHE? = null

    private fun getOrCreateClahe(): CLAHE =
        clahe ?: Imgproc.createCLAHE(
            CLIP_LIMIT, Size(TILE_GRID_W.toDouble(), TILE_GRID_H.toDouble())
        ).also { clahe = it }

    /**
     * Fast luminance estimation using pixel sampling — no Mat allocation.
     * Returns mean luminance (0–255).
     */
    private fun fastMeanLuminance(bitmap: Bitmap): Int {
        val w      = bitmap.width
        val h      = bitmap.height
        val stepX  = maxOf(1, w / SAMPLE_GRID)
        val stepY  = maxOf(1, h / SAMPLE_GRID)
        var sum    = 0L
        var count  = 0

        var y = stepY / 2
        while (y < h) {
            var x = stepX / 2
            while (x < w) {
                val pixel = bitmap.getPixel(x, y)
                // Rec. 601 luminance approximation (integer arithmetic)
                sum += (77 * Color.red(pixel) + 150 * Color.green(pixel) + 29 * Color.blue(pixel)) shr 8
                count++
                x += stepX
            }
            y += stepY
        }
        return if (count == 0) 255 else (sum / count).toInt()
    }

    /**
     * Enhance [bitmap] if it is considered low-light.
     * Returns the input unchanged if bright enough (fast path, no OpenCV).
     */
    fun enhance(bitmap: Bitmap): Bitmap {
        // ── Fast check — no Mat allocation ────────────────────────────────────
        if (fastMeanLuminance(bitmap) >= LOW_LIGHT_THRESHOLD) return bitmap

        // ── Full CLAHE pipeline (only on genuinely dark frames) ───────────────
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        val lab = Mat()
        Imgproc.cvtColor(src, lab, Imgproc.COLOR_RGBA2RGB)
        val labColor = Mat()
        Imgproc.cvtColor(lab, labColor, Imgproc.COLOR_RGB2Lab)

        val channels = ArrayList<Mat>()
        org.opencv.core.Core.split(labColor, channels)

        val lEnhanced = Mat()
        getOrCreateClahe().apply(channels[0], lEnhanced)
        channels[0].release()
        channels[0] = lEnhanced

        org.opencv.core.Core.merge(channels, labColor)
        val result = Mat()
        Imgproc.cvtColor(labColor, result, Imgproc.COLOR_Lab2RGB)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_RGB2RGBA)

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        Utils.matToBitmap(result, output)

        src.release(); lab.release(); labColor.release()
        lEnhanced.release(); result.release()
        channels.forEach { it.release() }

        return output
    }

    fun release() {
        clahe?.collectGarbage()
        clahe = null
    }
}
