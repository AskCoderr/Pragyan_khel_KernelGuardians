package com.pragyan.kernelguardians.utils

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.CLAHE
import org.opencv.imgproc.Imgproc

/**
 * Low-light enhancement using OpenCV CLAHE (Contrast Limited Adaptive Histogram Equalization).
 *
 * Applied on the luminance (L) channel of the LAB colour space so that
 * chrominance (colour) is untouched. This avoids the colour-shift artefacts
 * common with plain histogram equalisation.
 *
 * Uses a lightweight heuristic to skip the enhancement when the frame is
 * already well-lit, saving CPU cycles and preserving FPS.
 */
object LowLightEnhancer {

    private const val CLIP_LIMIT   = 2.0
    private const val TILE_GRID_W  = 8
    private const val TILE_GRID_H  = 8
    /** Mean luminance below this threshold → apply CLAHE */
    private const val LOW_LIGHT_THRESHOLD = 80.0

    private var clahe: CLAHE? = null

    private fun getOrCreateClahe(): CLAHE {
        return clahe ?: Imgproc.createCLAHE(CLIP_LIMIT, Size(TILE_GRID_W.toDouble(), TILE_GRID_H.toDouble())).also { clahe = it }
    }

    /**
     * Enhance [bitmap] if it is considered low-light.
     * @return The (possibly enhanced) bitmap. May return the input unchanged.
     */
    fun enhance(bitmap: Bitmap): Bitmap {
        val src = Mat()
        Utils.bitmapToMat(bitmap, src)

        // Convert to LAB
        val lab = Mat()
        Imgproc.cvtColor(src, lab, Imgproc.COLOR_RGBA2RGB)
        val labColor = Mat()
        Imgproc.cvtColor(lab, labColor, Imgproc.COLOR_RGB2Lab)

        // Extract L channel
        val channels = ArrayList<Mat>()
        org.opencv.core.Core.split(labColor, channels)
        val lChannel = channels[0]

        // Check mean luminance — skip if already bright
        val mean = org.opencv.core.Core.mean(lChannel)
        if (mean.`val`[0] >= LOW_LIGHT_THRESHOLD) {
            src.release(); lab.release(); labColor.release()
            lChannel.release()
            return bitmap  // No-op — frame is bright enough
        }

        // Apply CLAHE to L channel
        val lEnhanced = Mat()
        getOrCreateClahe().apply(lChannel, lEnhanced)
        channels[0] = lEnhanced

        // Merge back and convert to RGBA
        org.opencv.core.Core.merge(channels, labColor)
        val result = Mat()
        Imgproc.cvtColor(labColor, result, Imgproc.COLOR_Lab2RGB)
        Imgproc.cvtColor(result, result, Imgproc.COLOR_RGB2RGBA)

        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        Utils.matToBitmap(result, output)

        // Release Mats
        src.release(); lab.release(); labColor.release()
        lChannel.release(); lEnhanced.release(); result.release()
        channels.forEach { it.release() }

        return output
    }

    fun release() {
        clahe?.collectGarbage()
        clahe = null
    }
}
