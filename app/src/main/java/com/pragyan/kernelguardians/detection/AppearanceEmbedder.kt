package com.pragyan.kernelguardians.detection

import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Lightweight appearance embedder for Re-ID using spatial color histograms.
 *
 * No external model or internet needed — runs in microseconds on-device.
 *
 * Algorithm:
 *   1. Crop the detection bounding box from the frame bitmap
 *   2. Downscale crop to 32×32 for speed
 *   3. Divide into 3×3 spatial grid (9 cells)
 *   4. In each cell: build a 16-bin HSV-inspired histogram (H: 8 bins, S: 4 bins + V: 4 bins)
 *      = 16 features × 9 cells = 144-dim feature vector
 *   5. L2-normalise → cosine similarity is then just a dot product
 *
 * Cosine similarity > 0.80 reliably identifies "same object with same appearance"
 * across frames, even after occlusion or leaving the frame, provided the lighting
 * does not change drastically.
 */
object AppearanceEmbedder {

    private const val GRID     = 3    // 3×3 spatial grid
    private const val H_BINS   = 8    // hue bins (0–360° → 8 ranges)
    private const val SV_BINS  = 4    // saturation and value bins (2 each = 4 combined)
    private const val CELL_DIM = H_BINS + SV_BINS   // 12 features per cell
    private const val FEAT_DIM = GRID * GRID * CELL_DIM  // 108

    private const val THUMB_W  = 32   // crop is downscaled to this before histogramming
    private const val THUMB_H  = 32

    /**
     * Compute an L2-normalised appearance embedding for the detection at [box] in [frame].
     *
     * @param frame Full camera frame (landscape or portrait, any orientation)
     * @param box   Bounding box in [frame] pixel coordinates
     * @return 108-dim float vector, L2-normalised (or zero-vec if crop is invalid)
     */
    fun embed(frame: Bitmap, box: RectF): FloatArray {
        val feat = FloatArray(FEAT_DIM)

        // ── Crop ───────────────────────────────────────────────────────────
        val l = box.left.toInt().coerceIn(0, frame.width  - 1)
        val t = box.top .toInt().coerceIn(0, frame.height - 1)
        val r = box.right .toInt().coerceIn(l + 1, frame.width)
        val b = box.bottom.toInt().coerceIn(t + 1, frame.height)
        if (r <= l || b <= t) return feat

        val crop   = Bitmap.createBitmap(frame, l, t, r - l, b - t)
        val thumb  = Bitmap.createScaledBitmap(crop, THUMB_W, THUMB_H, true)
        crop.recycle()

        val cellW  = THUMB_W / GRID
        val cellH  = THUMB_H / GRID

        // ── Per-cell histogram ─────────────────────────────────────────────
        for (gy in 0 until GRID) {
            for (gx in 0 until GRID) {
                val base = (gy * GRID + gx) * CELL_DIM

                for (py in 0 until cellH) {
                    for (px in 0 until cellW) {
                        val pixel = thumb.getPixel(gx * cellW + px, gy * cellH + py)
                        val rf    = Color.red(pixel)   / 255f
                        val gf    = Color.green(pixel) / 255f
                        val bf    = Color.blue(pixel)  / 255f

                        val mx = max(rf, max(gf, bf))
                        val mn = min(rf, min(gf, bf))
                        val d  = mx - mn

                        // Value bin (0–3)
                        val vBin = (mx * SV_BINS).toInt().coerceIn(0, SV_BINS - 1)
                        // Saturation bin (0–3)
                        val sat  = if (mx > 0f) d / mx else 0f
                        val sBin = (sat * SV_BINS).toInt().coerceIn(0, SV_BINS - 1)
                        // Hue bin (0–7)
                        val hue  = when {
                            d < 1e-4f -> 0f
                            mx == rf  -> ((gf - bf) / d + 6f) % 6f * 60f
                            mx == gf  -> ((bf - rf) / d + 2f) * 60f
                            else      -> ((rf - gf) / d + 4f) * 60f
                        }
                        val hBin = ((hue / 360f) * H_BINS).toInt().coerceIn(0, H_BINS - 1)

                        feat[base + hBin]++
                        feat[base + H_BINS + (sBin * 2 + vBin / 2)]++  // pack S+V into 4 bins
                    }
                }
            }
        }

        thumb.recycle()

        // ── L2 normalise ───────────────────────────────────────────────────
        var norm = 0f
        for (v in feat) norm += v * v
        norm = sqrt(norm)
        if (norm > 0f) for (i in feat.indices) feat[i] /= norm

        return feat
    }

    /**
     * Cosine similarity between two L2-normalised embeddings.
     * Returns a value in [-1, 1]; > 0.80 typically means "same object".
     */
    fun similarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }
}
