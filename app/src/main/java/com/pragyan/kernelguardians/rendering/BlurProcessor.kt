package com.pragyan.kernelguardians.rendering

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RadialGradient
import android.graphics.RectF
import android.graphics.Shader
import kotlin.math.max


/**
 * CPU-based portrait blur compositor.
 *
 * Produces a composited Bitmap each frame:
 *   1. Rotate camera bitmap to portrait orientation
 *   2. Downscale → iterative upscale ("stack blur" approximation) → strong smooth blur
 *   3. Composite: blurred full frame + sharp subject region (feathered bbox mask)
 *
 * No GL / RenderScript dependency; works on all API levels 26+.
 */
object BlurProcessor {

    /** How many downscale/upscale iterations — higher = stronger blur */
    private const val ITERATIONS   = 3
    /** Downscale factor per iteration */
    private const val SCALE_FACTOR = 0.25f

    /**
     * Rotate [bitmap] 90° clockwise (to convert landscape sensor output → portrait).
     */
    fun rotateBitmap(bitmap: Bitmap, degrees: Float = 90f): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(degrees)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Produce a portrait-blur composited frame.
     *
     * @param frame      Camera bitmap already in correct orientation (portrait)
     * @param boxInFrame Bounding box of the sharp subject, in [frame] pixel coordinates
     * @return           New composited Bitmap (frame-sized)
     */
    fun process(frame: Bitmap, boxInFrame: RectF?): Bitmap {
        val w = frame.width
        val h = frame.height

        // ── Step 1: strong iterative blur ──────────────────────────────
        var blurred = frame
        repeat(ITERATIONS) {
            val sw = (blurred.width  * SCALE_FACTOR).toInt().coerceAtLeast(2)
            val sh = (blurred.height * SCALE_FACTOR).toInt().coerceAtLeast(2)
            val down = Bitmap.createScaledBitmap(blurred, sw, sh, true)
            blurred  = Bitmap.createScaledBitmap(down, w, h, true)
            down.recycle()
        }

        // ── Step 2: composite ──────────────────────────────────────────
        val output = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        // Draw fully blurred background
        canvas.drawBitmap(blurred, 0f, 0f, null)
        blurred.recycle()

        if (boxInFrame == null) return output

        // ── Step 3: paint sharp subject over blurred bg ────────────────
        // Use a save-layer + radial gradient mask so edges are soft
        val cx = boxInFrame.centerX()
        val cy = boxInFrame.centerY()
        // Radius covers the box with 15% extra padding
        val rx = boxInFrame.width()  * 0.65f
        val ry = boxInFrame.height() * 0.65f
        val r  = max(rx, ry)

        // Save a layer, draw sharp frame, then mask with radial gradient
        val sc = canvas.saveLayer(0f, 0f, w.toFloat(), h.toFloat(), null)

        // Draw the original sharp frame
        canvas.drawBitmap(frame, 0f, 0f, null)

        // Apply DST_IN mask — only keep pixels inside radial gradient
        val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
        }
        val gradient = RadialGradient(
            cx, cy, r,
            intArrayOf(
                Color.argb(255, 0, 0, 0),   // fully opaque inside
                Color.argb(255, 0, 0, 0),   // solid to 60%
                Color.argb(0,   0, 0, 0)    // fully transparent outside
            ),
            floatArrayOf(0f, 0.6f, 1.0f),
            Shader.TileMode.CLAMP
        )
        maskPaint.shader = gradient
        canvas.drawRect(0f, 0f, w.toFloat(), h.toFloat(), maskPaint)
        canvas.restoreToCount(sc)

        return output
    }
}
