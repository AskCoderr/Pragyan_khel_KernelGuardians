package com.pragyan.kernelguardians.rendering

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.util.AttributeSet
import android.view.View
import android.view.animation.DecelerateInterpolator

/**
 * Custom view that draws an animated focus ring at the tap location.
 *
 * Behaviour:
 *  1. Ring appears at touch point, scaled large → small (tap burst feel).
 *  2. Briefly pulses at locked size to confirm focus.
 *  3. Fades out after [HOLD_DURATION_MS] ms.
 */
class FocusRingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val RING_SIZE_DP      = 80f
        private const val RING_STROKE_DP    = 2f
        private const val CORNER_RADIUS_DP  = 8f
        private const val ANIM_DURATION_MS  = 400L
        private const val HOLD_DURATION_MS  = 800L
        private const val FADE_DURATION_MS  = 300L
    }

    private val density = context.resources.displayMetrics.density
    private val ringSize   = RING_SIZE_DP   * density
    private val strokeWidth= RING_STROKE_DP * density
    private val cornerR    = CORNER_RADIUS_DP * density

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = this@FocusRingView.strokeWidth
        color       = Color.WHITE
    }

    private var center  = PointF(0f, 0f)
    private var scale   = 1.5f   // animated
    private var alpha   = 0       // 0-255

    private var scaleAnimator: ValueAnimator? = null
    private var fadeAnimator:  ValueAnimator? = null

    /** Call this when the user taps to show the ring at [x], [y]. */
    fun show(x: Float, y: Float) {
        center.set(x, y)
        cancelAnimators()

        // Scale in: 1.5 → 1.0
        scaleAnimator = ValueAnimator.ofFloat(1.5f, 1.0f).apply {
            duration    = ANIM_DURATION_MS
            interpolator= DecelerateInterpolator()
            addUpdateListener {
                scale = it.animatedValue as Float
                paint.alpha = 255
                alpha = 255
                invalidate()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    // Hold then fade
                    postDelayed({ startFade() }, HOLD_DURATION_MS)
                }
            })
            start()
        }
    }

    /** Flash green to confirm successful focus lock */
    fun showLocked(x: Float, y: Float) {
        center.set(x, y)
        cancelAnimators()
        scale = 1.0f
        paint.color = Color.parseColor("#FF00C853") // green
        paint.alpha = 255
        alpha = 255
        invalidate()
        postDelayed({ startFade() }, HOLD_DURATION_MS)
    }

    fun hide() {
        cancelAnimators()
        alpha = 0
        paint.color = Color.WHITE
        invalidate()
    }

    private fun startFade() {
        fadeAnimator = ValueAnimator.ofInt(255, 0).apply {
            duration = FADE_DURATION_MS
            addUpdateListener {
                alpha       = it.animatedValue as Int
                paint.alpha = alpha
                paint.color = Color.WHITE
                invalidate()
            }
            start()
        }
    }

    private fun cancelAnimators() {
        scaleAnimator?.cancel()
        fadeAnimator?.cancel()
        removeCallbacks(null)
        paint.color = Color.WHITE
    }

    override fun onDraw(canvas: Canvas) {
        if (alpha == 0) return
        val half = (ringSize * scale) / 2f
        canvas.drawRoundRect(
            center.x - half, center.y - half,
            center.x + half, center.y + half,
            cornerR, cornerR,
            paint
        )
        // Corner ticks (classic camera focus aesthetic)
        val tickLen = half * 0.3f
        paint.strokeWidth = strokeWidth * 2
        // Top-left
        canvas.drawLine(center.x - half, center.y - half, center.x - half + tickLen, center.y - half, paint)
        canvas.drawLine(center.x - half, center.y - half, center.x - half, center.y - half + tickLen, paint)
        // Top-right
        canvas.drawLine(center.x + half, center.y - half, center.x + half - tickLen, center.y - half, paint)
        canvas.drawLine(center.x + half, center.y - half, center.x + half, center.y - half + tickLen, paint)
        // Bottom-left
        canvas.drawLine(center.x - half, center.y + half, center.x - half + tickLen, center.y + half, paint)
        canvas.drawLine(center.x - half, center.y + half, center.x - half, center.y + half - tickLen, paint)
        // Bottom-right
        canvas.drawLine(center.x + half, center.y + half, center.x + half - tickLen, center.y + half, paint)
        canvas.drawLine(center.x + half, center.y + half, center.x + half, center.y + half - tickLen, paint)
        paint.strokeWidth = strokeWidth
    }
}
