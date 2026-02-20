package com.pragyan.kernelguardians.rendering

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import android.view.animation.LinearInterpolator
import com.pragyan.kernelguardians.tracking.TrackingState

/**
 * Transparent overlay that draws a professional corner-bracket tracking indicator.
 *
 * Visual style (inspired by DSLR viewfinders / drone cameras):
 *  - 4 L-shaped corner brackets (no full border)
 *  - Very subtle semi-transparent fill
 *  - Pulsing bracket opacity when LOCKED
 *  - Thin dashed perimeter when PREDICTING
 *  - Red brackets when SEARCHING
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val density = context.resources.displayMetrics.density

    // ── Colors ────────────────────────────────────────────────────────────
    private val lockedColor  = Color.parseColor("#FF00E5FF")  // cyan
    private val predictColor = Color.parseColor("#FFFFC107")  // amber
    private val searchColor  = Color.parseColor("#FFFF5252")  // red

    // ── Paints ────────────────────────────────────────────────────────────
    private val bracketPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 2.5f * density
        strokeCap   = Paint.Cap.SQUARE
        color       = lockedColor
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(18, 0, 200, 255)
    }

    private val dashedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 1f * density
        color       = predictColor
        pathEffect  = DashPathEffect(floatArrayOf(6f * density, 4f * density), 0f)
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(160, 0, 0, 0)
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.WHITE
        textSize = 10.5f * density
        typeface = android.graphics.Typeface.create("monospace", android.graphics.Typeface.BOLD)
    }

    private val confidencePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.parseColor("#FF00E5FF")
        textSize = 10f * density
        typeface = android.graphics.Typeface.create("monospace", android.graphics.Typeface.NORMAL)
    }

    // ── State ─────────────────────────────────────────────────────────────
    @Volatile var boundingBox: RectF?          = null
    @Volatile var trackingState: TrackingState = TrackingState.IDLE
    @Volatile var objectLabel: String          = ""
    @Volatile var confidence: Float            = 0f

    // ── Pulse animation ───────────────────────────────────────────────────
    private var pulseAlpha = 255
    private val pulseAnimator = ValueAnimator.ofInt(255, 140).apply {
        duration      = 800
        repeatCount   = ValueAnimator.INFINITE
        repeatMode    = ValueAnimator.REVERSE
        interpolator  = LinearInterpolator()
        addUpdateListener {
            pulseAlpha = it.animatedValue as Int
            if (trackingState == TrackingState.LOCKED) postInvalidate()
        }
        start()
    }

    fun update(box: RectF?, state: TrackingState, label: String = "", conf: Float = 0f) {
        boundingBox   = box
        trackingState = state
        objectLabel   = label
        confidence    = conf
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val box   = boundingBox ?: return
        val state = trackingState
        if (state == TrackingState.IDLE) return

        val color = when (state) {
            TrackingState.LOCKED     -> lockedColor
            TrackingState.PREDICTING -> predictColor
            TrackingState.SEARCHING  -> searchColor
            TrackingState.IDLE       -> return
        }

        // ── Fill ──────────────────────────────────────────────────────────
        fillPaint.color = Color.argb(18, Color.red(color), Color.green(color), Color.blue(color))
        canvas.drawRoundRect(box, 4f * density, 4f * density, fillPaint)

        // ── Dashed perimeter when predicting ──────────────────────────────
        if (state == TrackingState.PREDICTING) {
            dashedPaint.color = color
            canvas.drawRoundRect(box, 4f * density, 4f * density, dashedPaint)
        }

        // ── Corner brackets ───────────────────────────────────────────────
        val alpha = if (state == TrackingState.LOCKED) pulseAlpha else 255
        bracketPaint.color = Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color))

        // Bracket arm length = 22% of the shorter box dimension
        val arm = (minOf(box.width(), box.height()) * 0.22f).coerceIn(12f * density, 28f * density)

        drawCornerBracket(canvas, box.left,  box.top,    arm,  1f,  1f)   // top-left
        drawCornerBracket(canvas, box.right, box.top,    arm, -1f,  1f)   // top-right
        drawCornerBracket(canvas, box.left,  box.bottom, arm,  1f, -1f)   // bottom-left
        drawCornerBracket(canvas, box.right, box.bottom, arm, -1f, -1f)   // bottom-right

        // Small dot at each corner centre
        val dotPaint = Paint(bracketPaint).apply { style = Paint.Style.FILL; strokeWidth = 0f }
        canvas.drawCircle(box.left,  box.top,    2.5f * density, dotPaint)
        canvas.drawCircle(box.right, box.top,    2.5f * density, dotPaint)
        canvas.drawCircle(box.left,  box.bottom, 2.5f * density, dotPaint)
        canvas.drawCircle(box.right, box.bottom, 2.5f * density, dotPaint)

        // ── Label ─────────────────────────────────────────────────────────
        val stateStr = when (state) {
            TrackingState.LOCKED     -> "LOCKED"
            TrackingState.PREDICTING -> "PREDICT"
            TrackingState.SEARCHING  -> "SEARCHING"
            TrackingState.IDLE       -> ""
        }
        val confStr  = if (confidence > 0f) " ${(confidence * 100).toInt()}%" else ""
        val fullLabel = if (objectLabel.isNotEmpty()) "$stateStr  $objectLabel$confStr" else stateStr

        labelPaint.color = color
        val textW  = labelPaint.measureText(fullLabel) + 12f * density
        val textH  = 18f * density
        val lTop   = (box.top - textH - 2f * density).coerceAtLeast(2f)

        canvas.drawRoundRect(box.left, lTop, box.left + textW, lTop + textH, 3f, 3f, labelBgPaint)
        canvas.drawText(fullLabel, box.left + 6f * density, lTop + textH - 4f * density, labelPaint)
    }

    /**
     * Draw an L-shaped bracket at corner ([cx],[cy]).
     * [sx]/[sy] control which direction the arms extend (±1).
     */
    private fun drawCornerBracket(
        canvas: Canvas,
        cx: Float, cy: Float,
        arm: Float,
        sx: Float, sy: Float
    ) {
        // Horizontal arm
        canvas.drawLine(cx, cy, cx + arm * sx, cy, bracketPaint)
        // Vertical arm
        canvas.drawLine(cx, cy, cx, cy + arm * sy, bracketPaint)
    }

    override fun onDetachedFromWindow() {
        super.onDetachedFromWindow()
        pulseAnimator.cancel()
    }
}
