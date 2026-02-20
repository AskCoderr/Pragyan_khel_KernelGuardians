package com.pragyan.kernelguardians.rendering

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import com.pragyan.kernelguardians.tracking.TrackingState

/**
 * Transparent overlay that draws the ML-Kit bounding box + tracking state label.
 * Updated by [ObjectTracker] on every processed frame via [update].
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val density = context.resources.displayMetrics.density

    private val lockedColor  = Color.parseColor("#FF00C853") // green
    private val predictColor = Color.parseColor("#FFFFC107") // amber — Kalman predicting
    private val searchColor  = Color.parseColor("#FFFF5722") // red — lost

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style       = Paint.Style.STROKE
        strokeWidth = 3f * density
        color       = lockedColor
    }

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(30, 0, 200, 83)
    }

    private val labelBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(180, 0, 0, 0)
    }

    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color    = Color.WHITE
        textSize = 12f * density
        typeface = android.graphics.Typeface.MONOSPACE
    }

    // State updated from worker thread — use @Volatile + post
    @Volatile var boundingBox: RectF?       = null
    @Volatile var trackingState: TrackingState = TrackingState.IDLE
    @Volatile var objectLabel: String       = ""
    @Volatile var confidence: Float         = 0f

    fun update(box: RectF?, state: TrackingState, label: String = "", conf: Float = 0f) {
        boundingBox   = box
        trackingState = state
        objectLabel   = label
        confidence    = conf
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val box    = boundingBox ?: return
        val state  = trackingState

        val boxColor = when (state) {
            TrackingState.LOCKED    -> lockedColor
            TrackingState.PREDICTING-> predictColor
            TrackingState.SEARCHING -> searchColor
            TrackingState.IDLE      -> return
        }

        boxPaint.color  = boxColor
        fillPaint.color = Color.argb(25, Color.red(boxColor), Color.green(boxColor), Color.blue(boxColor))

        // Filled rect
        canvas.drawRoundRect(box, 8f * density, 8f * density, fillPaint)
        // Border
        canvas.drawRoundRect(box, 8f * density, 8f * density, boxPaint)

        // Label
        val stateLabel = when (state) {
            TrackingState.LOCKED     -> "🔒 LOCKED"
            TrackingState.PREDICTING -> "⚡ PREDICTING"
            TrackingState.SEARCHING  -> "🔍 SEARCHING"
            TrackingState.IDLE       -> ""
        }

        val label = if (objectLabel.isNotEmpty())
            "$stateLabel  $objectLabel ${(confidence * 100).toInt()}%"
        else stateLabel

        val textW    = textPaint.measureText(label) + 16f * density
        val textH    = 20f * density
        val labelTop = (box.top - textH).coerceAtLeast(4f)

        canvas.drawRoundRect(
            box.left, labelTop,
            box.left + textW, labelTop + textH,
            4f, 4f, labelBgPaint
        )
        canvas.drawText(label, box.left + 8f * density, labelTop + textH - 4f * density, textPaint)
    }
}
