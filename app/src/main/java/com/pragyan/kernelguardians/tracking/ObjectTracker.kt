package com.pragyan.kernelguardians.tracking

import android.graphics.PointF
import android.graphics.RectF
import com.pragyan.kernelguardians.detection.AppearanceEmbedder
import com.pragyan.kernelguardians.detection.DetectionResult

/**
 * Manages the "lock onto a subject" lifecycle.
 *
 * Re-ID integration:
 *   On tap-to-lock: stores the detection's appearance embedding as a reference.
 *   When state is SEARCHING (locked ID gone > MAX_PREDICTION_FRAMES):
 *     Scans all current detections of the same label by cosine similarity.
 *     If similarity > RE_ID_THRESHOLD, re-locks onto that detection ID.
 *     This allows recovering the same object even if it left and re-entered the frame.
 *
 * States:
 *   IDLE       → no object selected
 *   LOCKED     → locked object found in this frame's detections
 *   PREDICTING → locked ID missing this frame; IoUTracker's Kalman is coasting
 *   SEARCHING  → ID gone > MAX_PREDICTION_FRAMES; Re-ID scanning active
 */
class ObjectTracker {

    companion object {
        private const val MAX_PREDICTION_FRAMES = KalmanFilter.MAX_PREDICTION_FRAMES
        /** Cosine similarity threshold for Re-ID re-lock */
        private const val RE_ID_THRESHOLD = 0.78f
    }

    private var lockedId:            Int?         = null
    private var missedFrames:        Int          = 0
    private var referenceEmbedding:  FloatArray?  = null
    private var referenceLabel:      String       = ""

    var currentState:      TrackingState = TrackingState.IDLE
        private set
    var currentBox:        RectF?        = null
        private set
    var currentLabel:      String        = ""
        private set
    var currentConfidence: Float         = 0f
        private set

    /**
     * Called when the user taps at [touchPoint] on screen.
     * Selects the nearest detection and stores its embedding as the Re-ID reference.
     */
    fun onUserTap(
        touchPoint: PointF,
        detections: List<Pair<DetectionResult, RectF>>,
        imageWidth: Int, imageHeight: Int,
        viewWidth: Int,  viewHeight: Int
    ): Boolean {
        if (detections.isEmpty()) return false

        val nearest = detections.minByOrNull { (_, box) ->
            val dx = box.centerX() - touchPoint.x
            val dy = box.centerY() - touchPoint.y
            dx * dx + dy * dy
        } ?: return false

        val (result, box) = nearest
        lockedId           = result.trackingId
        currentLabel       = result.label
        currentConfidence  = result.confidence
        currentState       = TrackingState.LOCKED
        currentBox         = box
        missedFrames       = 0

        // Capture Re-ID reference
        referenceEmbedding = result.embedding
        referenceLabel     = result.label

        return true
    }

    /**
     * Process a new set of detections.
     * When SEARCHING, attempts Re-ID across same-label detections using stored embedding.
     */
    fun processDetections(
        detections: List<Pair<DetectionResult, RectF>>
    ): Boolean {
        if (lockedId == null) {
            currentState = TrackingState.IDLE
            return false
        }

        val match = detections.firstOrNull { (result, _) -> result.trackingId == lockedId }

        return if (match != null) {
            val (result, box) = match
            currentBox        = box
            currentLabel      = result.label
            currentConfidence = result.confidence
            missedFrames      = 0
            // Update reference embedding with fresh observation
            if (result.embedding != null) referenceEmbedding = result.embedding
            val prev = currentState
            currentState = TrackingState.LOCKED
            prev != TrackingState.LOCKED

        } else {
            missedFrames++
            val prev = currentState
            currentState = if (missedFrames >= MAX_PREDICTION_FRAMES) {
                // ── Re-ID scan ──────────────────────────────────────────────
                val reIdResult = tryReId(detections)
                if (reIdResult != null) {
                    val (result, box) = reIdResult
                    lockedId          = result.trackingId
                    currentBox        = box
                    currentLabel      = result.label
                    currentConfidence = result.confidence
                    missedFrames      = 0
                    if (result.embedding != null) referenceEmbedding = result.embedding
                    TrackingState.LOCKED
                } else {
                    TrackingState.SEARCHING
                }
            } else {
                TrackingState.PREDICTING
            }
            prev != currentState
        }
    }

    /**
     * Scans [detections] for the best same-label candidate by cosine similarity.
     * Returns null if no candidate exceeds [RE_ID_THRESHOLD].
     */
    private fun tryReId(
        detections: List<Pair<DetectionResult, RectF>>
    ): Pair<DetectionResult, RectF>? {
        val ref = referenceEmbedding ?: return null
        var bestSim  = RE_ID_THRESHOLD
        var bestPair: Pair<DetectionResult, RectF>? = null

        for ((result, box) in detections) {
            if (result.label != referenceLabel) continue
            val emb = result.embedding ?: continue
            val sim = AppearanceEmbedder.similarity(ref, emb)
            if (sim > bestSim) { bestSim = sim; bestPair = Pair(result, box) }
        }
        return bestPair
    }

    fun clearLock() {
        lockedId           = null
        currentState       = TrackingState.IDLE
        currentBox         = null
        currentLabel       = ""
        missedFrames       = 0
        referenceEmbedding = null
        referenceLabel     = ""
    }

    val isLocked: Boolean get() = lockedId != null
}
