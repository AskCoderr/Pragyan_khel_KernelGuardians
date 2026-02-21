package com.pragyan.kernelguardians.tracking

import android.graphics.PointF
import android.graphics.RectF
import com.pragyan.kernelguardians.detection.DetectionResult

/**
 * Manages the "lock onto a subject" lifecycle.
 *
 * Since IoUTracker now runs a per-track Kalman filter (SORT-style),
 * ObjectTracker no longer needs its own KalmanFilter — the smoothed box
 * is already embedded in [DetectionResult.boundingBox].
 *
 * States:
 *   IDLE       → no object selected
 *   LOCKED     → locked object found in this frame's detections
 *   PREDICTING → locked ID missing this frame; IoUTracker's Kalman is
 *                coasting — but the ID may still be re-matched next frame
 *   SEARCHING  → ID has been gone > [MAX_PREDICTION_FRAMES] frames (stale)
 */
class ObjectTracker {

    companion object {
        /** Frames to show PREDICTING before giving up and showing SEARCHING */
        private const val MAX_PREDICTION_FRAMES = KalmanFilter.MAX_PREDICTION_FRAMES
    }

    private var lockedId:        Int?  = null
    private var missedFrames:    Int   = 0

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
     * Selects the detected object whose mapped box centre is nearest the tap.
     *
     * @param touchPoint  Tap location in view/overlay space
     * @param detections  Latest detections mapped to view space
     * @return true if an object was locked
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
        lockedId          = result.trackingId
        currentLabel      = result.label
        currentConfidence = result.confidence
        currentState      = TrackingState.LOCKED
        currentBox        = box
        missedFrames      = 0
        return true
    }

    /**
     * Process a new set of detections (already mapped to view space).
     * Called on every analysed frame.
     *
     * @return true if state changed (caller should update UI)
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
            currentBox        = box          // Kalman-smoothed box from IoUTracker
            currentLabel      = result.label
            currentConfidence = result.confidence
            missedFrames      = 0
            val prev          = currentState
            currentState      = TrackingState.LOCKED
            prev != TrackingState.LOCKED
        } else {
            // Locked ID not in this frame — IoUTracker is coasting on Kalman prediction
            // but may re-match next frame if the object reappears nearby
            missedFrames++
            val prev = currentState
            currentState = if (missedFrames >= MAX_PREDICTION_FRAMES) {
                TrackingState.SEARCHING
            } else {
                TrackingState.PREDICTING
            }
            // Keep showing the last known box while predicting
            prev != currentState
        }
    }

    fun clearLock() {
        lockedId      = null
        currentState  = TrackingState.IDLE
        currentBox    = null
        currentLabel  = ""
        missedFrames  = 0
    }

    val isLocked: Boolean get() = lockedId != null
}
