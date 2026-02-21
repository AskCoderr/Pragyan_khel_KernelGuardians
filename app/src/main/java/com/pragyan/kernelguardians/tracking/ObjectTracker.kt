package com.pragyan.kernelguardians.tracking

import android.graphics.PointF
import android.graphics.RectF
import com.pragyan.kernelguardians.detection.DetectionResult

/**
 * Manages the "lock onto a subject" lifecycle using [DetectionResult]
 * (model-agnostic, replaces ML Kit DetectedObject).
 *
 *  1. User taps → [onUserTap] picks the closest detected object.
 *  2. [processDetections] runs every frame.
 *     - Locked object ID found → update Kalman, LOCKED.
 *     - Not found → Kalman predicts, PREDICTING.
 *     - Too many misses → SEARCHING.
 *  3. [clearLock] releases the lock.
 */
class ObjectTracker {

    private val kalman         = KalmanFilter()
    private var lockedId: Int? = null

    var currentState: TrackingState = TrackingState.IDLE
        private set
    var currentBox: RectF? = null
        private set
    var currentLabel: String = ""
        private set
    var currentConfidence: Float = 0f
        private set

    /**
     * Called when the user taps at [touchPoint] on screen.
     * Selects the detected object whose mapped box centre is nearest the tap.
     *
     * @param touchPoint  Tap location in view/overlay space
     * @param detections  Latest detections already mapped to view space
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
        kalman.init(box)
        return true
    }

    /**
     * Process a new set of detections mapped to view space.
     * Called on every analyzed frame.
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

        val predicted = kalman.predict()

        val match = detections.firstOrNull { (result, _) -> result.trackingId == lockedId }

        return if (match != null) {
            val (result, box) = match
            val corrected     = kalman.update(box)
            currentBox        = corrected
            currentLabel      = result.label
            currentConfidence = result.confidence
            val prev          = currentState
            currentState      = TrackingState.LOCKED
            prev != TrackingState.LOCKED
        } else {
            currentBox = predicted
            val prev   = currentState
            currentState = if (kalman.predictionFrameCount >= KalmanFilter.MAX_PREDICTION_FRAMES) {
                TrackingState.SEARCHING
            } else {
                TrackingState.PREDICTING
            }
            prev != currentState
        }
    }

    fun clearLock() {
        lockedId     = null
        currentState = TrackingState.IDLE
        currentBox   = null
        currentLabel = ""
        kalman.reset()
    }

    val isLocked: Boolean get() = lockedId != null
}
