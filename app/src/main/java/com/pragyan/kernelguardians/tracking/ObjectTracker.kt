package com.pragyan.kernelguardians.tracking

import android.graphics.PointF
import android.graphics.RectF
import com.google.mlkit.vision.objects.DetectedObject
import com.pragyan.kernelguardians.utils.CoordinateUtils

/**
 * Manages the "lock onto a subject" lifecycle:
 *
 *  1. User taps → [onUserTap] picks the closest detected object.
 *  2. [processDetections] runs every frame in STREAM_MODE.
 *     - If the locked object ID reappears, update the Kalman Filter.
 *     - If it disappears, switch to PREDICTING (Kalman-only).
 *     - After [KalmanFilter.MAX_PREDICTION_FRAMES], switch to SEARCHING.
 *  3. [clearLock] releases the lock.
 */
class ObjectTracker {

    private val kalman        = KalmanFilter()
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
     * Selects the detected object whose box centre is closest to the tap.
     *
     * @param touchPoint     Tap location in view/overlay space
     * @param detections     Latest list of [DetectedObject] already mapped to view space
     * @param imageWidth     Original image width (for mapping if needed)
     * @param imageHeight    Original image height
     * @param viewWidth      View width
     * @param viewHeight     View height
     * @return true if an object was locked
     */
    fun onUserTap(
        touchPoint: PointF,
        detections: List<Pair<DetectedObject, RectF>>,
        imageWidth: Int, imageHeight: Int,
        viewWidth: Int,  viewHeight: Int
    ): Boolean {
        if (detections.isEmpty()) return false

        // Find nearest object center to tap
        val nearest = detections.minByOrNull { (_, box) ->
            val dx = box.centerX() - touchPoint.x
            val dy = box.centerY() - touchPoint.y
            dx * dx + dy * dy
        } ?: return false

        val (obj, box) = nearest
        lockedId          = obj.trackingId
        currentLabel      = obj.labels.firstOrNull()?.text ?: "Object"
        currentConfidence = obj.labels.firstOrNull()?.confidence ?: 0f
        currentState      = TrackingState.LOCKED
        currentBox        = box
        kalman.init(box)
        return true
    }

    /**
     * Process a new set of [DetectedObject]s mapped to view space.
     * Should be called on every analyzed frame.
     *
     * @return true if state changed (caller should update UI)
     */
    fun processDetections(
        detections: List<Pair<DetectedObject, RectF>>
    ): Boolean {
        if (lockedId == null) {
            currentState = TrackingState.IDLE
            return false
        }

        // Always predict first (advances Kalman by 1 frame)
        val predicted = kalman.predict()

        // Try to find the locked object
        val match = detections.firstOrNull { (obj, _) -> obj.trackingId == lockedId }

        return if (match != null) {
            val (obj, box)  = match
            val corrected   = kalman.update(box)
            currentBox      = corrected
            currentLabel    = obj.labels.firstOrNull()?.text ?: currentLabel
            currentConfidence = obj.labels.firstOrNull()?.confidence ?: currentConfidence
            val prev = currentState
            currentState    = TrackingState.LOCKED
            prev != TrackingState.LOCKED
        } else {
            // Object not found in this frame
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

    /** Release the current lock and reset. */
    fun clearLock() {
        lockedId     = null
        currentState = TrackingState.IDLE
        currentBox   = null
        currentLabel = ""
        kalman.reset()
    }

    val isLocked: Boolean get() = lockedId != null
}
