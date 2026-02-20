package com.pragyan.kernelguardians.tracking

import android.graphics.RectF

/**
 * Constant-velocity Kalman Filter for a 2-D bounding box.
 *
 * State vector: [cx, cy, w, h, vcx, vcy, vw, vh]
 *   cx, cy  = centre x/y
 *   w, h    = width / height
 *   vcx,vcy = velocity of centre
 *   vw, vh  = velocity of width/height
 *
 * Observation: [cx, cy, w, h]
 */
class KalmanFilter {

    companion object {
        private const val STATE_DIM = 8
        private const val OBS_DIM   = 4
        /** Process noise — how much we trust the motion model */
        private const val Q = 1e-2f
        /** Measurement noise — how much we trust the detector */
        private const val R = 1e-1f
        /** Max frames without detection before declaring target SEARCHING */
        const val MAX_PREDICTION_FRAMES = 15
    }

    // State estimate [cx, cy, w, h, vcx, vcy, vw, vh]
    private val x = FloatArray(STATE_DIM)
    // Estimate covariance (diagonal for speed)
    private val P = FloatArray(STATE_DIM) { 1f }

    private var initialized    = false
    var predictionFrameCount   = 0
        private set

    /** Initialise the filter from the first observed bounding box. */
    fun init(box: RectF) {
        x[0] = box.centerX(); x[1] = box.centerY()
        x[2] = box.width();    x[3] = box.height()
        x[4] = 0f; x[5] = 0f; x[6] = 0f; x[7] = 0f
        P.fill(1f)
        initialized           = true
        predictionFrameCount  = 0
    }

    /** Reset the filter entirely. */
    fun reset() { initialized = false; predictionFrameCount = 0; P.fill(1f) }

    /**
     * Predict the next state (call once per frame regardless of detection).
     * @return predicted bounding box
     */
    fun predict(): RectF {
        if (!initialized) return RectF()
        // x_pred = F * x   (constant-velocity model: pos += vel)
        x[0] += x[4]; x[1] += x[5]
        x[2] += x[6]; x[3] += x[7]
        // P_pred = P + Q  (diagonal approximation)
        for (i in 0 until STATE_DIM) P[i] += Q
        predictionFrameCount++
        return stateToRect()
    }

    /**
     * Update the filter with a new detector measurement.
     * @param observed  The bounding box returned by the detector.
     * @return corrected bounding box (fused estimate)
     */
    fun update(observed: RectF): RectF {
        if (!initialized) { init(observed); return observed }
        predictionFrameCount = 0

        val z = floatArrayOf(
            observed.centerX(), observed.centerY(),
            observed.width(),   observed.height()
        )

        // Kalman gain  K = P_obs / (P_obs + R)   for obs dimensions 0-3
        for (i in 0 until OBS_DIM) {
            val k = P[i] / (P[i] + R)
            val residual = z[i] - x[i]
            // Velocity update (innovation dampened)
            x[i + 4] = (x[i + 4] * 0.8f) + (k * residual * 0.2f)
            // Position update
            x[i] += k * residual
            // Covariance update
            P[i] = (1f - k) * P[i]
        }
        return stateToRect()
    }

    val isInitialized: Boolean get() = initialized

    private fun stateToRect(): RectF {
        val hw = x[2] / 2f; val hh = x[3] / 2f
        return RectF(x[0] - hw, x[1] - hh, x[0] + hw, x[1] + hh)
    }
}
