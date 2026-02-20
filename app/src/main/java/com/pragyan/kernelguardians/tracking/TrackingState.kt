package com.pragyan.kernelguardians.tracking

/**
 * Tracking state machine used by [ObjectTracker] and rendered by [OverlayView].
 */
enum class TrackingState {
    /** No subject selected yet. */
    IDLE,
    /** ML Kit confirmed the target object in this frame. */
    LOCKED,
    /** ML Kit didn't see the target — Kalman Filter is predicting position. */
    PREDICTING,
    /** Kalman prediction horizon exceeded — target is truly lost. */
    SEARCHING
}
