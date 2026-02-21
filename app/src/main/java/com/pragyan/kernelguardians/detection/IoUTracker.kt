package com.pragyan.kernelguardians.detection

import android.graphics.RectF
import com.pragyan.kernelguardians.tracking.KalmanFilter

/**
 * SORT-style tracker: each track owns a [KalmanFilter].
 *
 * Improvement over plain IoU matching:
 *   - Kalman predicts each track's position before matching
 *   - IoU is computed between the PREDICTED box and new detections
 *   - Fast-moving objects bridge frame gaps without losing their ID
 *   - Unmatched tracks coast on Kalman predictions for MAX_MISS_FRAMES
 *     then get removed
 *
 * Previously the tracker compared raw last-seen boxes, which broke when
 * an object moved more than its own width between frames.
 */
class IoUTracker {

    companion object {
        /** Minimum IoU between predicted track and new detection to match */
        private const val IOU_THRESHOLD   = 0.20f   // slightly lower since we're comparing predicted boxes
        /** Frames a track lives on prediction before being discarded */
        private const val MAX_MISS_FRAMES = 8
        private const val MAX_TRACKS      = 20
    }

    private inner class Track(
        val id: Int,
        initialBox: RectF
    ) {
        val kalman    = KalmanFilter()
        var missCount = 0

        /** The box we expose — either Kalman-corrected (on hit) or Kalman-predicted (on miss) */
        var box: RectF = RectF(initialBox)

        init { kalman.init(initialBox) }

        /** Step 1 called each frame — advance the state machine, returns predicted box */
        fun predictNext(): RectF {
            box = kalman.predict()
            missCount++
            return box
        }

        /** Step 2 called when a detection matches this track */
        fun update(detectedBox: RectF) {
            box = kalman.update(detectedBox)
            missCount = 0
        }
    }

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /**
     * Update the tracker with a new frame's detections.
     *
     * @param rawBoxes  Unordered list of (boundingBox → (label, confidence))
     * @return          DetectionResult list with stable tracking IDs and Kalman-smoothed boxes
     */
    fun update(
        rawBoxes: List<Pair<RectF, Pair<String, Float>>>
    ): List<DetectionResult> {

        // ── 1. Predict next position for all tracks ──────────────────────────
        val predictedBoxes = tracks.map { it.predictNext() }

        val usedTrackIndices     = mutableSetOf<Int>()
        val usedDetectionIndices = mutableSetOf<Int>()
        val results              = mutableListOf<DetectionResult>()

        // ── 2. Build IoU matrix: predicted-track vs new-detection ─────────────
        val ious = Array(tracks.size) { ti ->
            FloatArray(rawBoxes.size) { di ->
                iou(predictedBoxes[ti], rawBoxes[di].first)
            }
        }

        // ── 3. Greedy matching — highest IoU first ────────────────────────────
        while (true) {
            var bestIoU = IOU_THRESHOLD
            var bestTi  = -1
            var bestDi  = -1

            for (ti in tracks.indices) {
                if (ti in usedTrackIndices) continue
                for (di in rawBoxes.indices) {
                    if (di in usedDetectionIndices) continue
                    if (ious[ti][di] > bestIoU) {
                        bestIoU = ious[ti][di]
                        bestTi  = ti
                        bestDi  = di
                    }
                }
            }

            if (bestTi == -1) break

            val track       = tracks[bestTi]
            val (box, meta) = rawBoxes[bestDi]
            track.update(box)                             // Kalman update with real detection
            usedTrackIndices.add(bestTi)
            usedDetectionIndices.add(bestDi)

            results.add(DetectionResult(track.id, meta.first, meta.second, RectF(track.box)))
        }

        // ── 4. Unmatched detections → new tracks ──────────────────────────────
        rawBoxes.indices.filter { it !in usedDetectionIndices }.forEach { di ->
            val (box, meta) = rawBoxes[di]
            if (tracks.size < MAX_TRACKS) {
                val id = nextId++
                tracks.add(Track(id, RectF(box)))
                results.add(DetectionResult(id, meta.first, meta.second, RectF(box)))
            }
        }

        // ── 5. Remove stale tracks ────────────────────────────────────────────
        tracks.removeAll { it.missCount > MAX_MISS_FRAMES }

        return results
    }

    /** Reset all tracks (e.g. when camera restarts or app resumes). */
    fun reset() {
        tracks.clear()
        nextId = 1
    }

    // ── IoU helper ────────────────────────────────────────────────────────────

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = maxOf(a.left,   b.left)
        val interTop    = maxOf(a.top,    b.top)
        val interRight  = minOf(a.right,  b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interW = (interRight  - interLeft).coerceAtLeast(0f)
        val interH = (interBottom - interTop ).coerceAtLeast(0f)
        val inter  = interW * interH

        if (inter == 0f) return 0f

        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }
}
