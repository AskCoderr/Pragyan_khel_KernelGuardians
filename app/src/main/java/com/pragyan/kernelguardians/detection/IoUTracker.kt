package com.pragyan.kernelguardians.detection

import android.graphics.RectF

/**
 * Assigns stable tracking IDs to detections across frames using
 * Intersection-over-Union (IoU) matching — the same principle used in
 * SORT (Simple Online and Realtime Tracking).
 *
 * Without ML Kit's built-in tracking, we need this to keep a consistent
 * "identity" for each object so ObjectTracker can lock onto a subject.
 *
 * Algorithm per frame:
 *  1. For each existing track, compute IoU with every new detection.
 *  2. Greedily match the highest-IoU pair (if IoU ≥ threshold).
 *  3. Matched detections inherit the track's ID.
 *  4. Unmatched detections get a new unique ID.
 *  5. Tracks not matched for [MAX_MISS_FRAMES] frames are removed.
 */
class IoUTracker {

    companion object {
        private const val IOU_THRESHOLD      = 0.25f  // minimum overlap to consider "same object"
        private const val MAX_MISS_FRAMES    = 8       // frames before a lost track is removed
        private const val MAX_TRACKS        = 20
    }

    private data class Track(
        val id: Int,
        var box: RectF,
        var missCount: Int = 0
    )

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /**
     * Match [rawBoxes] (with labels/scores) to existing tracks.
     * Returns the same list with stable tracking IDs assigned.
     */
    fun update(
        rawBoxes: List<Pair<RectF, Pair<String, Float>>>   // box → (label, score)
    ): List<DetectionResult> {

        // Predict: mark all tracks as potentially missed
        tracks.forEach { it.missCount++ }

        val usedTrackIndices     = mutableSetOf<Int>()
        val usedDetectionIndices = mutableSetOf<Int>()
        val results              = mutableListOf<DetectionResult>()

        // --- Greedy IoU matching ---
        // Build IoU matrix
        val ious = Array(tracks.size) { ti ->
            FloatArray(rawBoxes.size) { di ->
                iou(tracks[ti].box, rawBoxes[di].first)
            }
        }

        // Repeatedly pick the highest IoU pair
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

            if (bestTi == -1) break  // no more good matches

            val track = tracks[bestTi]
            val (box, meta) = rawBoxes[bestDi]
            track.box       = box
            track.missCount = 0
            usedTrackIndices.add(bestTi)
            usedDetectionIndices.add(bestDi)

            results.add(DetectionResult(track.id, meta.first, meta.second, RectF(box)))
        }

        // --- Unmatched detections → new tracks ---
        rawBoxes.indices.filter { it !in usedDetectionIndices }.forEach { di ->
            val (box, meta) = rawBoxes[di]
            if (tracks.size < MAX_TRACKS) {
                val id = nextId++
                tracks.add(Track(id, RectF(box)))
                results.add(DetectionResult(id, meta.first, meta.second, RectF(box)))
            }
        }

        // --- Remove stale tracks ---
        tracks.removeAll { it.missCount > MAX_MISS_FRAMES }

        return results
    }

    /** Reset all tracks (e.g. when camera restarts). */
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

        val areaA = a.width() * a.height()
        val areaB = b.width() * b.height()
        val union = areaA + areaB - inter

        return if (union <= 0f) 0f else inter / union
    }
}
