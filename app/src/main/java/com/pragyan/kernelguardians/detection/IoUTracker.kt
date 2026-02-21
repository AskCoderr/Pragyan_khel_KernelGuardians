package com.pragyan.kernelguardians.detection

import android.graphics.RectF
import com.pragyan.kernelguardians.tracking.KalmanFilter

/**
 * SORT-style tracker: each track owns a [KalmanFilter].
 *
 * Matching pipeline — two stages per frame:
 *
 *   Stage 1 — IoU (fast, position-based):
 *     Standard Kalman-predicted box vs detection IoU.
 *     Works for normal motion and bridging short occlusions.
 *
 *   Stage 2 — Appearance Re-ID (fallback for lost tracks):
 *     For tracks that failed IoU matching (missCount > 0), compare the
 *     track's stored appearance embedding against unmatched detections of
 *     the SAME LABEL using cosine similarity.
 *     Allows re-acquiring a previously locked object that moved far away
 *     or temporarily left the frame.
 */
class IoUTracker {

    companion object {
        private const val IOU_THRESHOLD   = 0.20f
        /** Cosine similarity threshold for Re-ID re-acquisition */
        private const val SIM_THRESHOLD   = 0.80f
        private const val MAX_MISS_FRAMES = 8
        private const val MAX_TRACKS      = 20
    }

    private inner class Track(val id: Int, initialBox: RectF) {
        val kalman    = KalmanFilter()
        var missCount = 0
        var label     = ""

        var box: RectF = RectF(initialBox)

        /** Last-seen appearance embedding for Re-ID; updated on every hit */
        var embedding: FloatArray? = null

        init { kalman.init(initialBox) }

        fun predictNext(): RectF {
            box = kalman.predict()
            missCount++
            return box
        }

        fun update(detectedBox: RectF, detectedLabel: String, detectedEmbedding: FloatArray?) {
            box       = kalman.update(detectedBox)
            missCount = 0
            label     = detectedLabel
            if (detectedEmbedding != null) embedding = detectedEmbedding
        }
    }

    private val tracks = mutableListOf<Track>()
    private var nextId = 1

    /**
     * Update the tracker with a new frame's detections.
     *
     * @param rawBoxes   (bounding box → (label, confidence)) for each detection
     * @param embeddings Parallel list of appearance embeddings; may be empty or shorter than rawBoxes
     * @return           Stable-ID DetectionResult list with Kalman-smoothed boxes and embeddings
     */
    fun update(
        rawBoxes:   List<Pair<RectF, Pair<String, Float>>>,
        embeddings: List<FloatArray?> = emptyList()
    ): List<DetectionResult> {

        fun embeddingAt(i: Int) = embeddings.getOrNull(i)

        // ── 1. Predict next position for all tracks ──────────────────────────
        val predictedBoxes = tracks.map { it.predictNext() }

        val usedTi  = mutableSetOf<Int>()
        val usedDi  = mutableSetOf<Int>()
        val results = mutableListOf<DetectionResult>()

        // ── 2. IoU matrix ────────────────────────────────────────────────────
        val ious = Array(tracks.size) { ti ->
            FloatArray(rawBoxes.size) { di -> iou(predictedBoxes[ti], rawBoxes[di].first) }
        }

        // ── 3. Stage-1: greedy IoU matching ──────────────────────────────────
        while (true) {
            var best = IOU_THRESHOLD; var bTi = -1; var bDi = -1
            for (ti in tracks.indices) {
                if (ti in usedTi) continue
                for (di in rawBoxes.indices) {
                    if (di in usedDi) continue
                    if (ious[ti][di] > best) { best = ious[ti][di]; bTi = ti; bDi = di }
                }
            }
            if (bTi == -1) break
            val track       = tracks[bTi]
            val (box, meta) = rawBoxes[bDi]
            track.update(box, meta.first, embeddingAt(bDi))
            usedTi.add(bTi); usedDi.add(bDi)
            results.add(DetectionResult(track.id, meta.first, meta.second, RectF(track.box), embeddingAt(bDi)))
        }

        // ── 4. Stage-2: Re-ID appearance matching ────────────────────────────
        // For each unmatched track whose embedding we have, scan unmatched
        // detections of the same label by cosine similarity.
        for (ti in tracks.indices) {
            if (ti in usedTi) continue
            val track    = tracks[ti]
            val trackEmb = track.embedding ?: continue      // no reference yet

            var bestSim = SIM_THRESHOLD; var bestDi = -1
            for (di in rawBoxes.indices) {
                if (di in usedDi) continue
                val (_, meta) = rawBoxes[di]
                if (meta.first != track.label) continue     // same-label only
                val detEmb = embeddingAt(di) ?: continue
                val sim = AppearanceEmbedder.similarity(trackEmb, detEmb)
                if (sim > bestSim) { bestSim = sim; bestDi = di }
            }

            if (bestDi == -1) continue

            // Re-acquisition: update track with the appearance-matched detection
            val (box, meta) = rawBoxes[bestDi]
            track.update(box, meta.first, embeddingAt(bestDi))
            usedTi.add(ti); usedDi.add(bestDi)
            results.add(DetectionResult(track.id, meta.first, meta.second, RectF(track.box), embeddingAt(bestDi)))
        }

        // ── 5. Unmatched detections → new tracks ──────────────────────────────
        rawBoxes.indices.filter { it !in usedDi }.forEach { di ->
            val (box, meta) = rawBoxes[di]
            if (tracks.size < MAX_TRACKS) {
                val id    = nextId++
                val track = Track(id, RectF(box))
                track.label     = meta.first
                track.embedding = embeddingAt(di)
                tracks.add(track)
                results.add(DetectionResult(id, meta.first, meta.second, RectF(box), embeddingAt(di)))
            }
        }

        // ── 6. Remove stale tracks ────────────────────────────────────────────
        tracks.removeAll { it.missCount > MAX_MISS_FRAMES }

        return results
    }

    fun reset() { tracks.clear(); nextId = 1 }

    private fun iou(a: RectF, b: RectF): Float {
        val iL = maxOf(a.left,  b.left);  val iT = maxOf(a.top,    b.top)
        val iR = minOf(a.right, b.right); val iB = minOf(a.bottom, b.bottom)
        val iW = (iR - iL).coerceAtLeast(0f)
        val iH = (iB - iT).coerceAtLeast(0f)
        val inter = iW * iH
        if (inter == 0f) return 0f
        val union = a.width() * a.height() + b.width() * b.height() - inter
        return if (union <= 0f) 0f else inter / union
    }
}
