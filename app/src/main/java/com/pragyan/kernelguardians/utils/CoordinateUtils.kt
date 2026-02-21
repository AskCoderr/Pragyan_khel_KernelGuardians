package com.pragyan.kernelguardians.utils

import android.graphics.PointF
import android.graphics.RectF
import androidx.camera.core.MeteringPoint
import androidx.camera.core.MeteringPointFactory

/**
 * Converts touch / display coordinates ↔ sensor / image coordinates.
 * Accounts for rotation, mirroring and scale differences between the
 * PreviewView and the ImageAnalysis output.
 */
object CoordinateUtils {

    /**
     * Map a raw touch point on the preview surface to a [MeteringPoint]
     * for CameraX tap-to-focus.
     *
     * @param factory  The [MeteringPointFactory] from [PreviewView.getMeteringPointFactory]
     * @param touchX   Raw touch X on the PreviewView
     * @param touchY   Raw touch Y on the PreviewView
     */
    fun toMeteringPoint(
        factory: MeteringPointFactory,
        touchX: Float,
        touchY: Float
    ): MeteringPoint = factory.createPoint(touchX, touchY)

    /**
     * Map a bounding box from the ML Kit image coordinate space
     * (image width × image height) → preview screen space (view width × view height).
     *
     * ML Kit returns coords relative to the analyzed image; the preview
     * may be a different size / aspect. This function does a simple
     * proportional remap.
     *
     * @param box          The bounding box from the detector
     * @param imageWidth   Width of the image fed to the detector
     * @param imageHeight  Height of the image fed to the detector
     * @param viewWidth    Width of the PreviewView / OverlayView in px
     * @param viewHeight   Height of the PreviewView / OverlayView in px
     * @param isFrontCamera Whether the active lens is front-facing (needs X-flip)
     */
    fun mapBoundingBoxToView(
        box: RectF,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        isFrontCamera: Boolean = false,
        rotDeg: Int = 0
    ): RectF {
        // When the sensor image is rotated 90° or 270°, width and height are swapped
        // relative to what the preview shows — compensate before scaling.
        val effectiveImgW = if (rotDeg == 90 || rotDeg == 270) imageHeight else imageWidth
        val effectiveImgH = if (rotDeg == 90 || rotDeg == 270) imageWidth  else imageHeight

        // CameraX PreviewView defaults to FILL_CENTER (scale-to-fill, centered).
        val scaleX = viewWidth.toFloat()  / effectiveImgW.toFloat()
        val scaleY = viewHeight.toFloat() / effectiveImgH.toFloat()
        val scale  = maxOf(scaleX, scaleY)

        val offsetX = (viewWidth  - effectiveImgW * scale) / 2f
        val offsetY = (viewHeight - effectiveImgH * scale) / 2f

        var left   = box.left   * scale + offsetX
        var right  = box.right  * scale + offsetX
        val top    = box.top    * scale + offsetY
        val bottom = box.bottom * scale + offsetY

        if (isFrontCamera) {
            left  = viewWidth - (box.right * scale + offsetX)
            right = viewWidth - (box.left  * scale + offsetX)
        }

        return RectF(left, top, right, bottom)
    }

    /**
     * Map a single image-space point to view space.
     */
    fun mapPointToView(
        point: PointF,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Int,
        viewHeight: Int
    ): PointF {
        val scaleX = viewWidth.toFloat() / imageWidth.toFloat()
        val scaleY = viewHeight.toFloat() / imageHeight.toFloat()
        return PointF(point.x * scaleX, point.y * scaleY)
    }
}
