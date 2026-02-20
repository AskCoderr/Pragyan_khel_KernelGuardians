package com.pragyan.kernelguardians.rendering

import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * OpenGL ES 2.0 renderer that composites camera frame + subject mask
 * using a multi-pass GPU Gaussian blur:
 *
 *   - Subject pixels (white in mask) → rendered sharp.
 *   - Background pixels (black in mask) → strong Gaussian blur applied.
 *
 * The blur uses a two-pass separable approach (horizontal then vertical)
 * repeated [BLUR_ITERATIONS] times for a convincing "portrait mode" effect.
 *
 * If no mask is available yet, renders the camera frame sharp (no black screen).
 *
 * Usage: set [cameraBitmap] and [maskBitmap] before [GLSurfaceView.requestRender].
 */
class BackgroundBlurRenderer : GLSurfaceView.Renderer {

    companion object {
        private const val BLUR_ITERATIONS = 3
    }

    // ── Shader sources ──────────────────────────────────────────────────────

    private val VERTEX_SHADER = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        varying   vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord   = aTexCoord;
        }
    """.trimIndent()

    /** Passthrough shader — just renders a texture as-is. */
    private val PASSTHROUGH_FRAGMENT_SHADER = """
        precision mediump float;
        uniform sampler2D uTexture;
        varying vec2 vTexCoord;
        void main() {
            gl_FragColor = texture2D(uTexture, vTexCoord);
        }
    """.trimIndent()

    /**
     * Blur pass shader — applies a 13-tap 1D Gaussian along [uDirection].
     * uDirection = (1/w, 0) for horizontal, (0, 1/h) for vertical.
     */
    private val BLUR_FRAGMENT_SHADER = """
        precision mediump float;
        uniform sampler2D uTexture;
        uniform vec2 uDirection;
        uniform float uBlurRadius;
        varying vec2 vTexCoord;

        void main() {
            // 13-tap Gaussian weights
            float weights[7];
            weights[0] = 0.1964825501511404;
            weights[1] = 0.2969069646728344;
            weights[2] = 0.2190360083034351;
            weights[3] = 0.0997356038587480;
            weights[4] = 0.0279690609885120;
            weights[5] = 0.0048366093498171;
            weights[6] = 0.0005149688295122;

            vec4 color = texture2D(uTexture, vTexCoord) * weights[0];
            for (int i = 1; i < 7; i++) {
                vec2 offset = uDirection * float(i) * uBlurRadius;
                color += texture2D(uTexture, vTexCoord + offset) * weights[i];
                color += texture2D(uTexture, vTexCoord - offset) * weights[i];
            }
            gl_FragColor = color;
        }
    """.trimIndent()

    /**
     * Compositing shader — blends sharp camera with blurred result using the mask.
     * Uses smoothstep for soft subject-boundary transitions.
     */
    private val COMPOSITE_FRAGMENT_SHADER = """
        precision mediump float;
        uniform sampler2D uCamera;
        uniform sampler2D uBlurred;
        uniform sampler2D uMask;
        varying vec2 vTexCoord;

        void main() {
            float maskVal = texture2D(uMask, vTexCoord).r;
            vec4 sharp    = texture2D(uCamera, vTexCoord);
            vec4 blurred  = texture2D(uBlurred, vTexCoord);
            float alpha   = smoothstep(0.2, 0.8, maskVal);
            gl_FragColor  = mix(blurred, sharp, alpha);
        }
    """.trimIndent()

    // ── Geometry (full-screen quad) ─────────────────────────────────────────

    private val QUAD_VERTICES = floatArrayOf(
        -1f,  1f,   0f, 0f,   // top-left
        -1f, -1f,   0f, 1f,   // bottom-left
         1f,  1f,   1f, 0f,   // top-right
         1f, -1f,   1f, 1f    // bottom-right
    )

    private lateinit var quadBuffer: FloatBuffer

    // ── GL handles ──────────────────────────────────────────────────────────
    private var passthroughProgram = 0
    private var blurProgram        = 0
    private var compositeProgram   = 0
    private var texCamera          = 0
    private var texMask            = 0

    // Ping-pong FBOs for multi-pass blur
    private val fboIds  = IntArray(2)
    private val fboTexs = IntArray(2)
    private var fboW    = 0
    private var fboH    = 0

    // ── Public inputs (set from main thread) ───────────────────────────────
    @Volatile var cameraBitmap: Bitmap? = null
    @Volatile var maskBitmap:   Bitmap? = null
    @Volatile var blurRadius: Float = 1.5f

    private var viewW = 1; private var viewH = 1

    // ── GLSurfaceView.Renderer ─────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        quadBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(QUAD_VERTICES); position(0) }

        passthroughProgram = buildProgram(VERTEX_SHADER, PASSTHROUGH_FRAGMENT_SHADER)
        blurProgram        = buildProgram(VERTEX_SHADER, BLUR_FRAGMENT_SHADER)
        compositeProgram   = buildProgram(VERTEX_SHADER, COMPOSITE_FRAGMENT_SHADER)
        texCamera = genTexture()
        texMask   = genTexture()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width; viewH = height
        setupFBOs(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        val cam = cameraBitmap ?: return

        // Upload camera texture
        uploadTexture(GLES20.GL_TEXTURE0, texCamera, cam)

        val mask = maskBitmap
        if (mask == null) {
            // No mask yet — render camera sharp (avoid black screen)
            GLES20.glUseProgram(passthroughProgram)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texCamera)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(passthroughProgram, "uTexture"), 0)
            drawQuad(passthroughProgram)
            return
        }

        // Upload mask texture
        uploadTexture(GLES20.GL_TEXTURE1, texMask, mask)

        // ── Multi-pass blur ─────────────────────────────────────────────
        var readTex = texCamera
        val texelW = 1f / viewW
        val texelH = 1f / viewH

        for (i in 0 until BLUR_ITERATIONS) {
            // Horizontal pass → fbo[0]
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboIds[0])
            GLES20.glViewport(0, 0, fboW, fboH)
            GLES20.glUseProgram(blurProgram)

            GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, readTex)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(blurProgram, "uTexture"), 2)
            GLES20.glUniform2f(GLES20.glGetUniformLocation(blurProgram, "uDirection"), texelW, 0f)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(blurProgram, "uBlurRadius"), blurRadius)
            drawQuad(blurProgram)

            // Vertical pass → fbo[1]
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboIds[1])
            GLES20.glViewport(0, 0, fboW, fboH)

            GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexs[0])
            GLES20.glUniform1i(GLES20.glGetUniformLocation(blurProgram, "uTexture"), 2)
            GLES20.glUniform2f(GLES20.glGetUniformLocation(blurProgram, "uDirection"), 0f, texelH)
            GLES20.glUniform1f(GLES20.glGetUniformLocation(blurProgram, "uBlurRadius"), blurRadius)
            drawQuad(blurProgram)

            readTex = fboTexs[1]
        }

        // ── Composite pass (to screen) ─────────────────────────────────
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, viewW, viewH)
        GLES20.glUseProgram(compositeProgram)

        // Camera (sharp)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texCamera)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(compositeProgram, "uCamera"), 0)

        // Blurred
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexs[1])
        GLES20.glUniform1i(GLES20.glGetUniformLocation(compositeProgram, "uBlurred"), 2)

        // Mask
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texMask)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(compositeProgram, "uMask"), 1)

        drawQuad(compositeProgram)
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private fun drawQuad(program: Int) {
        val stride = 4 * 4
        val aPos   = GLES20.glGetAttribLocation(program, "aPosition")
        val aTex   = GLES20.glGetAttribLocation(program, "aTexCoord")

        quadBuffer.position(0)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, stride, quadBuffer)
        GLES20.glEnableVertexAttribArray(aPos)

        quadBuffer.position(2)
        GLES20.glVertexAttribPointer(aTex, 2, GLES20.GL_FLOAT, false, stride, quadBuffer)
        GLES20.glEnableVertexAttribArray(aTex)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aPos)
        GLES20.glDisableVertexAttribArray(aTex)
    }

    private fun setupFBOs(width: Int, height: Int) {
        fboW = width; fboH = height

        GLES20.glDeleteFramebuffers(2, fboIds, 0)
        GLES20.glDeleteTextures(2, fboTexs, 0)

        GLES20.glGenFramebuffers(2, fboIds, 0)
        GLES20.glGenTextures(2, fboTexs, 0)

        for (i in 0..1) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexs[i])
            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                width, height, 0, GLES20.GL_RGBA,
                GLES20.GL_UNSIGNED_BYTE, null
            )
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboIds[i])
            GLES20.glFramebufferTexture2D(
                GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, fboTexs[i], 0
            )
        }
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
    }

    private fun genTexture(): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, ids[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return ids[0]
    }

    private fun uploadTexture(unit: Int, texId: Int, bitmap: Bitmap) {
        GLES20.glActiveTexture(unit)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, texId)
        GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0)
    }

    private fun buildProgram(vertSrc: String, fragSrc: String): Int {
        fun compileShader(type: Int, src: String): Int {
            val sh = GLES20.glCreateShader(type)
            GLES20.glShaderSource(sh, src)
            GLES20.glCompileShader(sh)
            return sh
        }
        val vert = compileShader(GLES20.GL_VERTEX_SHADER,   vertSrc)
        val frag = compileShader(GLES20.GL_FRAGMENT_SHADER, fragSrc)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vert)
        GLES20.glAttachShader(prog, frag)
        GLES20.glLinkProgram(prog)
        return prog
    }
}
