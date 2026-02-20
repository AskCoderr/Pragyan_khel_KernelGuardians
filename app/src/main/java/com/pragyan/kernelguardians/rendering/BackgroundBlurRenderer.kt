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
 * using a GPU fragment shader blur:
 *
 *   - Subject pixels (white in mask) → rendered sharp.
 *   - Background pixels (black in mask) → 9-tap Gaussian blur applied.
 *
 * Both the camera [Bitmap] and mask [Bitmap] are uploaded as GL textures
 * each frame. The fragment shader samples them and blends accordingly.
 *
 * Usage: set [cameraBitmap] and [maskBitmap] before [GLSurfaceView.requestRender].
 */
class BackgroundBlurRenderer : GLSurfaceView.Renderer {

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

    /**
     * Fragment shader:
     *  - Samples [uCamera] and [uMask].
     *  - If mask ≈ 1 (subject) → output camera pixel directly.
     *  - Else → output 9-tap Gaussian-blurred camera pixel.
     */
    private val FRAGMENT_SHADER = """
        precision mediump float;
        uniform sampler2D uCamera;
        uniform sampler2D uMask;
        varying vec2 vTexCoord;
        uniform vec2 uTexelSize;  // 1/width, 1/height
        uniform float uBlurRadius;

        vec4 blur9(sampler2D tex, vec2 uv, vec2 texel) {
            // 3x3 Gaussian weights (approx)
            float w[9];
            w[0]=0.0625; w[1]=0.125; w[2]=0.0625;
            w[3]=0.125;  w[4]=0.25;  w[5]=0.125;
            w[6]=0.0625; w[7]=0.125; w[8]=0.0625;
            vec2 offsets[9];
            offsets[0]=vec2(-1.,-1.); offsets[1]=vec2(0.,-1.); offsets[2]=vec2(1.,-1.);
            offsets[3]=vec2(-1., 0.); offsets[4]=vec2(0., 0.); offsets[5]=vec2(1., 0.);
            offsets[6]=vec2(-1., 1.); offsets[7]=vec2(0., 1.); offsets[8]=vec2(1., 1.);
            vec4 color = vec4(0.0);
            for (int i = 0; i < 9; i++) {
                color += texture2D(tex, uv + offsets[i] * texel * uBlurRadius) * w[i];
            }
            return color;
        }

        void main() {
            float maskVal = texture2D(uMask, vTexCoord).r;
            vec4 sharp    = texture2D(uCamera, vTexCoord);
            vec4 blurred  = blur9(uCamera, vTexCoord, uTexelSize);
            // Smooth transition at subject boundary
            gl_FragColor  = mix(blurred, sharp, step(0.5, maskVal));
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
    private var program     = 0
    private var texCamera   = 0
    private var texMask     = 0

    // ── Public inputs (set from main thread) ───────────────────────────────
    @Volatile var cameraBitmap: Bitmap? = null
    @Volatile var maskBitmap:   Bitmap? = null
    @Volatile var blurRadius: Float = 3f

    private var viewW = 1; private var viewH = 1

    // ── GLSurfaceView.Renderer ─────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)

        quadBuffer = ByteBuffer.allocateDirect(QUAD_VERTICES.size * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer()
            .apply { put(QUAD_VERTICES); position(0) }

        program   = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        texCamera = genTexture()
        texMask   = genTexture()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width; viewH = height
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)

        val cam  = cameraBitmap ?: return
        val mask = maskBitmap   ?: return

        GLES20.glUseProgram(program)

        // Upload textures
        uploadTexture(GLES20.GL_TEXTURE0, texCamera, cam)
        uploadTexture(GLES20.GL_TEXTURE1, texMask,   mask)

        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uCamera"), 0)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "uMask"),   1)
        GLES20.glUniform2f(GLES20.glGetUniformLocation(program, "uTexelSize"),
            1f / viewW, 1f / viewH)
        GLES20.glUniform1f(GLES20.glGetUniformLocation(program, "uBlurRadius"), blurRadius)

        val stride = 4 * 4  // 4 floats × 4 bytes
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

    // ── Helpers ─────────────────────────────────────────────────────────────

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
