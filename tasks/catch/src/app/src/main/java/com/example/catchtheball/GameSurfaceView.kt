package com.example.catchtheball

import android.annotation.SuppressLint
import android.content.Context
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.lifecycle.LiveData

class GameSurfaceView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0,
    defStyleRes: Int = 0
) : SurfaceView(context, attrs, defStyleAttr, defStyleRes), SurfaceHolder.Callback {

    init {
        keepScreenOn = true
        holder.addCallback(this)
    }

    private val renderThread: GameRenderThread by lazy {
        GameRenderThread(
            holder,
            true
        )
    }

    val score: LiveData<Int> = renderThread.score

    val lives: LiveData<Int> = renderThread.lives

    val gameOver: LiveData<Boolean> = renderThread.gameOver

    override fun surfaceCreated(holder: SurfaceHolder) {
        renderThread.holder = holder
        renderThread.start()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        renderThread.holder = holder
        renderThread.setSize(width, height)
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        renderThread.holder = holder
        renderThread.quit()
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        renderThread.onTouchEvent(event.x, event.y)
        return super.onTouchEvent(event)
    }

    fun onResume() {
        renderThread.onResume()
    }

    fun onPause() {
        renderThread.onPause()
    }
}