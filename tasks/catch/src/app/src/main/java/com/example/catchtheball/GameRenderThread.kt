package com.example.catchtheball

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.PorterDuff
import android.graphics.RectF
import android.os.SystemClock
import android.view.SurfaceHolder
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import com.example.catchtheball.sprite.Background
import com.example.catchtheball.sprite.Ball
import com.example.catchtheball.sprite.Paddle
import com.example.catchtheball.sprite.Sprite
import com.google.game.rl.RLTask
import java.util.Random

class GameRenderThread(
    var holder: SurfaceHolder,
    private val hardwareAccelerated: Boolean = false
) : Thread(GameRenderThread::class.java.simpleName) {

    companion object {
        const val LIVES = 0
    }

    private var _lives: MutableLiveData<Int> = MutableLiveData<Int>()
        .apply { value = RLTask.get().get("lives", LIVES) }
    val lives: LiveData<Int> = _lives

    private var _quit = false

    private var _score: MutableLiveData<Int> = MutableLiveData<Int>()
        .apply { value = 0 }
    val score: LiveData<Int> = _score

    private val _gameOver: MutableLiveData<Boolean> = MutableLiveData<Boolean>()
        .apply {
            value = false
        }

    val gameOver: LiveData<Boolean> = _gameOver

    private var _width: Int = 0
    private var _height: Int = 0

    private var _paddleWidthInPercent: Float = 0.25f


    private val rand = Random()

    private val _sprites: MutableList<Sprite> = mutableListOf(
        Background(),
        resetBall(),
        Paddle(
            _widthInPercent = _paddleWidthInPercent,
            _width = _width,
            _height = _height
        )
    )

    private fun resetBall(): Sprite {
        return Ball(
            minSpeed = 10_000f,
            maxSpeed = 1,
            rand = rand,
            bouncing = false,
            paddleWidthInPercent = _paddleWidthInPercent,
            staticLeft = true,
            staticTop = false,
            maxBottomInPercent = 1.00f,
            maxRightInPercent = null,
            surfaceWidth = _width,
            surfaceHeight = _height
        )
    }

    fun onTouchEvent(x: Float, y: Float) {
        _sprites.forEach { sprite ->
            sprite.apply {
                touch(x, y)
            }
        }
    }

    fun setSize(width: Int, height: Int) {
        _width = width
        _height = height
        _sprites.forEach { sprite ->
            sprite.apply {
                playField(width, height)
            }
        }
    }

    override fun run() {
        super.run()
        _quit = false
        val dirtyF = RectF()
        var currentTime = SystemClock.elapsedRealtime().toFloat()
        while (!_quit) {
            val newTime = SystemClock.elapsedRealtime().toFloat()
            var frameTime = (newTime - currentTime) / 100_000.0f
            currentTime = newTime
            dirtyF.setEmpty()
            while (frameTime > 0.0f) {
                val deltaTime: Float = frameTime
                integrate(dirtyF, deltaTime)
                frameTime -= deltaTime
            }
            render()
            sleep(100L)
        }
    }

    private fun integrate(dirty: RectF, timeDelta: Float) {
        var result = 0
        _sprites.forEach { sprite ->
            sprite.apply {
                unionRect(dirty)
                val newResult = update(dirty, timeDelta)
                if (newResult != 0) {
                    if (result == 0) { // unset
                        result = newResult
                    } else if (newResult < result) { // lost > win
                        result = newResult
                    }
                }
                unionRect(dirty)
            }
        }
        var scoreValue: Int = _score.value ?: 0
        var livesValue: Int = _lives.value ?: RLTask.get().get("lives", LIVES)
        if (result < 0) { // game lost
            scoreValue += result
            _score.postValue(scoreValue)
            RLTask.get().logScore(scoreValue)
            if (livesValue <= 0) {
                _gameOver.postValue(true)
                RLTask.get().logEpisodeEnd()
                _quit = true
            } else {
                livesValue--
                _lives.postValue(livesValue)
                RLTask.get().logExtra("lives", "[$livesValue]")
                _sprites[1].reset()
            }
        } else if (result > 0) { // won
            scoreValue += result
            _score.postValue(scoreValue)
            RLTask.get().logScore(scoreValue)
            _sprites[1].reset()
        }
    }

    private fun render() {
        val c: Canvas = if (hardwareAccelerated) {
            holder.lockHardwareCanvas()
        } else {
            holder.lockCanvas()
        }
        c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)
        _sprites.forEach { sprite ->
            sprite.apply {
                draw(c)
            }
        }
        holder.unlockCanvasAndPost(c)
    }

    fun onResume() {
        _quit = false
    }

    fun onPause() {
        _quit = true
    }

    fun quit() {
        _quit = true
        try {
            this.join()
        } catch (e: InterruptedException) {
            e.printStackTrace()
        }
    }
}