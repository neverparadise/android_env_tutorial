package com.example.catchtheball.sprite

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.annotation.FloatRange
import com.google.game.rl.RLTask

class Paddle(
    @FloatRange(from = 0.0, to = 1.0)
    private val _widthInPercent: Float = 0.10f,
    @FloatRange(from = 0.0, to = 1.0)
    private val _heightInPercent: Float = 0.01f,
    private var _width: Int,
    private var _height: Int
) : Sprite() {

    private val _paint: Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private var _x: Float? = null

    private val _paddleWidth
        get() = _width * _widthInPercent

    private val _paddleHeight
        get() = _height * _heightInPercent

    override fun reset() {
        // DO NOTING
    }

    override fun touch(x: Float, y: Float) {
        _x = x
    }

    override fun playField(width: Int, height: Int) {
        _width = width
        _height = height
    }

    override fun update(dirty: RectF, timeDelta: Float): Int {
        rect.bottom = _height.toFloat()
        rect.top = rect.bottom - _paddleHeight

        val x: Float = _x ?: _width / 2f
        if (_x == null) {
            _x = x
        }

        rect.left = x - (_paddleWidth / 2f)
        rect.right = rect.left + _paddleWidth
        if (rect.left < 0f) {
            rect.left = 0f
        }
        if (rect.right > _width) {
            rect.right = _width.toFloat()
        }
        return 0 // handled in ball
    }

    override fun draw(c: Canvas) {
        RLTask.get().logExtra("paddle", "[${rect.centerX().toInt()}, ${rect.centerY().toInt()}]")
        RLTask.get().logExtra("paddle_width", "[${_paddleWidth.toInt()}]")
        c.drawRect(rect, _paint)
    }
}