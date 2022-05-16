package com.example.catchtheball.sprite

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import androidx.annotation.FloatRange
import com.google.game.rl.RLTask
import java.util.Random


class Ball(
    private val minSpeed: Float = 1f,
    private val maxSpeed: Int = 15,
    private val rand: Random = Random(),
    private val bouncing: Boolean = false,
    @FloatRange(from = 0.0, to = 1.0)
    private val paddleWidthInPercent: Float = 0.10f,
    private var staticLeft: Boolean = false,
    private var staticTop: Boolean = false,
    @FloatRange(from = 0.0, to = 1.0)
    private val maxBottomInPercent: Float? = null,
    @FloatRange(from = 0.0, to = 1.0)
    private val maxRightInPercent: Float? = null,
    private var surfaceWidth: Int,
    private var surfaceHeight: Int
) : Sprite() {

    private val _paint: Paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
    }

    private var _speed: Float? = null
    private val speed: Float
        get() {
            val newSpeed: Float = _speed ?: minSpeed + rand.nextInt(maxSpeed)
            _speed = newSpeed
            return newSpeed
        }

    @FloatRange(from = 0.0, to = 1.0)
    private var _leftInPercent: Float? = null
    private val leftInPercent: Float
        get() {
            val newLeftInPercent: Float = _leftInPercent ?: 0f + (rand.nextInt(100) / 100f)
            _leftInPercent = newLeftInPercent
            return newLeftInPercent
        }

    @FloatRange(from = 0.0, to = 1.0)
    private var _topInPercent: Float? = null
    private val topInPercent: Float
        get() {
            val newTopInPercent: Float = _topInPercent ?: 0f + (rand.nextInt(100) / 100f)
            _topInPercent = newTopInPercent
            return newTopInPercent
        }

    private var _xVelDir: Float = 1F
    private var _yVelDir: Float = speed
    private val _size
        get() = surfaceWidth / 20.0f

    private var _x: Float? = null
    private val x: Float
        get() {
            val x: Float = _x ?: surfaceWidth / 2f
            _x = x
            return x
        }

    private val _paddleWidth
        get() = surfaceWidth * paddleWidthInPercent

    override fun reset() {
        _speed = null
        _leftInPercent = null
        _topInPercent = null
        rect.apply {
            top = 0f
            bottom = 0f
            left = 0f
            right = 0f
        }
    }

    override fun touch(x: Float, y: Float) {
        _x = x
    }

    override fun playField(width: Int, height: Int) {
        surfaceWidth = width
        surfaceHeight = height
    }

    override fun update(dirty: RectF, timeDelta: Float): Int {
        var result = 0
        val rectBottom = rect.top + _size
        if (_x != null && rectBottom >= surfaceHeight) {

            val paddleRect = RectF()
            paddleRect.bottom = surfaceHeight.toFloat()
            paddleRect.top = 0f // do not matter
            paddleRect.left = x - (_paddleWidth / 2f)
            paddleRect.right = paddleRect.left + _paddleWidth
            if (RectF.intersects(paddleRect, rect)) {
                result = +1 // game won
                return result // won
            }
        }
        if (maxBottomInPercent != null) {
            if (rectBottom / surfaceHeight >= maxBottomInPercent) {
                result = -1 // game lost
                return result
            }
        }
        if (maxRightInPercent != null) {
            if (rect.left / surfaceWidth >= maxRightInPercent) {
                result = -1 // game lost
                return result
            }
        }

        if (!staticLeft) {
            rect.left += _xVelDir * timeDelta
        } else {
            rect.left = surfaceWidth * leftInPercent
        }

        if (!staticTop) {
            rect.top += _yVelDir * timeDelta
        } else {
            rect.top = surfaceHeight * topInPercent
        }
        rect.right = rect.left + _size
        rect.bottom = rect.top + _size

        if (rect.left <= 0) {
            rect.offset(-rect.left, 0F)
            if (bouncing) {
                _xVelDir = -_xVelDir // change direction
            }
        } else if (rect.right >= surfaceWidth) {
            rect.offset(surfaceWidth - rect.right, 0F)
            if (bouncing) {
                _xVelDir = -_xVelDir // change direction
            }
        }
        if (rect.top <= 0) {
            rect.offset(0f, -rect.top)
            if (bouncing) {
                _yVelDir = -_yVelDir
            }
        } else if (rect.bottom >= surfaceHeight) {
            rect.offset(0f, surfaceHeight - rect.bottom)
            if (bouncing) {
                _yVelDir = -_yVelDir
            }
        }
        return result
    }

    override fun draw(c: Canvas) {
        RLTask.get().logExtra("ball", "[${rect.centerX().toInt()}, ${rect.centerY().toInt()}]")
        c.drawCircle(
            rect.centerX(),
            rect.centerY(),
            rect.width() / 2.0f,
            _paint
        )
    }
}