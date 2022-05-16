package com.example.catchtheball.sprite

import android.graphics.Canvas
import android.graphics.RectF

abstract class Sprite {

    protected val rect = RectF()

    abstract fun touch(x: Float, y: Float)
    abstract fun playField(width: Int, height: Int)
    abstract fun update(dirty: RectF, timeDelta: Float): Int
    abstract fun draw(c: Canvas)
    abstract fun reset()

    fun unionRect(dirty: RectF) {
        dirty.union(rect)
    }
}