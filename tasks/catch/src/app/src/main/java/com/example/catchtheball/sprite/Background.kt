package com.example.catchtheball.sprite

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.RectF
import com.example.catchtheball.sprite.Sprite


class Background : Sprite() {

    override fun reset() {
        // DO NOTING
    }

    override fun touch(x: Float, y: Float) {
        // DO NOTHING
    }

    override fun playField(width: Int, height: Int) {
        // DO NOTHING
    }

    override fun update(dirty: RectF, timeDelta: Float): Int {
        // DO NOTHING
        return 0
    }

    override fun draw(c: Canvas) {
        c.drawColor(Color.BLACK)
    }
}