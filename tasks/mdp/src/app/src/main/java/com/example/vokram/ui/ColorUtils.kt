package com.example.vokram.ui

import android.graphics.Color
import android.os.Build
import android.widget.Button
import androidx.annotation.ColorInt

class ColorUtils {

  companion object {
    /** Maps a ratio (0.0 to 1.0) to a unique color value by varying hue.
     * A hueOffset (0.0 to 1.0) can be provided to offset the colors returned */
    @ColorInt
    fun ratioToColor(ratio: Float, hueOffset: Float): Int {
      // Vary the color (hue) based on state ratio
      val hsv = floatArrayOf(0f, 1f, 1f)
      hsv[0] = (ratio + hueOffset) * 360f % 360f
      return Color.HSVToColor(hsv)
    }

    fun isDarker(color: Int) : Boolean {
      return androidx.core.graphics.ColorUtils.calculateLuminance(color) < 0.5
    }

    fun setTextAppearanceLight(newActionButton: Button) {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        newActionButton.setTextAppearance(android.R.style.TextAppearance_Inverse)
      } else {
        newActionButton.setTextColor(Color.WHITE)
      }
    }

    fun setTextAppearanceDark(newActionButton: Button) {
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
        newActionButton.setTextAppearance(android.R.style.TextAppearance)
      } else {
        newActionButton.setTextColor(Color.BLACK)
      }
    }
  }
}
