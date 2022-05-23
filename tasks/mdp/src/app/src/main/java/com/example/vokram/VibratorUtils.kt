package com.example.vokram

import android.annotation.SuppressLint
import android.content.Context
import android.os.Vibrator

class VibratorUtils {

  companion object {

    @SuppressLint("MissingPermission")
    fun vibrate(context: Context, positive: Boolean) {
      val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
      vibrator.let {
        if (it.hasVibrator()) {
          if (positive) {
            @Suppress("DEPRECATION")
            vibrator.vibrate(10L)
          } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(
              longArrayOf(
                0L, // initial wait
                10L, 100L,
                10L, 100L,
                10L, 100L
              ), -1
            )
          }
        }
      }
    }
  }

}

