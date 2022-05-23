package com.example.vokram

import android.content.Context
import android.media.MediaPlayer
import androidx.annotation.RawRes

class SoundPlayer(context: Context, @RawRes resId: Int) {

  private val mediaPlayer: MediaPlayer? = MediaPlayer.create(context, resId)

  fun play() {
    mediaPlayer?.start()
  }

  fun release() {
    mediaPlayer?.release()
  }
}

