package com.example.vokram.ui

import android.content.Context
import android.util.TypedValue


@Suppress("unused", "MemberVisibilityCanBePrivate")
class ResourcesUtils {

  companion object {

    fun getDimension(context: Context, unit: Int, value: Int): Float {
      return TypedValue.applyDimension(
        unit,
        value.toFloat(),
        context.resources.displayMetrics
      )
    }

    fun convertSPtoPX(context: Context, sp: Int): Float {
      return getDimension(context, TypedValue.COMPLEX_UNIT_SP, sp)
    }

    fun convertDPtoPX(context: Context, dp: Int): Float {
      return getDimension(context, TypedValue.COMPLEX_UNIT_DIP, dp)
    }

    fun convertPXtoDP(context: Context, px: Int): Float {
      return px / context.resources.displayMetrics.density
    }
  }
}