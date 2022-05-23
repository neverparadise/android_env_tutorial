package com.example.vokram.ui

import kotlin.math.cos
import kotlin.math.sin

data class ActionMovement(var directionX: Float,
                          var directionY: Float,
                          var speed: Float) {
  constructor(angle: Float,
              speed: Float) :
    this(
      cos(angle),
      sin(angle),
      speed
    )
}