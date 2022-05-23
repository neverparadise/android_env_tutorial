package com.example.vokram.mdp

data class Probability(val value: Float) {

  constructor(
    id: Double
  ) : this(
    id.toFloat()
  )

  constructor(
    id: String
  ) : this(
    id.toFloat()
  )

  operator fun minus(other: Probability): Probability {
    return Probability(this.value - other.value)
  }
}