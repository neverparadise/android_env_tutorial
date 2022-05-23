package com.example.vokram.mdp

data class Reward(val value: Float) {

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
}