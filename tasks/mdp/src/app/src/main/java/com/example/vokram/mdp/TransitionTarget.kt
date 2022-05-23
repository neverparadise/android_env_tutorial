package com.example.vokram.mdp

data class TransitionTarget(
  val state: State,
  val probability: Probability,
  val reward: Reward
)