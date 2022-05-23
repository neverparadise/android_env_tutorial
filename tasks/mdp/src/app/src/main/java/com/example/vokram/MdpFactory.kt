package com.example.vokram

import android.content.Context
import android.util.Log
import com.example.vokram.mdp.Action as A
import com.example.vokram.mdp.Probability as P
import com.example.vokram.mdp.Reward as R
import com.example.vokram.mdp.State as S
import com.example.vokram.mdp.Transition as T

object MdpFactory {

  private val TAG = MdpFactory::class.java.simpleName

  fun defaultMdp(context: Context): Mdp {
    val sEarth = S(context.getString(com.example.vokram.R.string.state_earth))
    val sOrbit = S(context.getString(com.example.vokram.R.string.state_orbit))
    val sSpaceStation = S(context.getString(com.example.vokram.R.string.state_space_station))
    val sMoon = S(context.getString(com.example.vokram.R.string.state_moon))
    val sMars = S(context.getString(com.example.vokram.R.string.state_mars))

    val aFlyingSaucer = A(context.getString(com.example.vokram.R.string.action_flying_saucer))
    val aCatapult = A(context.getString(com.example.vokram.R.string.action_catapult))
    val aBus = A(context.getString(com.example.vokram.R.string.action_bus))
    val aSpaceship = A(context.getString(com.example.vokram.R.string.action_spaceship))
    val aRocket = A(context.getString(com.example.vokram.R.string.action_rocket))
    val aSpacecraft = A(context.getString(com.example.vokram.R.string.action_spacecraft))
    val aBike = A(context.getString(com.example.vokram.R.string.action_bike))
    val aTeleportation = A(context.getString(com.example.vokram.R.string.action_teleportation))

    return MdpBuilder()
      .addStates(
        listOf(
          sEarth, sOrbit, sSpaceStation, sMoon, sMars
        )
      )
      .setInitialState(sEarth)
      .setTerminalState(sMars)
      .addActions(
        listOf(
          aSpaceship, aSpacecraft, aBus, aBike, aFlyingSaucer, aRocket, aTeleportation, aCatapult
        )
      )
      .addTransitions(
        listOf(
          T(sEarth, aRocket, sMoon, P(0.5f), R(-1f)),
          T(sEarth, aRocket, sSpaceStation, P(0.5f), R(0f)),
          T(sEarth, aFlyingSaucer, sOrbit, P(1.0f), R(10f)),
          T(sMoon, aSpaceship, sMars, P(1.0f), R(30f)),
          T(sMoon, aSpacecraft, sEarth, P(1.0f), R(0.5f)),
          T(sSpaceStation, aBus, sSpaceStation, P(1.0f), R(-1f)),
          T(sSpaceStation, aBike, sEarth, P(1.0f), R(0f)),
          T(sOrbit, aTeleportation, sEarth, P(1.0f), R(-15f)),
          T(sOrbit, aCatapult, sOrbit, P(1.0f), R(-1f))
        )
      )
      .build()
  }

  fun sixArmsMdp(): Mdp {
    val transitions = getTransitionsSixArms()
    return MdpBuilder()
      .addStates((0..7).map { S(it) })
      .setInitialState(S(0))
      .addActions(('A'..'E').map { A(it.toString()) })
      .addTransitions(transitions)
      .build()
  }

  private fun getTransitionsSixArms(): List<T> {
    TODO("NOT READY")
  }

  fun riverSwimMdp(): Mdp {
    // config river swim parameters
    val riverLength = 5
    val rRiverMouth = R(5.0)
    val rRiverSource = R(10000.0)
    val pSwimAdvance = P(0.3)
    val pSwimRegress = P(0.1)

    val transitions = getTransitionsRiverSwim(
      riverLength,
      rRiverMouth,
      rRiverSource,
      pSwimAdvance,
      pSwimRegress
    )
    Log.d(TAG, "transitions: $transitions")

    return MdpBuilder()
      .addStates((0..riverLength).map { S(it) })
      .setInitialState(S(1))
      .addActions(('A'..'B').map { A(it) })
      .addTransitions(transitions)
      .build()
  }

  // Construct transitions (s_t, action, s_t+1, probability, reward).
  // A: swim forward. B: rest and fall downstream
  @Suppress("SameParameterValue")
  private fun getTransitionsRiverSwim(
    riverLength: Int,
    rRiverMouth: R,
    rRiverSrc: R,
    pSwimAdvance: P,
    pSwimRegress: P
  ): List<T> {
    val transitions = mutableListOf<T>()

    val sRiverSrc = S(riverLength)
    val sBeforeRiverSrc = S(riverLength - 1)
    val pCertainty = P(1.0)
    val rNone = R(0.0)
    val pSwimNoAdvance = pCertainty - pSwimAdvance
    val pSwimStay = pSwimNoAdvance - pSwimRegress

    val aSwimFwd = A("A")
    val aRestFallDS = A("B")
    // River Mouth
    transitions.add(T(S(0), aRestFallDS, S(0), pCertainty, rRiverMouth))
    transitions.add(T(S(0), aSwimFwd, S(0), pSwimNoAdvance, rNone))
    transitions.add(T(S(0), aSwimFwd, S(1), pSwimAdvance, rNone))

    for (s in 1 until riverLength) {
      transitions.add(T(S(s), aRestFallDS, S(s - 1), pCertainty, rNone))
      transitions.add(T(S(s), aSwimFwd, S(s - 1), pSwimRegress, rNone))
      transitions.add(T(S(s), aSwimFwd, S(s), pSwimStay, rNone))
      transitions.add(T(S(s), aSwimFwd, S(s + 1), pSwimAdvance, rNone))
    }

    // River Source
    transitions.add(T(sRiverSrc, aRestFallDS, sBeforeRiverSrc, pCertainty, rNone))
    transitions.add(T(sRiverSrc, aSwimFwd, sRiverSrc, pSwimAdvance, rRiverSrc))
    transitions.add(T(sRiverSrc, aSwimFwd, sBeforeRiverSrc, pSwimNoAdvance, rNone))

    Log.d(TAG, "transitions: $transitions")
    return transitions
  }

  fun floodItMdp(): Mdp {
    val actions = listOf(
      A("a"), // PURPLE
      A("b"), // BLUE
      A("c"), // GREEN
      A("d"), // YELLOW
      A("e"), // RED
      A("f") // PINK
    )
    val states = (0..1).map { S(it) }

    val transitions = mutableListOf<T>()

    states.forEach { state ->
      states
        .filter { it != state }
        .forEach { otherState ->
          actions.forEach { action ->
            transitions.add(T(state, action, otherState, P(1.0f / (states.size - 1)), R(1f)))
          }
        }
    }

    return MdpBuilder()
      .addStates(states)
      .setInitialState(states.first())
      .addActions(actions)
      .addTransitions(transitions)
      .build()
  }
}