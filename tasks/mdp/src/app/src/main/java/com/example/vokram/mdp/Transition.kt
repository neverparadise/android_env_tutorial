package com.example.vokram.mdp

import org.json.JSONArray
import org.json.JSONObject

data class Transition(
  val fromState: State,
  val action: Action,
  val toState: State,
  val probability: Probability,
  val reward: Reward
) {

  constructor(
    fromStateId: String,
    actionId: String,
    toStateId: String,
    probabilityValue: Double,
    rewardValue: Double
  ) : this(
    fromStateId,
    actionId,
    toStateId,
    probabilityValue.toFloat(),
    rewardValue.toFloat()
  )

  constructor(
    fromStateId: String,
    actionId: String,
    toStateId: String,
    probabilityValue: Float,
    rewardValue: Float
  ) : this(
    State(fromStateId),
    Action(actionId),
    State(toStateId),
    Probability(probabilityValue),
    Reward(rewardValue)
  )

  fun toJSON() = JSONObject(
    mapOf(
      "FS" to fromState.id,
      "A" to action.id,
      "TS" to toState.id,
      "P" to probability.value,
      "R" to reward.value
    )
  )

  companion object {

    private fun fromJSON(jsonObject: JSONObject): Transition {
      return Transition(
        jsonObject.getString("FS"),
        jsonObject.getString("A"),
        jsonObject.getString("TS"),
        jsonObject.getDouble("P"),
        jsonObject.getDouble("R")
      )
    }

    fun fromJSONArray(jsonArray: JSONArray): List<Transition> {
      val transitions = mutableListOf<Transition>()
      for (i in 0 until jsonArray.length()) {
        transitions.add(fromJSON(jsonArray.getJSONObject(i)))
      }
      return transitions
    }
  }
}