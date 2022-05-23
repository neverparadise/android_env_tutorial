package com.example.vokram.mdp

import org.json.JSONArray

data class State(val id: String) {

  constructor(
    id: Int
  ) : this(
    id.toString()
  )

  fun toJSON() = id

  companion object {
    fun fromJSONArray(jsonArray: JSONArray): List<State> {
      val states = mutableListOf<State>()
      for (i in 0 until jsonArray.length()) {
        states.add(State(jsonArray.getString(i)))
      }
      return states
    }
  }
}