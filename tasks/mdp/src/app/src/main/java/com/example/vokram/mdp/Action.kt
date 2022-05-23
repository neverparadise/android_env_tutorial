package com.example.vokram.mdp

import org.json.JSONArray

data class Action(val id: String) {

  constructor(id: Char) : this(id.toString())

  fun toJSON() = id

  val idWithQuotes: String
    get() = "'$id'"

  val valid: Boolean = id.isNotEmpty()

  companion object {

    fun fromJSONArray(jsonArray: JSONArray): List<Action> {
      val states = mutableListOf<Action>()
      for (i in 0 until jsonArray.length()) {
        states.add(Action(jsonArray.getString(i)))
      }
      return states
    }
  }
}