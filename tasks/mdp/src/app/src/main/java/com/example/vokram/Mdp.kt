package com.example.vokram

import android.util.Log
import com.example.vokram.math.RandomGenerator
import com.example.vokram.mdp.Action
import com.example.vokram.mdp.Probability
import com.example.vokram.mdp.Reward
import com.example.vokram.mdp.State
import com.example.vokram.mdp.Transition
import com.example.vokram.mdp.TransitionTarget
import org.json.JSONArray
import org.json.JSONObject

data class Mdp constructor(
  val randGenerator: RandomGenerator = RandomGenerator(),
  val states: Map<State, State>,
  val actions: Map<Action, Action>,
  val initialStatesProb: List<Pair<State, Probability>>,
  val terminalState: State?,
  val transitions: Map<Pair<State, Action>, List<TransitionTarget>>
) {

  val initialState: State
    get() {
      val r = randGenerator.nextFloat()
      var p = 0f
      for ((state, probability) in initialStatesProb) {
        p += probability.value
        if (r <= p) {
          return state
        }
      }
      throw AssertionError("Failed to select initial state from $initialStatesProb!")
    }

  val initialStatesProbJSON: String
    get() {
      val jsonArray = JSONArray()
      for ((state, probability) in initialStatesProb) {
        jsonArray.put(
          JSONObject(
            mapOf(
              "S" to state.id,
              "P" to probability.value
            )
          )
        )
      }
      return jsonArray.toString()
    }

  val statesJSON: String
    get() {
      val jsonArray = JSONArray()
      for (state in states.values) {
        jsonArray.put(state.toJSON())
      }
      return jsonArray.toString()
    }

  val actionsJSON: String
    get() {
      val jsonArray = JSONArray()
      for (action in actions.values) {
        jsonArray.put(action.toJSON())
      }
      return jsonArray.toString()
    }

  val transitionsJSON: String
    get() {
      val jsonArray = JSONArray()
      for ((stateAction, transitionTargets) in transitions) {
        for (transitionTarget in transitionTargets) {
          jsonArray.put(
            Transition(
              stateAction.first,
              stateAction.second,
              transitionTarget.state,
              transitionTarget.probability,
              transitionTarget.reward
            ).toJSON()
          )
        }
      }
      return jsonArray.toString()
    }

  fun performAction(currentState: State, action: Action): Pair<State, Reward> {
    val targets = transitions[Pair(currentState, action)]
    targets ?: run {
      throw AssertionError("No transition found for ($currentState, $action)!")
    }
    val r = randGenerator.nextFloat()
    var p = 0f
    for ((state, probability, reward) in targets) {
      p += probability.value
      if (r <= p) {
        return Pair(state, reward)
      }
    }
    throw AssertionError("Failed to select next state for ($currentState, $action)!")
  }

  fun isTerminalState(currentState: State): Boolean {
    return terminalState == currentState
  }

  fun validate() {
    if (states.isEmpty()) {
      throw AssertionError("Must add at least one state")
    }
    if (actions.isEmpty()) {
      throw AssertionError("Must add at least one action")
    }
    if (transitions.isEmpty()) {
      throw AssertionError("Must add at least one transition")
    }

    // Make sure all transition probabilities add up to 1.
    for ((fromStateAction, transitionTargets) in transitions) {
      var acc = 0.0f
      for ((_, probability) in transitionTargets) {
        acc += probability.value
      }
      if (acc != 1.0f) {
        throw AssertionError("Probabilities do not add up to 1.0 from state ${fromStateAction.first}, total is $acc!")
      }
    }

    // validate initial state(s) probability
    var acc = 0.0f
    for ((_, prob) in initialStatesProb) {
      acc += prob.value
    }
    if (acc != 1.0f) {
      throw AssertionError("Initial state probabilities do not add up to 1.0 for initial states ${initialStatesProb}, total is $acc!")
    }

    // Make sure that all states are reachable from the initial state(s)
    val visitedStates = mutableListOf<State>()
    for ((state, _) in initialStatesProb) {
      visit(state, visitedStates)
    }
    if (states.size != visitedStates.size) {
      throw AssertionError("Not all states are reachable! ($states != $visitedStates)")
    }
  }

  private fun visit(
    currentState: State,
    visitedStates: MutableList<State>
  ) {
    if (visitedStates.contains(currentState)) {
      return
    }
    visitedStates.add(currentState)
    for ((fromStateAction, transitionTargets) in transitions) {
      if (fromStateAction.first != currentState) {
        continue
      }
      for ((toState, _, _) in transitionTargets) {
        if (visitedStates.contains(toState)) {
          continue
        }
        visit(toState, visitedStates)
      }
    }
  }

  fun setSeed(seed: Long) {
    randGenerator.setSeed(seed)
  }

  fun getStateRatio(state: State): Float {
    return getIndex(states, state).toFloat() / states.size
  }

  fun getActionRatio(action: Action): Float {
    return getIndex(actions, action).toFloat() / actions.size
  }

  private fun <Type : Any> getIndex(
    map: Map<Type, Type>,
    objectType: Type
  ): Int {
    for ((index, key) in map.keys.withIndex()) {
      if (key == objectType) {
        return index
      }
    }
    throw AssertionError("Failed to find $objectType in $map!")
  }

  fun getActionsFromState(fromState: State): MutableList<Action> = transitions
    .filter { it.key.first == fromState }
    .map { it.key.second }
    .toMutableList()
}

data class MdpBuilder(
  var randGenerator: RandomGenerator = RandomGenerator(),
  var states: MutableMap<State, State> = mutableMapOf(),
  var actions: MutableMap<Action, Action> = mutableMapOf(),
  var initialStatesProb: MutableList<Pair<State, Probability>> = mutableListOf(),
  var terminalState: State? = null,
  var transitions: MutableMap<Pair<State, Action>, MutableList<TransitionTarget>> = mutableMapOf()
) {

  companion object {
    private val TAG = MdpBuilder::class.java.simpleName
  }

  fun addStates(states: Collection<State>) = apply {
    for (state in states) {
      addState(state)
    }
  }

  private fun addState(state: State) = apply { this.states[state] = state }

  private fun addStateIfNecessary(state: State) = apply {
    if (!states.containsKey(state)) {
      addState(state)
    }
  }

  fun addActions(actions: Collection<Action>) = apply {
    for (action in actions) {
      addAction(action)
    }
  }

  private fun addAction(action: Action) = apply { this.actions[action] = action }

  private fun addActionIfNecessary(action: Action) = apply {
    if (!actions.containsKey(action)) {
      addAction(action)
    }
  }

  fun setInitialState(initialState: State) = apply {
    this.initialStatesProb.clear()
    addInitialStateProb(initialState)
  }

  fun addInitialStatesProb(jsonArray: JSONArray) = apply {
    for (i in 0 until jsonArray.length()) {
      jsonArray.getJSONObject(i).let { jsonObject ->
        addInitialStateProb(
          jsonObject.getString("S"),
          jsonObject.getDouble("P")
        )
      }
    }
  }

  private fun addInitialStateProb(
    initialStateId: String,
    probabilityValue: Double = 1.0
  ) = apply {
    addInitialStateProb(
      State(initialStateId),
      Probability(probabilityValue)
    )
  }

  private fun addInitialStateProb(
    initialState: State,
    probability: Probability = Probability(1.0)
  ) = apply {
    if (states.isNotEmpty()) { // validate
      if (!states.containsKey(initialState)) {
        throw AssertionError("Initial state '${initialState}' not in states '$states'!")
      }
    }
    var acc = 0.0f
    for ((_, prob) in initialStatesProb) {
      acc += prob.value
    }
    acc += probability.value
    if (acc > 1.0f) {
      throw AssertionError("Initial state probabilities add up to more than 1.0 from ${initialState}, total is $acc!")
    }
    this.initialStatesProb.add(Pair(initialState, probability))
  }

  fun setTerminalState(terminalState: State) = apply {
    if (states.isNotEmpty()) { // validate
      if (!states.containsKey(terminalState)) {
        throw AssertionError("Terminal state '${terminalState}' not in states '$states'!")
      }
    }
    this.terminalState = terminalState
  }

  fun addTransitions(transitions: Collection<Transition>) = apply {
    for (transition in transitions) {
      addTransition(transition)
    }
  }

  private fun addTransition(
    transition: Transition
  ) = apply {
    addTransition(
      transition.fromState,
      transition.action,
      transition.toState,
      transition.probability,
      transition.reward
    )
  }

  private fun addTransition(
    fromState: State,
    action: Action,
    toState: State,
    probability: Probability,
    reward: Reward
  ) = apply {
    addTransition(Pair(fromState, action), TransitionTarget(toState, probability, reward))
  }

  private fun addTransition(
    fromStateAction: Pair<State, Action>,
    transitionTarget: TransitionTarget
  ) = apply {
    if (states.isNotEmpty()) { // validate
      if (!states.containsKey(fromStateAction.first)) {
        throw AssertionError("From state '${fromStateAction.first}' not in states '$states'!")
      }
      if (!states.containsKey(transitionTarget.state)) {
        throw AssertionError("To state '${transitionTarget.state}' not in states '$states'!")
      }
    }
    if (actions.isNotEmpty()) { // validate
      if (!actions.containsKey(fromStateAction.second)) {
        throw AssertionError("Action '${fromStateAction.second}' not in actions '$actions'!")
      }
    }
    transitions
      .getOrPut(fromStateAction, { arrayListOf() })
      .add(transitionTarget)
  }

  fun build(): Mdp {
    if (states.isEmpty()) { // auto-complete
      initialStatesProb
        .map { it.first }
        .forEach { addStateIfNecessary(it) }
      for (transition in transitions) {
        addStateIfNecessary(transition.key.first)
        for (transitionTarget in transition.value) {
          addStateIfNecessary(transitionTarget.state)
        }
      }
      terminalState?.let {
        addStateIfNecessary(it)
      }
      Log.i(TAG, "Generated states: $states")
    }
    if (actions.isEmpty()) { // auto-complete
      for (transition in transitions) {
        addActionIfNecessary(transition.key.second)
      }
      Log.i(TAG, "Generated actions: $actions")
    }
    if (initialStatesProb.isEmpty()) {
      throw AssertionError("No initial state(s) set")
    }
    return Mdp(
      randGenerator,
      states,
      actions,
      initialStatesProb,
      terminalState,
      transitions
    )
  }
}