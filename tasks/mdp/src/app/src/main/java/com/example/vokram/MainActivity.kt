package com.example.vokram

import android.annotation.SuppressLint
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.text.SpannableString
import android.text.style.StyleSpan
import android.util.DisplayMetrics
import android.util.Log
import android.util.TypedValue
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.os.bundleOf
import androidx.core.view.isVisible
import androidx.core.view.plusAssign
import com.example.vokram.DialogUtils.DialogProvider
import com.example.vokram.databinding.ActivityMainBinding
import com.example.vokram.math.MathUtils
import com.example.vokram.math.RandomFloat
import com.example.vokram.math.RandomGenerator
import com.example.vokram.math.RandomInt
import com.example.vokram.mdp.Action
import com.example.vokram.mdp.State
import com.example.vokram.mdp.Transition
import com.example.vokram.ui.ActionMovement
import com.example.vokram.ui.ColorUtils
import com.example.vokram.ui.ResourcesUtils
import com.google.game.rl.RLTask
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.absoluteValue
import kotlin.math.cos
import kotlin.math.sin

/**
 * Activity where we show the MDP to the user.
 */
class MainActivity : AppCompatActivity(), Runnable, DialogProvider {

  companion object {

    private val TAG = MainActivity::class.java.simpleName

    private const val EXTRA_SHOW_EXTRA_INFO = "showExtraInfo"
    private const val EXTRA_TOTAL_NUM_STEPS = "totalNumSteps"
    private const val EXTRA_INITIAL_SEED = "initialSeed"
    private const val EXTRA_MOVEMENT_ANGLE = "movementAngle"
    private const val EXTRA_MOVEMENT_SPEED = "movementSpeed"
    private const val EXTRA_STICK_TO_GRID = "stickToGrid"
    private const val EXTRA_GRID_SIZE = "gridSize"
    private const val EXTRA_ACTION_RES_PREFIX = "actionResPrefix"
    private const val EXTRA_ACTION_WIDTH_IN_DP = "actionWidthDp"
    private const val EXTRA_ACTION_HEIGHT_IN_DP = "actionHeightDp"
    private const val EXTRA_ACTION_ARE_COLORS = "actionColors"
    private const val EXTRA_BACKGROUND_COLOR = "bgColors"

    @Deprecated(
      message = "User state+prob instead", replaceWith = ReplaceWith("EXTRA_INITIAL_STATES_PROB")
    )
    private const val EXTRA_INITIAL_STATE = "initial_state"
    private const val EXTRA_INITIAL_STATES_PROB = "initial_states_prob"
    private const val EXTRA_TERMINAL_STATE = "terminal_state"
    private const val EXTRA_STATES = "states"
    private const val EXTRA_ACTIONS = "actions"
    private const val EXTRA_TRANSITIONS = "transitions"

    fun newInstance(context: Context): Intent {
      val intent = Intent(context, MainActivity::class.java)
      intent.putExtras(
        bundleOf(
          RLTask.EXTRA_ENABLED to true
        )
      )
      return intent
    }

    fun newInstance(
      context: Context,
      movementSpeed: FloatArray,
      totalNumSteps: Int? = null,
      stickToGrid: Boolean = false,
      gridSize: Int? = null,
      actionResPrefix: String? = null,
      actionWidthInDp: Int? = null,
      actionHeightInDp: Int? = null,
      actionColors: Boolean = false,
      bgColorString: String? = null,
      mdp: Mdp
    ): Intent {
      val intent = newInstance(context)
      intent.putExtras(
        bundleOf(
          RLTask.EXTRA_GAME_CONFIG to JSONObject(
            mapOf(
              EXTRA_MOVEMENT_SPEED to movementSpeed,
              EXTRA_STICK_TO_GRID to stickToGrid,
              EXTRA_GRID_SIZE to gridSize,
              EXTRA_ACTION_RES_PREFIX to actionResPrefix,
              EXTRA_ACTION_WIDTH_IN_DP to actionWidthInDp,
              EXTRA_ACTION_HEIGHT_IN_DP to actionHeightInDp,
              EXTRA_ACTION_ARE_COLORS to actionColors,
              EXTRA_BACKGROUND_COLOR to bgColorString,
              EXTRA_STATES to mdp.statesJSON,
              EXTRA_ACTIONS to mdp.actionsJSON,
              EXTRA_TRANSITIONS to mdp.transitionsJSON,
              EXTRA_TOTAL_NUM_STEPS to totalNumSteps,
              EXTRA_INITIAL_STATES_PROB to mdp.initialStatesProbJSON
            )
          ).toString()
        )
      )
      return intent
    }

    // default global params for game view and action
    private var SHUFFLE_ACTIONS = true
    private const val UPDATE_TIME_MS = 50L
    private var totalNumSteps = -1
    private var stepsPlayed = 0
    private var showExtraInfo = false

    @Suppress("unused")
    private fun lerp(a: Float, b: Float, coefficient: Float): Float {
      return a * (1.0f - coefficient) + b * coefficient
    }
  }

  private var score = 0.0f
  private lateinit var mdp: Mdp
  private var initialSeed: Long? = null
  private lateinit var currState: State
  private val randomGenerator = RandomGenerator()
  private val handler = Handler()

  private val movements = ArrayList<ActionMovement>()
  private var movementAngle = RandomFloat(0f, Math.PI.toFloat(), 1.0f)
  private var movementSpeed = RandomFloat(0.0f, 0.0f, 1.0f)
  private var stickToGrid = false
  private var gridSize: Int? = null
  private var actionResPrefix: String? = null
  private val actionRes: Boolean
    get() = actionResPrefix?.isNotBlank() ?: false
  private var actionWidthInDp: Int? = null
  private var actionHeightInDp: Int? = null
  private var actionColors = false
  private var bgColor: String? = null

  private val positiveSound: SoundPlayer by lazy {
    SoundPlayer(this, R.raw.positive_reward)
  }
  private val negativeSound: SoundPlayer by lazy {
    SoundPlayer(this, R.raw.negative_reward)
  }

  private var btnClicked: View.OnClickListener = View.OnClickListener { v ->
    stepsPlayed++

    val action = v.tag as Action
    RLTask.get().logExtra("clicks [${action.idWithQuotes}]")
    val (state, reward) = mdp.performAction(currState, action)

    Log.d(TAG, "State change: (${currState.id}, ${action.id}) -> (${state.id}, ${reward.value})")

    currState = state
    addScore(reward.value)

    updateScoreBoard(currState)

    if (mdp.isTerminalState(currState) || stepsPlayed == totalNumSteps) {
      RLTask.get().logEpisodeEnd()
      DialogUtils.showOkDialog(
        this@MainActivity,
        TAG,
        getString(R.string.episode_ended_and_score, score.toString()),
        DialogInterface.OnClickListener { _, _ ->
          startNewGame()
        })
    }

    updateView()
  }

  private val hideReward = Runnable {
    viewBinding.rewardTv.visibility = View.GONE
  }

  private lateinit var viewBinding: ActivityMainBinding

  override fun onCreate(savedInstanceState: Bundle?) {
    RLTask.get().onCreateActivity(this)
    super.onCreate(savedInstanceState)
    viewBinding = ActivityMainBinding.inflate(layoutInflater)
    setContentView(viewBinding.root)

    initFromExtras(intent.extras)
    mdp.validate()

    startNewGame()
    handler.postDelayed(this, UPDATE_TIME_MS)

    // set the view for info such as score, current state, steps played
    viewBinding.extraInfoTv.isVisible = showExtraInfo
    updateScoreBoard(currState)
  }

  override fun onNewIntent(intent: Intent?) {
    RLTask.get().onNewIntentActivity(intent)
    super.onNewIntent(intent)
  }

  override fun onDestroy() {
    super.onDestroy()
    this.positiveSound.release()
    this.negativeSound.release()
  }

  private fun updateScoreBoard(currentState: State?) {
    val currentStateName = currentState?.id ?: "(no state)"
    @SuppressLint("SetTextI18n")
    viewBinding.extraInfoTv.text =
      getString(R.string.score_and_value, this.score.toString()) + "; " +
        getString(R.string.state_and_name, currentStateName) + "; \n" +
        "#steps played: $stepsPlayed (${if (totalNumSteps == -1) "unlimited" else (totalNumSteps - stepsPlayed).toString()} left)"
    viewBinding.stepsCountTv.text = getString(
      R.string.steps_count_and_played_total,
      stepsPlayed.toString(),
      if (totalNumSteps == -1) "∞" else totalNumSteps
    )
    viewBinding.scoreTv.text = getString(R.string.score_and_value, this.score.toString())
    viewBinding.currentStateTv.text = getString(R.string.state_and_name, currentStateName)
  }

  private fun initFromExtras(extras: Bundle?) {
    extras.printExtras(TAG)

    initGlobalParametersFromExtra()

    initMDPFromExtra()
  }

  private fun initGlobalParametersFromExtra() {
    Log.d(TAG, "Initializing global parameters from extras...\n***************")
    showExtraInfo = RLTask.get().get(EXTRA_SHOW_EXTRA_INFO, showExtraInfo)
    totalNumSteps = RLTask.get().get(EXTRA_TOTAL_NUM_STEPS, totalNumSteps)
    Log.d(TAG, "$EXTRA_TOTAL_NUM_STEPS provided: $totalNumSteps")

    val initialSeedExtra = RLTask.get().get(EXTRA_INITIAL_SEED, -1L)
    Log.d(TAG, "$EXTRA_INITIAL_SEED provided: $initialSeedExtra.")
    if (initialSeedExtra >= 0L) {
      this.initialSeed = initialSeedExtra
    }

    val movementAngleSpec = RLTask.get().get(EXTRA_MOVEMENT_ANGLE, FloatArray(0))
    Log.d(TAG, "$EXTRA_MOVEMENT_ANGLE provided: ${movementAngleSpec?.contentToString()}.")
    if (movementAngleSpec == null || movementAngleSpec.size == 3) {
      movementAngle = RandomFloat(movementAngleSpec[0], movementAngleSpec[1], movementAngleSpec[2])
    } else {
      Log.e(TAG, "$EXTRA_MOVEMENT_ANGLE should have exactly 3 floats.")
    }

    val movementSpeedSpec = RLTask.get().get(EXTRA_MOVEMENT_SPEED, FloatArray(0))
    Log.d(TAG, "$EXTRA_MOVEMENT_SPEED provided: ${movementSpeedSpec?.contentToString()}.")
    if (movementSpeedSpec != null && movementSpeedSpec.size == 3) {
      movementSpeed = RandomFloat(movementSpeedSpec[0], movementSpeedSpec[0], movementSpeedSpec[1])
    } else {
      Log.e(TAG, "$EXTRA_MOVEMENT_SPEED should have exactly 3 floats.")
    }

    stickToGrid = RLTask.get().get(EXTRA_STICK_TO_GRID, false)
    Log.d(TAG, "$EXTRA_STICK_TO_GRID provided: ${stickToGrid}.")

    val gridSizeSpec = RLTask.get().get(EXTRA_GRID_SIZE, -1)
    Log.d(TAG, "$EXTRA_GRID_SIZE provided: ${gridSizeSpec}.")
    if (gridSizeSpec > 0) {
      gridSize = gridSizeSpec
    } else {
      Log.e(TAG, "$EXTRA_GRID_SIZE should not be '$gridSizeSpec'.")
    }

    val actionResPrefixString = RLTask.get().get(EXTRA_ACTION_RES_PREFIX, "")
    Log.d(TAG, "$EXTRA_ACTION_RES_PREFIX provided: ${actionResPrefixString}.")
    if (actionResPrefixString != null && !actionResPrefixString.isBlank()) {
      actionResPrefix = actionResPrefixString
    } else {
      Log.e(TAG, "$EXTRA_BACKGROUND_COLOR should not be empty '$actionResPrefixString'.")
    }

    val actionWidthSpec = RLTask.get().get(EXTRA_ACTION_WIDTH_IN_DP, -1)
    Log.d(TAG, "$EXTRA_ACTION_WIDTH_IN_DP provided: ${actionWidthSpec}.")
    if (actionWidthSpec > 0) {
      actionWidthInDp = actionWidthSpec
    } else {
      Log.e(TAG, "$EXTRA_ACTION_WIDTH_IN_DP should not be '$actionWidthSpec'.")
    }

    val actionHeightSpec = RLTask.get().get(EXTRA_ACTION_HEIGHT_IN_DP, -1)
    Log.d(TAG, "$EXTRA_ACTION_HEIGHT_IN_DP provided: ${actionHeightSpec}.")
    if (actionHeightSpec > 0) {
      actionHeightInDp = actionHeightSpec
    } else {
      Log.e(TAG, "$EXTRA_ACTION_HEIGHT_IN_DP should not be '$actionHeightSpec'.")
    }

    actionColors = RLTask.get().get(EXTRA_ACTION_ARE_COLORS, false)
    Log.d(TAG, "$EXTRA_ACTION_ARE_COLORS provided: ${actionColors}.")

    val bgColorString = RLTask.get().get(EXTRA_BACKGROUND_COLOR, "")
    Log.d(TAG, "$EXTRA_BACKGROUND_COLOR provided: ${bgColorString}.")
    if (bgColorString != null && !bgColorString.isBlank()) {
      bgColor = bgColorString
    } else {
      Log.e(TAG, "$EXTRA_BACKGROUND_COLOR should not be empty '$bgColorString'.")
    }

    Log.d(TAG, "***************\nDone initializing global parameters from extras.")
  }

  private fun initMDPFromExtra() {
    Log.d(TAG, "Initializing the MDP from extras...\n***************")

    val mdpBuilder = MdpBuilder()

    val statesExtra = RLTask.get().get(EXTRA_STATES, JSONArray())
    if (statesExtra.length() == 0) {
      Log.w(TAG, "$EXTRA_STATES invalid: $statesExtra.")
      initDefaultMdp()
      return
    }
    val states = State.fromJSONArray(statesExtra)
    Log.d(TAG, "$EXTRA_STATES provided: $states")
    mdpBuilder.addStates(states)

    val actionsExtra = RLTask.get().get(EXTRA_ACTIONS, JSONArray())
    if (actionsExtra.length() == 0) {
      Log.w(TAG, "$EXTRA_ACTIONS invalid: $actionsExtra.")
      initDefaultMdp()
      return
    }
    val actions = Action.fromJSONArray(actionsExtra)
    Log.d(TAG, "$EXTRA_ACTIONS provided: $actions")
    mdpBuilder.addActions(actions)

    val transitionsExtra = RLTask.get().get(EXTRA_TRANSITIONS, JSONArray())
    if (transitionsExtra.length() == 0) {
      Log.w(TAG, "$EXTRA_TRANSITIONS invalid: $transitionsExtra.")
      initDefaultMdp()
      return
    }
    mdpBuilder.addTransitions(Transition.fromJSONArray(transitionsExtra))
    Log.d(TAG, "$EXTRA_TRANSITIONS provided: ${mdpBuilder.transitions}")

    val initialStatesProb = RLTask.get().get(EXTRA_INITIAL_STATES_PROB, JSONArray())
    if (initialStatesProb.length() == 0) {
      Log.w(TAG, "$EXTRA_INITIAL_STATES_PROB invalid: $initialStatesProb.")
      // TRYING FALL BACK TO OLD EXTRA_INITIAL_STATE
      @Suppress("DEPRECATION") //
      val initialStateId = RLTask.get().get(EXTRA_INITIAL_STATE, "")?.trim()
      if (initialStateId.isNullOrBlank()) {
        @Suppress("DEPRECATION") //
        Log.w(TAG, "$EXTRA_INITIAL_STATE invalid: $initialStateId.")
        initDefaultMdp()
        return
      }
      mdpBuilder.setInitialState(State(initialStateId))
      @Suppress("DEPRECATION") //
      Log.d(TAG, "$EXTRA_INITIAL_STATE provided: $initialStateId")
    } else {
      mdpBuilder.addInitialStatesProb(initialStatesProb)
    }
    Log.d(TAG, "$EXTRA_INITIAL_STATES_PROB provided: ${mdpBuilder.initialStatesProb}")

    val terminalStateId = RLTask.get().get(EXTRA_TERMINAL_STATE, "")?.trim()
    if (!terminalStateId.isNullOrBlank()) {
      mdpBuilder.setTerminalState(State(terminalStateId))
      Log.d(TAG, "$EXTRA_TERMINAL_STATE provided: $terminalStateId")
    }

    mdp = mdpBuilder.build()

    Log.d(TAG, "***************\nDone initializing the MDP from extras.")
  }

  private fun initDefaultMdp() {
    Log.d(TAG, "initializing MDP with default states, actions, transitions...")
    mdp = MdpFactory.defaultMdp(this)
  }

  private fun startNewGame() {
    stepsPlayed = 0
    score = 0.0f
    initialSeed?.let { seed ->
      mdp.setSeed(seed)
      randomGenerator.setSeed(seed)
    }
    currState = mdp.initialState
    updateView()
  }

  private fun updateView() {
    updateState()
    updateStateButtons()
    updateScoreBoard(currState)
  }

  private fun stateHueRatio(ratio: Float): Float {
    return ratio / 2.0f
  }

  private fun actionHueRatio(ratio: Float): Float {
    return ratio / 2.0f + 0.5f
  }

  private fun updateState() {
    val stateRatio = stateHueRatio(mdp.getStateRatio(currState))
    val backgroundColor: Int = if (bgColor?.isNotBlank() == true) {
      Color.parseColor(bgColor)
    } else {
      ColorUtils.ratioToColor(stateRatio, 0f)
    }
    viewBinding.actionsLayout.setBackgroundColor(backgroundColor)
  }

  @Suppress("UnnecessaryVariable")
  private fun updateStateButtons() {
    viewBinding.actionsLayout.removeAllViews()

    val actions = mdp.getActionsFromState(currState)
    if (actions.isEmpty()) { // terminal state reached
      return
    }

    gridSize?.let {
      (actions.size until it).forEach { _ ->
        actions.add(Action(""))
      }
    }

    if (SHUFFLE_ACTIONS) {
      actions.shuffle(randomGenerator)
    }
    RLTask.get().logExtra("actions ${actions
      .filter { it.valid }
      .map { it.idWithQuotes }}"
    )
    //
    val displayMetrics = DisplayMetrics()
    windowManager.defaultDisplay.getMetrics(displayMetrics)
    val screenWidthInPx = displayMetrics.widthPixels
    val screenHeightInPx = displayMetrics.heightPixels
    val usableScreenHeightInPx =
      screenHeightInPx - ResourcesUtils.convertDPtoPX(this, 60 + 35).toInt()
    val sizeFactor = actions.size
    val spaceFactor = sizeFactor + 1
    val usableScreenWidth = actionWidthInDp?.also { actionWidth ->
      ResourcesUtils.convertDPtoPX(this, actionWidth).toInt()
    } ?: screenWidthInPx / sizeFactor
    val usableScreenHeight = actionHeightInDp?.also { actionHeight ->
      ResourcesUtils.convertDPtoPX(this, actionHeight).toInt()
    } ?: usableScreenHeightInPx / sizeFactor

    val usableSpacingWidth = (screenWidthInPx - (usableScreenWidth * actions.size)) / spaceFactor
    val usableSpacingHeight =
      (usableScreenHeightInPx - (usableScreenHeight * actions.size)) / (spaceFactor - 1)
    val usableSpacing = usableSpacingWidth.coerceAtMost(usableSpacingHeight)

    val width = RandomInt(usableScreenWidth, 50, 1.0f)
    val height = RandomInt(usableScreenHeight, 50, 1.0f)
    val spacing = RandomInt(usableSpacing, 25, 1.0f)
    val positionX = if (actionRes) {
      RandomInt(usableSpacing, screenWidthInPx - usableScreenWidth, 1.0f)
    } else {
      RandomInt(usableSpacing, 64, 1.0f)
    }
    movements.clear()
    var buttonY: Int = if (stickToGrid) {
      0 // spacing is used by score board
    } else {
      MathUtils.getRandomValue(spacing, randomGenerator).absoluteValue
    }
    val verticalPaddingInPx = ResourcesUtils.convertDPtoPX(this, 8).toInt()
    val horizontalPaddingInPx = ResourcesUtils.convertDPtoPX(this, 16).toInt()
    val buttonsBounds = arrayOfNulls<IntArray>(actions.size)
    for (a in 0 until actions.size) {
      val action = actions[a]
      val buttonWidthInPx = if (stickToGrid) {
        usableScreenWidth
      } else {
        MathUtils.getRandomValue(width, randomGenerator)
      }
      val buttonHeightInPx = if (stickToGrid) {
        usableScreenHeight
      } else {
        MathUtils.getRandomValue(height, randomGenerator)
      }
      val buttonX = if (stickToGrid) {
        usableSpacingWidth + (0 until actions.size).random() * buttonWidthInPx
      } else {
        MathUtils.getRandomValue(positionX, randomGenerator).absoluteValue
      }
      val buttonSpacing = if (stickToGrid) {
        usableSpacingHeight
      } else {
        MathUtils.getRandomValue(spacing, randomGenerator).absoluteValue
      }
      val actionRatio: Float =
        if (action.valid) actionHueRatio(mdp.getActionRatio(action)) else 1F
      val topLeftX = buttonX
      val topLeftY = buttonY
      val bottomRightX = topLeftX + buttonWidthInPx
      val bottomRightY = topLeftY + buttonHeightInPx
      buttonsBounds[a] =
        if (action.valid) intArrayOf(topLeftX, topLeftY, bottomRightX, bottomRightY) else null
      val newActionButton = getNewActionButton(
        action,
        actionRatio,
        buttonX,
        buttonY,
        buttonHeightInPx,
        buttonWidthInPx,
        horizontalPaddingInPx,
        verticalPaddingInPx
      )
      if (action.valid) {
        viewBinding.actionsLayout += newActionButton
      }
      buttonY += buttonHeightInPx + buttonSpacing  // actionButtons[a].height returns 0, so the height has to be stored
      val angle = MathUtils.getRandomValue(movementAngle, randomGenerator)
      val speed = MathUtils.getRandomValue(movementSpeed, randomGenerator)
      movements.add(ActionMovement(angle, speed))
    }
    RLTask.get().logExtra(
      "buttons ${buttonsBounds
        .filterNotNull()
        .toTypedArray()
        .contentDeepToString()}"
    )
  }

  private fun getNewActionButton(
    action: Action,
    actionRatio: Float,
    buttonX: Int,
    buttonY: Int,
    buttonHeightInPx: Int,
    buttonWidthInPx: Int,
    horizontalPaddingInPx: Int,
    verticalPaddingInPx: Int
  ): View {
    val newActionView: View
    if (actionRes) {
      newActionView = ImageView(this)
      newActionView.layoutParams = ViewGroup.LayoutParams(
        buttonWidthInPx,
        buttonHeightInPx
      )
      val id = resources.getIdentifier(actionResPrefix + action.id, "drawable", packageName)
      newActionView.setImageResource(id)
      newActionView.scaleType = ImageView.ScaleType.FIT_CENTER
    } else {
      newActionView = Button(this)
      newActionView.layoutParams = ViewGroup.LayoutParams(
        ViewGroup.LayoutParams.WRAP_CONTENT,
        buttonHeightInPx
      )
      newActionView.minimumWidth = buttonWidthInPx

      val actionName = SpannableString(action.id)
      actionName.setSpan(StyleSpan(Typeface.BOLD), 0, actionName.length, 0)
      newActionView.text = if (actionRes || actionColors) {
        null
      } else {
        actionName
      }
      newActionView.setTextSize(TypedValue.COMPLEX_UNIT_DIP, 16f)
      newActionView.maxLines = 2
      val bgColor = if (actionColors) {
        action.id.toInt()
      } else {
        ColorUtils.ratioToColor(actionRatio, 0.0f)
      }
      newActionView.setBackgroundColor(bgColor)
      if (ColorUtils.isDarker(bgColor)) {
        ColorUtils.setTextAppearanceLight(newActionView)
      } else {
        ColorUtils.setTextAppearanceDark(newActionView)
      }
      newActionView.setPadding(
        horizontalPaddingInPx,
        verticalPaddingInPx,
        horizontalPaddingInPx,
        verticalPaddingInPx
      )
    }
    newActionView.x = buttonX.toFloat()
    newActionView.y = buttonY.toFloat()
    newActionView.tag = action
    newActionView.setOnClickListener(btnClicked)
    return newActionView
  }

  override fun showOkDialog(
    title: String,
    message: String,
    onClickOkListener: DialogInterface.OnClickListener
  ) {
    DialogUtils.showOkDialog(this, title, message, onClickOkListener)
  }

  override fun showOkDialog(
    titleResId: Int,
    message: String,
    onClickOkListener: DialogInterface.OnClickListener
  ) {
    showOkDialog(getString(titleResId), message, onClickOkListener)
  }

  private fun addScore(delta: Float) {
    if (delta > 0.0f) {
      this.positiveSound.play()
      VibratorUtils.vibrate(this, true)
    } else if (delta < 0.0f) {
      this.negativeSound.play()
      VibratorUtils.vibrate(this, false)
    }
    viewBinding.rewardTv.apply {
      visibility = View.GONE
      text = when {
        delta > 0.0f -> "+${delta.toInt()}"
        else -> "${delta.toInt()}"
      }
      removeCallbacks(hideReward)
      postDelayed(hideReward, 500L)
    }
    score += delta
    RLTask.get().logScore(score)
  }

  override fun run() {
    val layoutSizeX = viewBinding.actionsLayout.width.toFloat()
    val layoutSizeY = viewBinding.actionsLayout.height.toFloat()
    for (i in 0 until viewBinding.actionsLayout.childCount) {
      val child = viewBinding.actionsLayout.getChildAt(i)
      val movement = movements[i]
      val positionMaxX = layoutSizeX - child.width.toFloat()
      val positionMaxY = layoutSizeY - child.height.toFloat()
      var positionX = child.x + movement.directionX * movement.speed
      var positionY = child.y + movement.directionY * movement.speed
      // collision
      var collided = false
      if (positionX < 0.0f) {
        movement.directionX = -movement.directionX
        positionX = 0.0f
        collided = true
      } else if (positionX > positionMaxX) {
        movement.directionX = -movement.directionX
        positionX = positionMaxX
        collided = true
      }
      if (positionY < 0.0f) {
        movement.directionY = -movement.directionY
        positionY = 0.0f
        collided = true
      } else if (positionY > positionMaxY) {
        movement.directionY = -movement.directionY
        positionY = positionMaxY
        collided = true
      }
      //
      if (collided) {
        val angle = MathUtils.getRandomValue(movementAngle, randomGenerator)
        movement.speed = MathUtils.getRandomValue(movementSpeed, randomGenerator)
        movement.directionX = cos(angle)
        movement.directionY = sin(angle)
      }
      //
      child.x = positionX
      child.y = positionY
    }
    //
    handler.postDelayed(this, UPDATE_TIME_MS)
  }
}
