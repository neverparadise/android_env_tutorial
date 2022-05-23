package com.example.vokram

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.example.vokram.databinding.ActivityMenuBinding


class MenuActivity : AppCompatActivity() {

  companion object {
    private val TAG = MenuActivity::class.java.simpleName
  }

  private lateinit var viewBinding: ActivityMenuBinding

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    viewBinding = ActivityMenuBinding.inflate(layoutInflater)
    setContentView(viewBinding.root)

    // register buttons to fire different MDPs
    viewBinding.mdpDefaultBtn.setOnClickListener { v ->
      Log.d(TAG, "default button clicked")
      startActivity(
        MainActivity.newInstance(
          v.context
        )
      )
    }

    viewBinding.riverSwimBtn.setOnClickListener { v ->
      Log.d(TAG, "river swim button clicked")
      startActivity(
        MainActivity.newInstance(
          context = v.context,
          movementSpeed = floatArrayOf(0f, 0f, 1f),
          totalNumSteps = 100,
          mdp = MdpFactory.riverSwimMdp()
        )
      )
    }

    viewBinding.sixArmsBtn.setOnClickListener { v ->
      Log.d(TAG, "6 arms button clicked")
      startActivity(
        MainActivity.newInstance(
          context = v.context,
          movementSpeed = floatArrayOf(0f, 0f, 1f),
          mdp = MdpFactory.sixArmsMdp()
        )
      )
    }

    viewBinding.floodItBtn.setOnClickListener { v ->
      Log.d(TAG, "FloodIt button clicked")
      startActivity(
        MainActivity.newInstance(
          context = v.context,
          movementSpeed = floatArrayOf(0f, 0f, 1f),
          stickToGrid = true,
          actionResPrefix = "flood_it_ball_1",
          actionWidthInDp = 50,
          actionHeightInDp = 57,
          bgColorString = "#F6F6F6",
          mdp = MdpFactory.floodItMdp()
        )
      )
    }

    viewBinding.versionTv.text = BuildConfig.VERSION_NAME

    viewBinding.languageBtn.setOnClickListener {
      startActivity(Intent(android.provider.Settings.ACTION_LOCALE_SETTINGS))
    }
  }
}
