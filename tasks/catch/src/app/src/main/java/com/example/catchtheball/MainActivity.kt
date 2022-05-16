package com.example.catchtheball

import android.content.Intent
import android.os.Bundle
import android.view.Window
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.isVisible
import androidx.lifecycle.Observer
import com.example.catchtheball.databinding.ActivityMainBinding
import com.google.game.rl.RLTask
import kotlinx.android.synthetic.main.activity_main.livesTv
import kotlinx.android.synthetic.main.activity_main.scoreTv

class MainActivity : AppCompatActivity() {

    private lateinit var viewBinding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        RLTask.get().onCreateActivity(this)
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.apply {
            setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        viewBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        viewBinding.surfaceView.score.observe(this, Observer { newScore ->
            scoreTv.apply {
                isVisible = true
                text = getString(R.string.score_and_score, newScore)
            }
        })
        viewBinding.surfaceView.lives.observe(this, Observer { newLives ->
            livesTv.apply {
                isVisible = true
                text = getString(R.string.lives_and_lives, newLives)
            }
        })
        viewBinding.surfaceView.gameOver.observe(this, Observer { newGameOver ->
            viewBinding.gameOverTv.apply {
                isVisible = newGameOver
            }
        })
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        RLTask.get().onNewIntentActivity(intent)
    }

    override fun onResume() {
        super.onResume()
        viewBinding.surfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        viewBinding.surfaceView.onPause()
    }
}