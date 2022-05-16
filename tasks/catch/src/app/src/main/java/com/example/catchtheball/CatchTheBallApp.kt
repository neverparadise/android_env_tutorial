package com.example.catchtheball

import android.app.Application
import timber.log.Timber
import timber.log.Timber.DebugTree


@Suppress("unused")
class CatchTheBallApp : Application() {

    override fun onCreate() {
        super.onCreate()
        if (BuildConfig.DEBUG) {
            Timber.plant(DebugTree())
        }
    }
}