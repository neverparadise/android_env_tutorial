package com.google.game.rl;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

// adb shell am broadcast -a com.google.game.rl.intent.action.RESET -n <ANDROID_APP_PKG>/com.google.game.rl.RLTaskReceiver
public class RLTaskReceiver extends BroadcastReceiver {

  public static final String ACTION_RESET = "com.google.game.rl.intent.action.RESET";

  @Override
  public void onReceive(Context context, Intent intent) {
    if (context == null) {
      return;
    }
    String action = intent == null ? null : intent.getAction();
    if (ACTION_RESET.equals(action)) {
      RLTask.get().reset();
    }
  }
}
