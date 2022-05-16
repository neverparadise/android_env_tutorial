package com.google.game.rl;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Locale;
import java.util.Map;
import java.util.WeakHashMap;

/**
 * Class to handle most RL-task setup <br>
 * <code>
 * adb shell am start -n com.google.example.games.abcd/.MainActivity \
 * --ez "RL_TASK_ENABLED" "true"
 * --ez "RL_JSON_EXTRA" "true"
 * --es "RL_TASK_GAME_CONFIG" "{\"booleanParam\":true\,\"intParam\":0}"
 * </code> <br>
 * <code>
 * adb shell am force-stop com.google.example.games.abcd
 * </code>
 */
@SuppressWarnings({"WeakerAccess", "unused"})
public class RLTask implements IRLTask {

  public static final String EXTRA_ENABLED = "RL_TASK_ENABLED";
  public static final boolean EXTRA_ENABLED_DEFAULT = false; // disabled by default
  public static final String EXTRA_JSON = "RL_JSON_EXTRA";
  public static final boolean EXTRA_JSON_DEFAULT = false; // disabled by default
  public static final String EXTRA_GAME_CONFIG = "RL_TASK_GAME_CONFIG";

  private static final boolean LOG_ENABLED = true;
  // adb logcat -s "AndroidRLTask"
  public static final String LOG_TAG = "AndroidRLTask";

  public static final String LOG_MSG_READY = "ready";
  public static final String LOG_MSG_EPISODE_END = "episode end";
  public static final String LOG_MSG_SCORE_AND_SCORE = "score: %s";
  public static final String LOG_MSG_EXTRA_AND_EXTRA = "extra: %s";
  public static final String LOG_MSG_JSON_EXTRA_AND_EXTRA = "json_extra: %s";

  private static RLTask INSTANCE = null;

  private boolean enabled = EXTRA_ENABLED_DEFAULT;

  private RLGameConfiguration gameConfiguration = new RLGameConfiguration();

  private boolean jsonExtraEnabled = EXTRA_JSON_DEFAULT;

  public static synchronized RLTask get() {
    if (INSTANCE == null) {
      INSTANCE = new RLTask();
    }
    return INSTANCE;
  }

  private RLTask() {
    super();
  }

  @SuppressWarnings("BooleanMethodIsAlwaysInverted")
  public boolean isEnabled() {
    return enabled;
  }

  public boolean get(String key, boolean defaultValue) {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public int get(String key, int defaultValue) {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public long get(String key, long defaultValue) {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public float get(String key, float defaultValue) {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public String get(String key, String defaultValue) {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public float[] get(String key, float[] defaultValue) throws JSONException {
    return this.gameConfiguration.get(key, defaultValue);
  }

  public JSONArray get(String key, JSONArray defaultValue) throws JSONException {
    return this.gameConfiguration.get(key, defaultValue);
  }

  @Override
  public void logEpisodeEnd() {
    if (!isEnabled()) {
      return;
    }
    log(LOG_MSG_EPISODE_END);
  }

  @Override
  public void logScore(Object score) {
    if (!isEnabled()) {
      return;
    }
    log(String.format(Locale.ENGLISH, LOG_MSG_SCORE_AND_SCORE, score));
  }

  @Deprecated
  @Override
  public void logExtra(Object extra) {
    if (!isEnabled()) {
      return;
    }
    final String extraString = extra.toString();
    final int indexOfSpace = extraString.indexOf(" ");
    if (indexOfSpace > 0) {
      logExtra(
        extraString.substring(0, indexOfSpace),
        extraString.substring(indexOfSpace + 1)
      );
    }
  }

  public void logExtras(Iterable<Pair<String, Object>> nameValueList) {
    if (!isEnabled()) {
      return;
    }
    if (this.jsonExtraEnabled) {
      final JSONObject jsonExtras = new JSONObject();
      for (Pair<String, Object> nameValue : nameValueList) {
        try {
          jsonExtras.put(nameValue.first, nameValue.second);
        } catch (JSONException e) {
          Log.w(
            RLTask.class.getSimpleName(),
            "Error while saving extra JSON '" + nameValue.first + "' (" + nameValue.second + ")!",
            e
          );
        }
      }
      log(String.format(Locale.ENGLISH, LOG_MSG_JSON_EXTRA_AND_EXTRA, jsonExtras.toString()));
    } else {
      for (Pair<String, Object> nameValue : nameValueList) {
        log(String.format(Locale.ENGLISH, LOG_MSG_EXTRA_AND_EXTRA, nameValue.first + " " + nameValue.second));
      }
    }
  }

  @Override
  public void logExtra(String name, Object value) {
    if (!isEnabled()) {
      return;
    }
    if (this.jsonExtraEnabled) {
      final JSONObject jsonExtras = new JSONObject();
      try {
        jsonExtras.put(name, value);
      } catch (JSONException e) {
        Log.w(
          RLTask.class.getSimpleName(),
          "Error while saving extra JSON '" + name + "' (" + value + ")!",
          e
        );
      }
      log(String.format(Locale.ENGLISH, LOG_MSG_JSON_EXTRA_AND_EXTRA, jsonExtras.toString()));
    } else {
      log(String.format(Locale.ENGLISH, LOG_MSG_EXTRA_AND_EXTRA, name + " " + value));
    }
  }

  @Override
  public void logReady() {
    if (!isEnabled()) {
      return;
    }
    log(LOG_MSG_READY);
  }

  private void log(String msg) {
    if (!LOG_ENABLED) {
      return;
    }
    Log.i(LOG_TAG, msg);
  }

  public static void logDebug(Object object, String msg) {
    logDebug(object.getClass(), msg);
  }

  public static void logDebug(Class<?> clazz, String msg) {
    Log.d(clazz.getSimpleName(), msg);
  }

  public void nativeLogDebug(String tag, String msg) {
    Log.d(tag, msg);
  }

  public void nativeLogInfo(String tag, String msg) {
    Log.i(tag, msg);
  }

  public void onCreateActivity(Activity activity) {
    processIntent(activity.getIntent());
  }

  public void onCreateActivity(Intent intent) {
    processIntent(intent);
  }

  public void onNewIntentActivity(Intent intent) {
    processIntent(intent);
  }

  private void processIntent(Intent intent) {
    if (intent == null) {
      return;
    }
    Bundle extras = intent.getExtras();
    logDebug(this, "processIntent() > extras: " + extras + ".");
    if (extras == null) {
      return;
    }
    this.enabled = extras.getBoolean(EXTRA_ENABLED, EXTRA_ENABLED_DEFAULT);
    logDebug(this, "processIntent() > this.enabled: " + this.enabled + ".");
    this.jsonExtraEnabled = extras.getBoolean(EXTRA_JSON, EXTRA_JSON_DEFAULT);
    logDebug(this, "processIntent() > this.jsonExtraEnabled: " + this.jsonExtraEnabled + ".");
    if (this.enabled) {
      String jsonString = extras.getString(EXTRA_GAME_CONFIG);
      logDebug(this, "processIntent() > jsonString: " + jsonString + ".");
      this.gameConfiguration = RLGameConfiguration.from(jsonString);
      logDebug(this, "processIntent() > this.gameConfiguration: " + this.gameConfiguration + ".");
    }
  }

  private final WeakHashMap<RLResetListener, Object> resetListeners = new WeakHashMap<RLResetListener, Object>();

  public RLResetListener addResetListener(RLResetListener newResetListener) {
    if (newResetListener == null) {
      return null;
    }
    this.resetListeners.put(newResetListener, null);
    return newResetListener;
  }

  public void removeResetListener(RLResetListener resetListener) {
    if (resetListener == null) {
      return;
    }
    this.resetListeners.remove(resetListener);
  }

  public void reset() {
    if (!isEnabled()) {
      return;
    }
    for (RLResetListener resetListener : resetListeners.keySet()) {
      if (resetListener == null) {
        continue;
      }
      resetListener.onReset();
    }
  }

  public interface RLResetListener {
    void onReset();
  }

  /**
   * Game-specific RL-task configuration
   */
  public static class RLGameConfiguration {

    private final Map<String, Object> config = new HashMap<String, Object>();

    public static RLGameConfiguration from(String jsonString) {
      RLGameConfiguration newGameConfig = new RLGameConfiguration();
      if (jsonString != null && !jsonString.isEmpty()) {
        try {
          JSONObject jsonObject = new JSONObject(jsonString);
          Iterator<String> keysIt = jsonObject.keys();
          while (keysIt.hasNext()) {
            String key = keysIt.next();
            Object value = jsonObject.get(key);
            newGameConfig.put(key, value);
          }
        } catch (JSONException e) {
          Log.w(
            RLGameConfiguration.class.getSimpleName(),
            "Error while parsing JSON '" + jsonString + "'!",
            e);
        }
      }
      return newGameConfig;
    }

    public boolean get(String key, boolean defaultValue) {
      Object value = config.get(key);
      if (value instanceof Boolean) {
        return Boolean.parseBoolean(String.valueOf(value));
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public int get(String key, int defaultValue) {
      Object value = config.get(key);
      if (value instanceof Integer) {
        return Integer.parseInt(String.valueOf(value));
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public long get(String key, long defaultValue) {
      Object value = config.get(key);
      if (value instanceof Long
        || value instanceof Integer) {
        return Long.parseLong(String.valueOf(value));
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public float get(String key, float defaultValue) {
      Object value = config.get(key);
      if (value instanceof Float
        || value instanceof Double
        || value instanceof Integer) {
        return Float.parseFloat(String.valueOf(value));
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public String get(String key, String defaultValue) {
      Object value = config.get(key);
      if (value instanceof String
        || value instanceof Integer
        || value instanceof JSONArray) {
        return String.valueOf(value);
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public float[] get(String key, float[] defaultValue) throws JSONException {
      Object value = config.get(key);
      if (value instanceof float[]) {
        return (float[]) value;
      }
      if (value instanceof JSONArray) {
        JSONArray jsonArray = (JSONArray) value;
        float[] floats = new float[jsonArray.length()];
        for (int a = 0; a < jsonArray.length(); a++) {
          Object aValue = jsonArray.get(a);
          if (aValue instanceof Float
            || aValue instanceof Integer) {
            floats[a] = Float.parseFloat(String.valueOf(aValue));
          } else {
            logValueNotFound(key, value);
            return defaultValue;
          }
        }
        return floats;
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    public JSONArray get(String key, JSONArray defaultValue) throws JSONException {
      Object value = config.get(key);
      if (value instanceof JSONArray) {
        return (JSONArray) value;
      } else if (value instanceof String) {
        return new JSONArray((String) value);
      }
      logValueNotFound(key, value);
      return defaultValue;
    }

    private void put(String key, Object value) {
      this.config.put(key, value);
    }

    private void logValueNotFound(String key, Object value) {
      Log.d(RLGameConfiguration.class.getSimpleName(), "Not able to find config value for key '" + key + "'! (value:" + value + "|class:" + (value == null ? null : value.getClass()) + ")");
    }

    @SuppressWarnings("NullableProblems")
    @Override
    public String toString() {
      return RLGameConfiguration.class.getSimpleName() + "{" +
        "config=" + config +
        '}';
    }
  }
}
