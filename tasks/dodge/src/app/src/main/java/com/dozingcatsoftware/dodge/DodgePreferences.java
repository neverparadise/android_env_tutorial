package com.dozingcatsoftware.dodge;

import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.preference.CheckBoxPreference;
import android.preference.Preference;
import android.preference.PreferenceActivity;
import android.preference.PreferenceManager;

@SuppressWarnings("deprecation")
public class DodgePreferences extends PreferenceActivity {

    private static final int ACTIVITY_SELECT_IMAGE = 1;

    public static final String USE_BACKGROUND_KEY = "useBackgroundImage";
    public static final String IMAGE_URI_KEY = "backgroundImageURI";
    public static final String FLASHING_COLORS_KEY = "flashingColors";
    public static final String TILT_CONTROL_KEY = "tiltControl";
    public static final String SHOW_FPS_KEY = "showFPS";

    BackgroundImagePreference selectBackgroundPref;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.preferences);

        selectBackgroundPref = (BackgroundImagePreference) findPreference("selectBackgroundImage");
        selectBackgroundPref.setOnPreferenceClickListener(new Preference.OnPreferenceClickListener() {
            public boolean onPreferenceClick(Preference preference) {
                selectBackgroundImage();
                return true;
            }
        });
    }

    /**
     * Starts the Gallery (or other image picker) activity to select an image
     */
    void selectBackgroundImage() {
        startActivityForResult(new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.INTERNAL_CONTENT_URI), ACTIVITY_SELECT_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent intent) {
        super.onActivityResult(requestCode, resultCode, intent);

        switch (requestCode) {
            case ACTIVITY_SELECT_IMAGE:
                if (resultCode == RESULT_OK) {
                    // retrieve selected image URI and make sure image background is enabled
                    Uri imageURI = intent.getData();
                    if (imageURI != null) {
                        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getBaseContext());
                        SharedPreferences.Editor editor = prefs.edit();
                        editor.putString(IMAGE_URI_KEY, imageURI.toString());
                        editor.apply();
                    }

                    CheckBoxPreference useImagePref = (CheckBoxPreference) findPreference(USE_BACKGROUND_KEY);
                    useImagePref.setChecked(true);

                    selectBackgroundPref.updateBackgroundImage();
                }
                break;
        }
    }
}
