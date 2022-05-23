package com.example.vokram

import android.app.Activity
import android.content.DialogInterface
import androidx.annotation.StringRes
import androidx.appcompat.app.AlertDialog

class DialogUtils {

  companion object {

    fun showOkDialog(
      activity: Activity,
      title: String,
      message: String,
      onClickOkListener: DialogInterface.OnClickListener
    ) {
      AlertDialog.Builder(activity)
        .setTitle(title)
        .setMessage(message)
        .setPositiveButton(activity.getString(R.string.restart), onClickOkListener)
        .setCancelable(false)
        .create()
        .show()
    }
  }

  interface DialogProvider {
    fun showOkDialog(
      title: String,
      message: String,
      onClickOkListener: DialogInterface.OnClickListener
    )

    fun showOkDialog(
      @StringRes titleResId: Int,
      message: String,
      onClickOkListener: DialogInterface.OnClickListener
    )
  }
}