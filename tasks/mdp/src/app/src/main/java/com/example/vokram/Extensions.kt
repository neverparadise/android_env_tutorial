package com.example.vokram

import android.os.Bundle
import android.util.Log

fun Bundle?.printExtras(logTag: String) {
  Log.d(logTag, "--------------- extras ---------------")
  if (this != null) {
    for (key in keySet()) {
      val value = get(key)
      Log.d(logTag, "$key ${value?.toString()} (${value?.javaClass?.name})")
    }
  }
  Log.d(logTag, "--------------------------------------")
}

fun String?.splitAndTrimNotEmpty(regex: String): List<String> {
  if (this == null) {
    return emptyList()
  }
  return this
    .split(regex.toRegex())
    .filter { it.isNotEmpty() }
    .map {
      it.trim()
        .trimStart('"')
        .trimEnd('"')
    }
    .toList()
}