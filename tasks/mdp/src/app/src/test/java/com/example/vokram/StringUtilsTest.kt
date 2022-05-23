package com.example.vokram

import org.junit.Assert.assertEquals
import org.junit.Test

class StringUtilsTest {

  @Test
  fun splitAndTrim_Null() {
    val string : String? = null
    val regex = ","

    val result = string.splitAndTrimNotEmpty(regex)

    assertEquals(0, result.size)
  }

  @Test
  fun splitAndTrim_Empty() {
    val string = ""
    val regex = ","

    val result = string.splitAndTrimNotEmpty(regex)

    assertEquals(0, result.size)
  }

  @Test
  fun splitAndTrim_Default() {
    val string = "A,B"
    val regex = ","

    val result = string.splitAndTrimNotEmpty(regex)

    assertEquals(2, result.size)
    assertEquals("A", result[0])
    assertEquals("B", result[1])
  }

  @Test
  fun splitAndTrim_Trim() {
    val string = " A , B "
    val regex = ","

    val result = string.splitAndTrimNotEmpty(regex)

    assertEquals(2, result.size)
    assertEquals("A", result[0])
    assertEquals("B", result[1])
  }

  @Test
  fun splitAndTrim_SkipEmpty() {
    val string = " A,,C"
    val regex = ","

    val result = string.splitAndTrimNotEmpty(regex)

    assertEquals(2, result.size)
    assertEquals("A", result[0])
    assertEquals("C", result[1])
  }
}