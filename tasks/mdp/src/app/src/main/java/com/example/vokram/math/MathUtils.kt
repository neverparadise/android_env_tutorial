package com.example.vokram.math

import kotlin.math.roundToInt

@Suppress("unused")
class MathUtils {
  companion object {
    fun getRandomValue(value: RandomValue<Int>, randomGenerator: RandomGenerator): Int {
      val random = (randomGenerator.nextFloat() * 2.0f - 1.0f) * value.coefficient
      return value.mean + (value.range * random).roundToInt()
    }

    fun getRandomValue(value: RandomValue<Float>, randomGenerator: RandomGenerator): Float {
      val random = (randomGenerator.nextFloat() * 2.0f - 1.0f) * value.coefficient
      return value.mean + (value.range * random)
    }

    fun getRandomValue(value: RandomFloat, randomGenerator: RandomGenerator): Float {
      val random = (randomGenerator.nextFloat() * 2.0f - 1.0f) * value.coefficient
      return value.mean + value.range * random
    }
  }
}