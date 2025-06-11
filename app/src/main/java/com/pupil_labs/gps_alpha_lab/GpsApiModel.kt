package com.pupil_labs.gps_alpha_lab

data class GpsApiModel(
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
    val altitude: Double,
    val accuracy: Float,
    val speed: Float,
    val bearing: Float
)

data class Gps(
    val timestamp: Long,
    val latitude: Double,
    val longitude: Double,
)

interface GpsApi {
    fun startGpsRecording(): Boolean
    fun stopGpsRecording(): Boolean
}