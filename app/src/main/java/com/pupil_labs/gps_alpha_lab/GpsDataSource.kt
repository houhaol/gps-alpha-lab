package com.pupil_labs.gps_alpha_lab

class GpsDataSource(
    private val gpsApi: GpsApi
) {
    fun startGpsRecording() = gpsApi.startGpsRecording()
    fun stopGpsRecording() = gpsApi.stopGpsRecording()
}