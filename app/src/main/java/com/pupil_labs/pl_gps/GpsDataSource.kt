package com.pupil_labs.pl_gps

class GpsDataSource(
    private val gpsApi: GpsApi
) {
    fun startGpsRecording() = gpsApi.startGpsRecording()
    fun stopGpsRecording() = gpsApi.stopGpsRecording()
}