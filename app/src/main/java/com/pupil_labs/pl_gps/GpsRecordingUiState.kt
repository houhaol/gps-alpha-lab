package com.pupil_labs.pl_gps

data class GpsRecordingUiState(
    val isRecording: Boolean = false,
    val numSamples: Int = 0,
    val dataSaved: Boolean = false,
    val statusMessage: String = "",
    val savedMessage: String = "",
    val buttonText: String = "Start recording"
)
