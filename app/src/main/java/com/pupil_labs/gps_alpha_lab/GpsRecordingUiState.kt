package com.pupil_labs.gps_alpha_lab

data class GpsRecordingUiState(
    val isRecording: Boolean = false,
    val numSamples: Int = 0,
    val dataSaved: Boolean = false,
    val statusMessage: String = "",
    val savedMessage: String = "",
    val buttonText: String = "Start recording"
)
