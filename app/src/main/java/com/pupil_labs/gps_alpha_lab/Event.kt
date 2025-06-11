package com.pupil_labs.gps_alpha_lab

data class Event(private var name: String = "gps_event") {
    fun toJson() = """{"name": "${name}"}"""
    fun setName(newName: String) {
        name = newName
    }
}