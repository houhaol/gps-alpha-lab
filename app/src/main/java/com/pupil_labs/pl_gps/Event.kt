package com.pupil_labs.pl_gps

data class Event(private var name: String = "gps_event") {
    fun toJson() = """{"name": "${name}"}"""
    fun setName(newName: String) {
        name = newName
    }
}