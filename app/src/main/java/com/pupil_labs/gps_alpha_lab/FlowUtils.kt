package com.pupil_labs.gps_alpha_lab

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow

fun <T> Flow<T>.chunked(chunkSize: Int): Flow<List<T>> = flow {
    require(chunkSize > 0) { "chunkSize must be greater than 0" }

    val buffer = mutableListOf<T>()

    collect { value ->
        buffer.add(value)
        if (buffer.size >= chunkSize) {
            emit(buffer.toList())
            buffer.clear()
        }
    }

    // Emit remaining items if any
    if (buffer.isNotEmpty()) {
        emit(buffer.toList())
    }
}