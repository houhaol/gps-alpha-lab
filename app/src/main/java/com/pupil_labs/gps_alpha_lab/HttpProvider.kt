package com.pupil_labs.gps_alpha_lab

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

class HttpProvider(private val url: String) {
    private val client = OkHttpClient()

    private suspend fun postData(endpoint: String, json: String): String {
        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = json.toRequestBody(mediaType)

        val url = url + endpoint

        Log.d("GPS", "Posting GPS Event")

        return withContext(Dispatchers.IO) {
            val request = Request.Builder().url(url).post(body).build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) throw IOException("Unexpected code $response")
                response.body?.string() ?: ""
            }
        }
    }

    suspend fun sendEvent(event: Event) {
        Log.d("GPS", "Sending GPS event")

        try {
            postData("/api/event", event.toJson())
        } catch (e: Exception) {
            Log.d("GPS", "Error in POST: ${e.message}")
            e.printStackTrace()
        }
    }
}