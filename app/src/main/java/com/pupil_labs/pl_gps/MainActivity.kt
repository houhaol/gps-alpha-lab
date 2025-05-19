package com.pupil_labs.pl_gps

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Geocoder
import android.location.Location
import android.os.Environment
import android.os.SystemClock
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.util.Locale

suspend fun postData(url: String, json: String): String {
    return withContext(Dispatchers.IO) {
        val client = OkHttpClient()
        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = json.toRequestBody(mediaType)
        val request = Request.Builder().url(url).post(body).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IOException("Unexpected code $response")
            response.body?.string() ?: ""
        }
    }
}

// Data class for storing GPS readings.
data class GPSData(val timestamp: Long, val latitude: Double, val longitude: Double, val altitude: Double, val accuracy: Float, val speed: Float, val bearing: Float)

// Function to save GPS data to a CSV file in the Documents folder.
fun saveGPSDataToCSV(context: Context, data: List<GPSData>): String? {
    return try {
        // Get the app-specific Documents directory.
        val docsDir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        val fileName = "gps_${System.currentTimeMillis()}.csv"
        val file = File(docsDir, fileName)
        FileWriter(file).use { writer ->
            writer.append("timestamp,latitude,longitude,altitude,accuracy,speed,bearing\n")
            data.forEach { record ->
                writer.append("${record.timestamp},${record.latitude},${record.longitude},${record.altitude},${record.accuracy},${record.speed},${record.bearing}\n")
            }
        }
        file.absolutePath
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: android.os.Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MyApp() }
    }
}

@Composable
fun MyApp() {
    MaterialTheme() {
        GPSRecorderScreen()
    }
}

@Composable
fun HttpPostButton(latestData: GPSData?) {
    val context = LocalContext.current

    val coroutineScope = rememberCoroutineScope()
    Button(onClick = {
        coroutineScope.launch {
            try {
                if (latestData != null) {
                    val geocoder = Geocoder(context, Locale.getDefault())
                    val addresses = geocoder.getFromLocation(latestData.latitude, latestData.longitude, 1)
                    if (addresses != null && addresses.isNotEmpty()) {
                        val addr = addresses[0].getAddressLine(0)
                        postData("http://localhost:8080/api/event", """{"name": "${addr}"}""")
                    } else {
                        postData("http://localhost:8080/api/event", """{"name": "gps_event"}""")
                    }
                } else {
                    postData("http://localhost:8080/api/event", """{"name": "gps_event"}""")
                }
            } catch (e: Exception) {
                println("Error in POST: ${e.message}")
                e.printStackTrace()
            }
        }
    }) {
        Text("Send Location Event")
    }
}

@Composable
fun GPSRecorderScreen() {
    val context = LocalContext.current

    // State for recording toggle and status message.
    var recording by remember { mutableStateOf(false) }
    var message by remember { mutableStateOf("") }
    // State list to hold GPS data records.
    val gpsData = remember { mutableStateListOf<GPSData>() }

    // Get FusedLocationProviderClient.
    val fusedLocationClient = remember {
        LocationServices.getFusedLocationProviderClient(context)
    }

    val nowMillis = System.currentTimeMillis()
    val nowElapsedNanos = SystemClock.elapsedRealtimeNanos()
    val offsetNanos = nowMillis * 1_000_000 - nowElapsedNanos

    // Define a LocationCallback that appends new GPS data.
    val locationCallback = remember {
        object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                result.lastLocation?.let { loc: Location ->
                    val locationUtcNanos = loc.elapsedRealtimeNanos + offsetNanos
                    gpsData.add(GPSData(locationUtcNanos, loc.latitude, loc.longitude, loc.altitude, loc.accuracy, loc.speed, loc.bearing))
                }
            }
        }
    }

    // Register for location updates using a DisposableEffect,
    // with a LocationRequest configured for maximum update rate.
    DisposableEffect(recording) {
        if (recording) {
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.ACCESS_FINE_LOCATION)
                == PackageManager.PERMISSION_GRANTED
            ) {
                // Configure the LocationRequest to update as fast as possible.
                val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 0L)
                    .setMinUpdateIntervalMillis(0L)
                    .build()
                fusedLocationClient.requestLocationUpdates(
                    locationRequest,
                    locationCallback,
                    context.mainLooper
                )
            } else {
                message = "Location permission not granted"
            }
        }
        onDispose {
            fusedLocationClient.removeLocationUpdates(locationCallback)
        }
    }

    // When recording stops, save the data.
    LaunchedEffect(recording) {
        if (!recording && gpsData.isNotEmpty()) {
            val path = saveGPSDataToCSV(context, gpsData.toList())
            message = if (path != null) "Saved to: $path" else "Save failed"
            // Optionally clear the list:
            gpsData.clear()
        }
    }

    // Derive the most recent data, if available.
    val latestData = gpsData.lastOrNull()

    // UI Layout.
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "GPS Recorder", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Recorded samples: ${gpsData.size}")
        Spacer(modifier = Modifier.height(16.dp))
        Text(text = message)
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = {
            // Toggle the recording state.
            recording = !recording
            if (recording) {
                message = "Recording started..."
                gpsData.clear() // clear previous records if desired
            } else {
                message = "Recording stopped. Saving..."
            }
        }) {
            Text(text = if (recording) "Stop Recording" else "Start Recording")
        }
        Spacer(modifier = Modifier.height((24.dp)))
        HttpPostButton(latestData)
    }
}
