package com.pupil_labs.pl_gps

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.os.Binder
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow

class GpsLocalProvider() : GpsApi, Service() {
    companion object {
        const val NOTIFICATION_ID = 1234
        val ACTION_STOP_SERVICE = "ACTION_STOP_SERVICE"
    }

    private lateinit var fusedLocationClient: FusedLocationProviderClient

    val nowMillis = System.currentTimeMillis()
    val nowElapsedNanos = SystemClock.elapsedRealtimeNanos()
    val offsetNanos = nowMillis * 1_000_000 - nowElapsedNanos

    private var isRecording = false

    private val _gpsDataFlow = MutableSharedFlow<GpsApiModel>(replay = 1)
    val gpsDataFlow: SharedFlow<GpsApiModel> = _gpsDataFlow.asSharedFlow()

    inner class LocalBinder : Binder() {
        fun getService(): GpsLocalProvider = this@GpsLocalProvider
    }

    override fun onBind(intent: Intent): IBinder {
        return LocalBinder()
    }

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.lastLocation?.let { loc: Location ->
                Log.d("GPS", "Received Location Result")
                val locationUtcNanos = loc.elapsedRealtimeNanos + offsetNanos
                val gpsDatum = GpsApiModel(
                    locationUtcNanos,
                    loc.latitude,
                    loc.longitude,
                    loc.altitude,
                    loc.accuracy,
                    loc.speed,
                    loc.bearing
                 )
                _gpsDataFlow.tryEmit(gpsDatum)
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP_SERVICE) {
            Log.d("GPS", "Received stop intent")
            stopGpsRecording()
            stopForeground(true)
            stopSelf()
            return START_NOT_STICKY
        }

        // 1. Start foreground with notification
        startForegroundService()

        // 2. Start requesting location updates
        startGpsRecording()

        // If the system kills the service, do NOT recreate until explicitly started again
        return START_STICKY
    }

    private fun startForegroundService() {
        val channelId = "location_channel"
        val channel = NotificationChannel(
            channelId,
            "GPS Recording",
            NotificationManager.IMPORTANCE_LOW
        )
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)

        val notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Running GPS Service in background")
            .setContentText("GPS data is being saved")
            .setSmallIcon(R.drawable.pl_gps)
            .setOngoing(true)
            .build()

        startForeground(1, notification)
    }

    override fun startGpsRecording(): Boolean {
        Log.d("GPS", "Requesting start of GPS recording")

        // Configure the LocationRequest to update as fast as possible.
        val locationRequest = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            0L)
            .setMinUpdateIntervalMillis(0L)
            .build()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
            == PackageManager.PERMISSION_GRANTED
        ) {
            Log.d("GPS", "Permissions granted, starting GPS recording")

            fusedLocationClient.requestLocationUpdates(
                locationRequest,
                locationCallback,
                Looper.getMainLooper()
            )

            isRecording = true
            return isRecording
        } else {
            isRecording = false
            return isRecording
        }
    }

    override fun stopGpsRecording(): Boolean {
        Log.d("GPS", "Stopping GPS recording")

        fusedLocationClient.removeLocationUpdates(locationCallback)

        isRecording = false
        return isRecording
    }

    override fun onDestroy() {
        stopGpsRecording()
        super.onDestroy()
    }
}