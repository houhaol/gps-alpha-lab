package com.pupil_labs.pl_gps

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.IBinder
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.documentfile.provider.DocumentFile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class GpsRepository(
    private val context: Context,
    private val gpsDataSource: GpsDataSource
) {
    private var isRecording = false

    private val gpsData = mutableListOf<GpsApiModel>()

    private var service: GpsLocalProvider? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? GpsLocalProvider.LocalBinder
            val service = localBinder?.getService()

            service?.gpsDataFlow?.let { flow ->
                CoroutineScope(Dispatchers.IO).launch {
                    flow.collect {
                        Log.d("GPS", "Got a GPS datum")
                        gpsData.add(it)
                    }
                }
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            service = null
        }
    }

    fun bindService() {
        val intent = Intent(context, GpsLocalProvider::class.java)
        context.bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)
    }

    fun unbindService() {
        context.unbindService(serviceConnection)
    }

    private var _userFolder: Uri? = null
    fun setUserFolder(userFolder: Uri?) {
        _userFolder = userFolder
    }

    fun startGpsRecording() = gpsDataSource.startGpsRecording()
    fun stopGpsRecording() {
        Log.d("GPS", "Sending stop intent")

        val stopIntent = Intent(context, GpsLocalProvider::class.java).apply {
            action = GpsLocalProvider.ACTION_STOP_SERVICE
        }
        ContextCompat.startForegroundService(context, stopIntent)
    }

    fun startStopGpsRecording(): Pair<Boolean, String?> {
        Log.d("GPS", "Toggling GPS recording state")

        if (isRecording) {
            stopGpsRecording()
            val path = saveGPSData()
            gpsData.clear()
            isRecording = !isRecording
            return isRecording to path
        } else {
            val intent = Intent(context, GpsLocalProvider::class.java)
            ContextCompat.startForegroundService(context, intent)

            Log.d("GPS", "Sent Foreground service intent")

            isRecording = !isRecording
            return isRecording to null
        }
    }

    fun fetchLatestGpsData(): Gps {
        val latestGpsDatum = gpsData.lastOrNull()
        return Gps(latestGpsDatum?.timestamp!!, latestGpsDatum.latitude, latestGpsDatum.longitude)
    }

    fun fecthAllGpsData(): List<Gps> {
        return gpsData.map { Gps(it.timestamp, it.latitude, it.longitude) }
    }

    fun currentNumSamples() = gpsData.size

    fun saveGPSData(): String? {
        val gpsData = fecthAllGpsData()

        Log.d("GPS", "Preparing to save GPS data")
        Log.d("GPS", "Size of gpsData: ${gpsData.size}")

        if (gpsData.isNotEmpty()) {
            Log.d("GPS", "Saving GPS data")

            try {
                val resolver = context.contentResolver

                val prefs = context.getSharedPreferences("gps_prefs", Context.MODE_PRIVATE)
                val uriString = prefs.getString("gps_folder_uri", null)
                val savedUri = uriString?.let { Uri.parse(it) }
                if (savedUri != null && hasUriPermission(context, savedUri)) {
                    _userFolder = savedUri
                }

                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val fileName = "gps_$timestamp.csv"

                Log.d("GPS", "${savedUri}")
                Log.d("GPS", "${fileName}")

                val gpsFolder = DocumentFile.fromTreeUri(context, _userFolder)
                val file = gpsFolder?.createFile("text/csv", fileName)

                Log.d("GPSWriter", "Writing to: ${file?.uri}")

                val csv = buildString{
                    //                writer.append("timestamp [ns],latitude,longitude,altitude,accuracy,speed,bearing\n")
                    //                    writer.append("${record.timestamp},${record.latitude},${record.longitude},${record.altitude},${record.accuracy},${record.speed},${record.bearing}\n")
                    append("timestamp [ns],latitude,longitude\n")
                    gpsData.forEach { record ->
                        append("${record.timestamp},${record.latitude},${record.longitude}\n")
                    }
                }

                file?.uri?.let { fileUri ->
                    context.contentResolver.openOutputStream(fileUri)?.use { outputStream ->
                        outputStream.write(csv.toByteArray())
                        outputStream.flush()
                        return fileName
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                return null
            }
        }
        return null
    }
}

fun hasUriPermission(context: Context, uri: Uri): Boolean {
    val perms = context.contentResolver.persistedUriPermissions
    return perms.any {
        uri.toString().startsWith(it.uri.toString()) && it.isWritePermission
    }
}