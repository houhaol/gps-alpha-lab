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
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class GpsRepository(
    private val context: Context,
    private val gpsDataSource: GpsDataSource
) {
    private var isRecording = false

    private val gpsData = mutableListOf<GpsApiModel>()

    private var csvFile: DocumentFile? = null
    private var csvWriter: BufferedWriter? = null
    private var writerJob: Job? = null
    private var csvPath: String? = null

    private var service: GpsLocalProvider? = null

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            val localBinder = binder as? GpsLocalProvider.LocalBinder
            val service = localBinder?.getService()

            service?.gpsDataFlow?.let { flow ->
                writerJob = CoroutineScope(Dispatchers.IO).launch {
                    flow
                        .chunked(10)
                        .collect { batch ->
                            Log.d("GPS", "Got a batch of GPS data")
                            batch.forEach {
                                gpsData.add(it)
                            }

                            //                        csvWriter?.write("${it.timestamp},${it.latitude},${it.longitude}\n")
                            //                        csvWriter?.flush()

                            val lines = batch.joinToString("\n") { it ->
                                "${it.timestamp},${it.latitude},${it.longitude}\n"
                            }
                            csvWriter?.write(lines + "\n")
                            csvWriter?.flush()
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

    fun startGpsRecording() {
        bindService()

        val fileparts = openCSVFile()
        csvFile = fileparts.first
        csvPath = fileparts.second

        val outputStream = context.contentResolver.openOutputStream(csvFile?.uri!!, "wa")
        csvWriter = BufferedWriter(OutputStreamWriter(outputStream!!)).apply {
            write("timestamp [ns],latitude,longitude\n")
            flush()
        }

//        gpsDataSource.startGpsRecording()

        val intent = Intent(context, GpsLocalProvider::class.java)
        ContextCompat.startForegroundService(context, intent)

        Log.d("GPS", "Sent Foreground service intent")
    }

    fun stopGpsRecording() {
        Log.d("GPS", "Sending stop intent")

        val stopIntent = Intent(context, GpsLocalProvider::class.java).apply {
            action = GpsLocalProvider.ACTION_STOP_SERVICE
        }
        ContextCompat.startForegroundService(context, stopIntent)

        writerJob?.cancel()
        writerJob = null

        csvWriter?.flush()
        csvWriter?.close()
        csvWriter = null

        unbindService()
    }

    fun startStopGpsRecording(): Pair<Boolean, String?> {
        Log.d("GPS", "Toggling GPS recording state")

        if (isRecording) {
            stopGpsRecording()
//            val csvPath = saveGPSData()
            gpsData.clear()
            isRecording = !isRecording
            return isRecording to csvPath
        } else {
            startGpsRecording()
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

    fun openCSVFile(): Pair<DocumentFile?, String> {
        val prefs = context.getSharedPreferences("gps_prefs", Context.MODE_PRIVATE)
        val uriString = prefs.getString("gps_folder_uri", null)
        val savedUri = uriString?.let { Uri.parse(it) }
        if (savedUri != null && hasUriPermission(context, savedUri)) {
            _userFolder = savedUri
        }

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "gps_$timestamp.csv"

        Log.d("GPS", "${savedUri}")
        Log.d("GPS", fileName)

        val gpsFolder = DocumentFile.fromTreeUri(context, _userFolder!!)
        val file = gpsFolder?.createFile("text/csv", fileName)

        return file to fileName
    }

    fun saveGPSData(): String? {
        val gpsData = fecthAllGpsData()

        Log.d("GPS", "Preparing to save GPS data")
        Log.d("GPS", "Size of gpsData: ${gpsData.size}")

        if (gpsData.isNotEmpty()) {
            Log.d("GPS", "Saving GPS data")

            try {
                val fileparts = openCSVFile()
                val file = fileparts.first
                val fileName = fileparts.second

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