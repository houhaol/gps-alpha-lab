package com.pupil_labs.pl_gps

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Geocoder
import android.net.Uri
import android.os.Bundle
import android.os.VibrationEffect
import android.os.VibratorManager
import android.provider.DocumentsContract
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.SideEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val geocoder = Geocoder(this, Locale.getDefault())
        val geoCodingProvider = GeoCodingProvider(geocoder)

        // neon's real-time api can be accessed via localhost when running on same phone
        val httpProvider = HttpProvider("http://localhost:8080")

        val gpsLocalProvider = GpsLocalProvider()
        val gpsLocalDataSource = GpsDataSource(gpsLocalProvider)
        val gpsRepository = GpsRepository(this, gpsLocalDataSource)

        val gpsViewModel = GpsViewModel(
            gpsRepository,
            geoCodingProvider,
            httpProvider
        )
        gpsViewModel.listenGpsNumSamples()

        Log.d("GPS", "Initialized all Providers and ViewModel")

        setContent {
            BlackWhiteTheme() {
                SetupAndStartMainScren(gpsViewModel)
            }
        }
    }
}

@Composable
fun BlackWhiteTheme(content: @Composable () -> Unit) {
    val view = LocalView.current
    val window = (view.context as? Activity)?.window

    SideEffect {
        window?.statusBarColor = Color.Black.toArgb()
        window?.navigationBarColor = Color.Black.toArgb()
        WindowCompat.getInsetsController(window!!, view).apply {
            isAppearanceLightStatusBars = false // Use light text/icons
            isAppearanceLightNavigationBars = false
        }
    }

    MaterialTheme(
        colorScheme = darkColorScheme(
            background = Color.Black,
            surface = Color.Black,
            onBackground = Color.White,
            onSurface = Color.White,
            primary = Color.White
        ),
        content = content
    )
}

fun areNotificationsEnabled(context: Context): Boolean {
    return NotificationManagerCompat.from(context).areNotificationsEnabled()
}

@Composable
fun SetupAndStartMainScren(gpsViewModel: GpsViewModel) {
    val context = LocalContext.current
    val sharedPrefs = context.getSharedPreferences("gps_prefs", Context.MODE_PRIVATE)
    val savedUriString = sharedPrefs.getString("gps_folder_uri", null)
    Log.d("GPS", "Saved data dir: ${savedUriString}")

    Log.d("GPS", "Requesting permissions")

    var notificationGranted by remember {
        mutableStateOf(areNotificationsEnabled(context))
    }
    Log.d("GPS", "Notification permission status: ${notificationGranted}")

    var fineLocationGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    var backgroundLocationGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.ACCESS_BACKGROUND_LOCATION
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    // if it was not granted, we must ask again if it is wanted
    var backgroundLocationWanted by remember { mutableStateOf(backgroundLocationGranted) }

    var folderWanted by remember { mutableStateOf(savedUriString != null) }
    var folderGranted by remember { mutableStateOf(savedUriString != null) }

    var returnedFromFilePicker by remember { mutableStateOf(false) }

    var folderUri by remember { mutableStateOf<Uri?>(
        if (savedUriString != null) {
            Uri.parse(savedUriString)
        } else {
            null
        })
    }

    val notificationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        notificationGranted = granted
    }

    // Request fine location permission launcher
    val fineLocationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        fineLocationGranted = granted
    }

    // Request background location permission launcher
    val backgroundLocationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        Log.d("GPS", "Background location granted: ${backgroundLocationGranted}")
        backgroundLocationGranted = true
    }

    // Folder picker launcher
    val folderPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocumentTree()
    ) { uri ->
        folderUri = uri

        Log.d("GPS", "Folder URI: ${uri}")
        if (uri == null) {
            returnedFromFilePicker = true
            folderWanted = false
        }

        uri?.let {
            context.contentResolver.takePersistableUriPermission(
                it,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or
                        Intent.FLAG_GRANT_WRITE_URI_PERMISSION
            )

            val gpsUri = createGpsSubfolder(context, it)
            gpsViewModel.setUserFolder(gpsUri)

            sharedPrefs.edit()
                .putString("gps_folder_uri", gpsUri.toString())
                .apply()

            folderWanted = true
            folderGranted = true
        }
    }

    LaunchedEffect(Unit) {
        if (!notificationGranted) {
            if (ContextCompat.checkSelfPermission(
                    context, Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    LaunchedEffect(notificationGranted) {
        if (notificationGranted && !fineLocationGranted) {
            fineLocationLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }
    }

    // When fine location granted, inform about background location
    LaunchedEffect(fineLocationGranted) {
        if (notificationGranted && fineLocationGranted && !backgroundLocationWanted) { }
    }

    // after informing about background location, ask if desired
    LaunchedEffect(backgroundLocationWanted) {
        if (notificationGranted && fineLocationGranted && backgroundLocationWanted && !backgroundLocationGranted) {
            backgroundLocationLauncher.launch(Manifest.permission.ACCESS_BACKGROUND_LOCATION)
        }
    }

    // When both permissions granted, launch folder picker
    LaunchedEffect(backgroundLocationGranted) {
        if (notificationGranted && fineLocationGranted && backgroundLocationGranted && !folderWanted) { }
    }

    LaunchedEffect(folderWanted) {
        if (notificationGranted && fineLocationGranted && backgroundLocationGranted && folderWanted && !folderGranted) {
            folderPickerLauncher.launch(null)
        }
    }

    LaunchedEffect(folderGranted) {
        if (notificationGranted && fineLocationGranted && backgroundLocationGranted && folderWanted && folderGranted) { }
    }

    // Effect to re-check permission after returning from settings
    LaunchedEffect(returnedFromFilePicker) {
        if (returnedFromFilePicker) {
            folderWanted = false
            returnedFromFilePicker = false
        }
    }

    Scaffold() { padding ->
        Surface(
            modifier = androidx.compose.ui.Modifier.padding(padding).fillMaxSize(),
            color = MaterialTheme.colorScheme.background
        ) {
            when {
                !notificationGranted -> Text("")
                !fineLocationGranted -> Text("")
                !backgroundLocationWanted -> {
                    // Show background permission dialog
                    BackgroundLocationPermissionDialog(
                        onGoToSettings = {
                            backgroundLocationWanted = true
                        },
                        onDismiss = {
                            backgroundLocationWanted = false
                            Toast.makeText(
                                context,
                                "The app only functions with this permission.",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    )
                }
                !backgroundLocationGranted -> Text("")
                !folderWanted -> {
                    // Show SAF request dialog
                    FolderRequestDialog(
                        onGoToFolderPicker = {
                            folderWanted = true
                        },
                        onDismiss = {
                            folderWanted = false
                            Toast.makeText(
                                context,
                                "The app only functions with this permission.",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    )
                }
                !folderGranted -> Text("")
                else -> {
                    Log.d("GPS", "Permission requests completed; starting main interface")
                    GPSRecorderScreen(gpsViewModel)
                }
            }
        }
    }
}

fun createGpsSubfolder(context: Context, parentUri: Uri): Uri? {
    val docUri = DocumentsContract.buildDocumentUriUsingTree(
        parentUri,
        DocumentsContract.getTreeDocumentId(parentUri)
    )

    var gpsFolderUri = findFolderUri(context, docUri, "GPS")

    if (gpsFolderUri != null) {
        Log.d("GPS", "Folder already exists")
    } else {
        Log.d("GPS", "Creating GPS folder...")
        gpsFolderUri = DocumentsContract.createDocument(
            context.contentResolver,
            docUri,
            DocumentsContract.Document.MIME_TYPE_DIR,
            "GPS"
        )
        Log.d("GPS", "Created GPS folder: $gpsFolderUri")
    }

    return gpsFolderUri
}

fun findFolderUri(context: Context, parentUri: Uri, folderName: String): Uri? {
    val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(
        parentUri,
        DocumentsContract.getTreeDocumentId(parentUri)
    )

    context.contentResolver.query(
        childrenUri,
        arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        ),
        null,
        null,
        null
    )?.use { cursor ->
        while (cursor.moveToNext()) {
            val docId = cursor.getString(0)
            val name = cursor.getString(1)
            val mime = cursor.getString(2)

            if (name == folderName && mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                return DocumentsContract.buildDocumentUriUsingTree(parentUri, docId)
            }
        }
    }

    return null // Not found
}

@Composable
fun BackgroundLocationPermissionDialog(
    onGoToSettings: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Background Location Required", color = Color.White) },
        text = {
            Text(
                "To save GPS data in the background, please enable " +
                    "\"Allow all the time\" in the system settings that will appear." +
                    "Then, you can return here with Android's Back (<) button.",
                color = Color.White
            )
        },
        confirmButton = {
            TextButton(onClick = onGoToSettings) {
                Text("Open Settings", color = Color.White)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = Color.White)
            }
        },
        containerColor = Color.Black
    )
}

@Composable
fun FolderRequestDialog(
    onGoToFolderPicker: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("File Storage Required", color = Color.White) },
        text = {
            Text(
                "We now request permission to save GPS data to " +
                        "the `Documents/GPS` folder. Simply press \"Use This Folder\" " +
                        "when the dialog appears and allow access. " +
                        "This is the easiest way, similar " +
                        "to how it is done for the Neon Companion app.",
                color = Color.White
            )
        },
        confirmButton = {
            TextButton(onClick = onGoToFolderPicker) {
                Text("Open Settings", color = Color.White)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel", color = Color.White)
            }
        },
        containerColor = Color.Black
    )
}

fun vibrate(context: Context, durationMillis: Long = 200) {
    // Android 12+ (API 31+): use VibratorManager
    val vibratorManager =
        context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
    val vibrator = vibratorManager.defaultVibrator
    vibrator.vibrate(
        VibrationEffect.createOneShot(durationMillis, VibrationEffect.DEFAULT_AMPLITUDE)
    )
}

@Composable
fun GPSRecorderScreen(viewModel: GpsViewModel) {
    val context = LocalContext.current
    val gpsRecordingUiState by viewModel.uiState.collectAsState()

    if (gpsRecordingUiState.isRecording && gpsRecordingUiState.numSamples == 1) {
        vibrate(context)
        Thread.sleep(500)
        vibrate(context)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(50.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "GPS Recorder",
            color = MaterialTheme.colorScheme.onBackground,
            style = MaterialTheme.typography.headlineMedium
        )

        Spacer(modifier = Modifier.height(210.dp))

        GpsEventButton(viewModel)

        Spacer(modifier = Modifier.height(10.dp))

        Text(
            text = gpsRecordingUiState.savedMessage,
            color = MaterialTheme.colorScheme.onBackground,
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth().height(85.dp)
        )

        Spacer(modifier = Modifier.height((20.dp)))

        Text(
            text = (
                    if (gpsRecordingUiState.isRecording && gpsRecordingUiState.numSamples == 0) {
                        "Initializing sensor..."
                    } else if (gpsRecordingUiState.isRecording) {
                        "Number of samples: ${gpsRecordingUiState.numSamples}"
                    } else {
                        ""
                    }
                ),
            color = MaterialTheme.colorScheme.onBackground
        )

        Spacer(modifier = Modifier.height((10.dp)))

        GpsRecordingButton(
            gpsRecordingUiState.isRecording,
            gpsRecordingUiState.numSamples > 0,
            onClick = { viewModel.startStopGpsRecording() }
        )
    }
}

@Composable
fun GpsRecordingButton(
    isRecording: Boolean,
    samplesObtained: Boolean,
    onClick: () -> Unit
) {
    val backgroundColor = if (isRecording && samplesObtained) {
        Color.Red
    } else if(isRecording) {
        Color.Gray
    } else {
        Color.White
    }

    Box(
        modifier = Modifier
            .size(82.dp)
            .aspectRatio(1f, matchHeightConstraintsFirst = true)
            .clip(CircleShape)
            .background(backgroundColor)
            .border(3.dp, Color.Gray, CircleShape)
            .clickable(onClick = onClick)
    )
}