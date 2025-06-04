package com.pupil_labs.pl_gps

import android.location.Geocoder
import android.util.Log

class GeoCodingProvider(private val geocoder: Geocoder) {
    fun geocode(gpsDatum: Gps): String? {
        val addresses = geocoder.getFromLocation(
            gpsDatum.latitude,
            gpsDatum.longitude,
            1
        )

        Log.d("GPS", "Requested geocoding")

        if (addresses != null && addresses.isNotEmpty()) {
            return addresses[0].getAddressLine(0)
        }

        return null
    }
}