package com.example.overseen.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val WcagAaaHighContrastColorScheme = darkColorScheme(
    primary = HighContrastYellow,
    onPrimary = HighContrastBlack,
    secondary = HighContrastWhite,
    onSecondary = HighContrastBlack,
    tertiary = HighContrastYellow,
    onTertiary = HighContrastBlack,
    background = HighContrastBlack,
    onBackground = HighContrastYellow,
    surface = HighContrastBlack,
    onSurface = HighContrastWhite,
    surfaceVariant = HighContrastBlack,
    onSurfaceVariant = HighContrastYellow
)

@Composable
fun OverseenTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = WcagAaaHighContrastColorScheme,
        typography = Typography,
        content = content
    )
}
