@echo off
echo Installing F^&O Breakout Screener shortcut on your Desktop...
set SCRIPT="%TEMP%\CreateScreenerShortcut.vbs"
echo Set oWS = WScript.CreateObject("WScript.Shell") >> %SCRIPT%
echo sLinkFile = oWS.SpecialFolders("Desktop") ^& "\F^&O Breakout Screener.url" >> %SCRIPT%
echo Set oUrlLink = oWS.CreateShortcut(sLinkFile) >> %SCRIPT%
echo oUrlLink.TargetPath = "https://f-o-breakout-screener-system.streamlit.app" >> %SCRIPT%
echo oUrlLink.Save >> %SCRIPT%
cscript /nologo %SCRIPT%
del %SCRIPT%
echo.
echo [SUCCESS] Shortcut created on your Desktop!
pause
