# Citrix Workspace Keep-Alive Script

## Overview
This Python script prevents your Citrix Workspace VDI connection from disconnecting due to idle timeout by automatically simulating mouse activity at regular intervals.

## Features
- 🖱️ Simulates minimal mouse movements (1 pixel) every 60 seconds
- ⏱️ Configurable interval and movement distance
- 📝 Timestamped logging of all activities
- 🛡️ Safe and non-intrusive (minimal mouse movement)
- ⌨️ Easy to stop with Ctrl+C

## Requirements
- Python 3.6 or higher
- `pyautogui` library

## Installation

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install required library**:
   ```bash
   pip install pyautogui
   ```

## Usage

### Basic Usage
Simply run the script:
```bash
python citrix_keepalive.py
```

The script will:
- Start moving the mouse slightly every 60 seconds
- Display timestamped logs showing each activity
- Keep running until you stop it

### Stopping the Script
Press `Ctrl+C` to safely stop the script.

### Running in Background
To run the script minimized or in background:

**Option 1: Using pythonw (Windows)**
```bash
pythonw citrix_keepalive.py
```
This runs without showing a console window. To stop, you'll need to kill the process from Task Manager.

**Option 2: Keep console minimized**
Just minimize the command prompt window after running the script.

## Configuration

You can modify these settings in the script:

```python
MOVE_INTERVAL = 60  # seconds between movements (default: 60)
MOVE_DISTANCE = 1   # pixels to move (default: 1)
```

### Recommended Settings:
- **Conservative**: 60 seconds interval (default)
- **Aggressive**: 30 seconds interval (for stricter timeout policies)
- **Very conservative**: 120 seconds interval

## Advanced: Auto-Start on Login

### Windows Task Scheduler Method:
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: "When I log on"
4. Set action: "Start a program"
5. Program: `pythonw.exe`
6. Arguments: `C:\Users\admin\Desktop\citrix\citrix_keepalive.py`
7. Click Finish

### Startup Folder Method:
1. Press `Win+R` and type `shell:startup`
2. Create a batch file (`start_keepalive.bat`) with:
   ```batch
   @echo off
   pythonw "C:\Users\admin\Desktop\citrix\citrix_keepalive.py"
   ```
3. Place it in the Startup folder

## Troubleshooting

### Issue: Script doesn't prevent disconnection
- **Solution**: Reduce `MOVE_INTERVAL` to 30 seconds
- Some VDI policies might require more frequent activity

### Issue: Mouse movements are disruptive
- **Solution**: The movements are only 1 pixel, which should be imperceptible
- If still noticeable, ensure you're not moving the mouse during the exact moment of movement

### Issue: "pyautogui not found" error
- **Solution**: Run `pip install pyautogui` in your command prompt

### Issue: Can't stop the background process
- **Solution**: Open Task Manager (Ctrl+Shift+Esc), find `python.exe` or `pythonw.exe`, and end the task

## Safety Notes
- ✅ This script only moves the mouse by 1 pixel and immediately back
- ✅ It does not send any keystrokes or click any buttons
- ✅ It does not interfere with your work
- ✅ You maintain full control and can stop it anytime with Ctrl+C

## License
Free to use and modify as needed.
