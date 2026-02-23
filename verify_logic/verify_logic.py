"""
Citrix Workspace Keep-Alive Script - Enhanced Version

This script prevents Citrix Workspace VDI from disconnecting due to idle timeout
by simulating periodic keyboard activity using the Scroll Lock key.

Features:
- Uses Scroll Lock key (non-intrusive, doesn't affect your work)
- Adds random variation to avoid detection
- More frequent intervals (30 seconds default)
- Fallback to mouse movement if keyboard fails

Usage:
    python citrix_keepalive.py

Press Ctrl+C to stop the script.
"""

import pyautogui
import time
import sys
import random
import ctypes
from datetime import datetime

# Windows Constants for SetThreadExecutionState
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

# Configuration
INTERVAL_MIN = 25   # Minimum seconds between activities
INTERVAL_MAX = 35   # Maximum seconds between activities (adds randomness)
USE_KEYBOARD = True  # Use keyboard (Scroll Lock) - more effective for Citrix
USE_MOUSE = False    # Fallback to mouse if keyboard fails

def prevent_sleep():
    """Use Windows API to prevent system from sleeping or turning off display."""
    try:
        # ES_SYSTEM_REQUIRED: Prevents the system from entering sleep.
        # ES_DISPLAY_REQUIRED: Forces the display to be on (though monitor power buttons still work).
        # ES_CONTINUOUS: Keeps these settings in effect until the script stops.
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        return True
    except Exception as e:
        log_message(f"✗ Failed to set execution state: {e}")
        return False

def reset_sleep():
    """Reset system sleep settings to default."""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except:
        pass

def log_message(message):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def press_scroll_lock():
    """Press Scroll Lock key to simulate activity without affecting work."""
    try:
        # Press and release Scroll Lock (toggles state but doesn't affect typing)
        pyautogui.press('scrolllock')
        log_message("✓ Scroll Lock key pressed (activity simulated)")
        return True
    except Exception as e:
        log_message(f"✗ Error pressing Scroll Lock: {e}")
        return False

def press_shift():
    """Press Shift key as alternative (doesn't type anything)."""
    try:
        pyautogui.press('shift')
        log_message("✓ Shift key pressed (activity simulated)")
        return True
    except Exception as e:
        log_message(f"✗ Error pressing Shift: {e}")
        return False

def move_mouse():
    """Move mouse slightly to simulate activity (fallback method)."""
    try:
        current_x, current_y = pyautogui.position()
        
        # Move in a small pattern to seem more natural
        moves = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        move = random.choice(moves)
        
        pyautogui.moveRel(move[0], move[1], duration=0.2)
        time.sleep(0.1)
        pyautogui.moveRel(-move[0], -move[1], duration=0.2)
        
        log_message(f"✓ Mouse activity at ({current_x}, {current_y})")
        return True
    except Exception as e:
        log_message(f"✗ Error moving mouse: {e}")
        return False

def simulate_activity():
    """Simulate user activity to prevent idle timeout."""
    success = False
    
    if USE_KEYBOARD:
        # Try Scroll Lock first (best option - doesn't interfere)
        success = press_scroll_lock()
        
        # If Scroll Lock fails, try Shift key
        if not success:
            success = press_shift()
    
    # Fallback to mouse if keyboard methods fail
    if not success and USE_MOUSE:
        success = move_mouse()
    
    return success

def main():
    """Main function to keep the Citrix session alive."""
    log_message("="*60)
    log_message("CITRIX KEEP-ALIVE SCRIPT - ENHANCED VERSION")
    log_message("="*60)
    log_message(f"Interval: {INTERVAL_MIN}-{INTERVAL_MAX} seconds (randomized)")
    log_message(f"Method: {'Keyboard (Scroll Lock)' if USE_KEYBOARD else 'Mouse Movement'}")
    
    # Enable system sleep prevention
    if prevent_sleep():
        log_message("✓ System Sleep Prevention: ACTIVE")
    
    log_message("Press Ctrl+C to stop")
    log_message("-"*60)
    
    # Disable fail-safe for uninterrupted operation
    pyautogui.FAILSAFE = False
    
    try:
        iteration = 0
        total_successes = 0
        total_failures = 0
        
        while True:
            iteration += 1
            log_message(f"\n>>> Iteration #{iteration}")
            
            # Simulate activity
            if simulate_activity():
                total_successes += 1
                log_message(f"Status: SUCCESS (Total: {total_successes} successes, {total_failures} failures)")
            else:
                total_failures += 1
                log_message(f"Status: FAILED (Total: {total_successes} successes, {total_failures} failures)")
                log_message("⚠ WARNING: All activity methods failed! Check if script has necessary permissions.")
            
            # Random wait time to avoid detection
            wait_time = random.randint(INTERVAL_MIN, INTERVAL_MAX)
            import datetime
            next_time = datetime.datetime.now() + datetime.timedelta(seconds=wait_time)
            log_message(f"Next activity at: {next_time.strftime('%H:%M:%S')} ({wait_time}s)")
            
            # Wait for the interval
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        log_message("\n" + "="*60)
        log_message("Script stopped by user (Ctrl+C)")
    except Exception as e:
        log_message(f"\n⚠ UNEXPECTED ERROR: {e}")
        import traceback
        log_message(traceback.format_exc())
    finally:
        # Restore normal system sleep behavior
        reset_sleep()
        log_message("✓ System Sleep Settings: RESTORED")
        
        if 'total_successes' in locals():
            log_message(f"Final Stats: {total_successes} successes, {total_failures} failures")
        
        log_message("Citrix Keep-Alive Script Terminated")
        log_message("="*60)
        sys.exit(0)

if __name__ == "__main__":
    # Check if pyautogui is installed
    try:
        import pyautogui
        main()
    except ImportError:
        print("ERROR: pyautogui library is not installed!")
        print("Please install it using: pip install pyautogui")
        sys.exit(1)
    
    # Display startup banner
