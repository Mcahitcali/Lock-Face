import ctypes
import keyboard

def lock():
    ctypes.windll.user32.LockWorkStation()

keyboard.add_hotkey('l', lambda: lock())

keyboard.wait('esc')