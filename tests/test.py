from pynput import mouse, keyboard

# Flag to stop the listener
stop_flag = False

def on_move(x, y):
    print(f"Mouse at ({x}, {y})")

def on_press(key):
    global stop_flag
    if key == keyboard.Key.esc:
        stop_flag = True
        # Stop listener
        return False

# Start mouse listener
mouse_listener = mouse.Listener(on_move=on_move)
mouse_listener.start()

# Start keyboard listener
with keyboard.Listener(on_press=on_press) as keyboard_listener:
    while not stop_flag:
        pass

mouse_listener.stop()
print("Exited.")
