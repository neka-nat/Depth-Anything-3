import time

from pynput import keyboard
from pylekiwi.key_listener import KeyListener
from pylekiwi.nodes import ClientControllerNode


class LekiwiController:
    def __init__(self):
        self.key_listener = KeyListener()
        self.controller = ClientControllerNode()

    def run(self):
        with keyboard.Listener(
            on_press=self.key_listener.on_key_press,
            on_release=self.key_listener.on_key_release,
        ):
            while True:
                if self.key_listener.current_command is not None:
                    self.controller.send_base_command(command=self.key_listener.current_command)
                time.sleep(0.01)


if __name__ == "__main__":
    controller = LekiwiController()
    controller.run()
