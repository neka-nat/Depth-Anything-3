import time

import numpy as np
import zenoh
from pynput import keyboard
from pylekiwi.key_listener import KeyListener
from pylekiwi.nodes import ClientControllerNode
from pylekiwi.models import BaseCommand


class LekiwiController:
    def __init__(self):
        self.key_listener = KeyListener()
        self.controller = ClientControllerNode()
        self.session = zenoh.open(zenoh.Config())
        self.sub_obj_points = self.session.declare_subscriber("obj_points", self._on_obj_points)
        self.sub_current_pos = self.session.declare_subscriber("current_pos", self._on_current_pos)
        self._obj_points = None
        self._current_pos = None
        self._odom_pos = None

    def _on_obj_points(self, sample):
        self._obj_points = np.frombuffer(bytes(sample.payload), dtype=np.float32)
        print(f"obj_points: {self._obj_points}")

    def _on_current_pos(self, sample):
        self._current_pos = np.frombuffer(bytes(sample.payload), dtype=np.float32)
        print(f"current_pos: {self._current_pos}")

    def run(self):
        with keyboard.Listener(
            on_press=self.key_listener.on_key_press,
            on_release=self.key_listener.on_key_release,
        ):
            dt = 0.01
            while True:
                if self.key_listener.current_command is not None:
                    self.controller.send_base_command(command=self.key_listener.current_command)
                if self._odom_pos is None and self._current_pos is not None:
                    self._odom_pos = self._current_pos.copy()
                if self._odom_pos is not None and self._obj_points is not None:
                    diff = -(self._obj_points - self._odom_pos)
                    print(f"diff: {diff}")
                    norm = np.linalg.norm(diff)
                    if norm > 0.2:
                        x_vel = diff[0] / norm * 0.1
                        y_vel = diff[1] / norm * 0.1
                        self.controller.send_base_command(command=BaseCommand(
                           x_vel=x_vel,
                           y_vel=y_vel,
                           theta_deg_vel=0,
                        ))
                        self._odom_pos -= np.array([x_vel, y_vel, 0]) * dt
                    else:
                        self._obj_points = None
                else:
                    self.controller.send_base_command(command=BaseCommand(
                        x_vel=0,
                        y_vel=0,
                        theta_deg_vel=0,
                    ))
                time.sleep(dt)


if __name__ == "__main__":
    controller = LekiwiController()
    controller.run()
