# Copyright (c) 2026, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""LabBridge: the physics provider backed by the real CPM lab over ROS 2.

This is the only module in sigmarl.lab that imports rclpy / cpm_lab_lab_msgs,
and it is imported lazily (only when physics_backend == "lab"). It owns an
rclpy node on a daemon-thread executor and exposes the synchronous
``BridgeBase`` interface that ``WorldLab`` calls on the control thread:
``apply_actions`` publishes one SPEED_CURVATURE command per vehicle,
``wait_for_state_list`` blocks for the next post-command VehicleStateList, and
``respawn`` is a blocking wrapper around the /lab/respawn_vehicles service.

Runtime: source ROS + the lab workspace, then run with the sigmarl venv python
(ABI-compatible, Python 3.10). The bridge talks the same /planner/* topics the
old cpm_lab_sigmarl_control node used, so the lab-side bridge/ControlCenter
treat it identically.
"""

import atexit
import threading
import time

import torch
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor

from cpm_lab_lab_msgs.msg import (
    PathPoint,
    SystemTrigger,
    VehicleCommand,
    VehicleStateList,
)
from cpm_lab_lab_msgs.srv import RespawnVehicles
from geometry_msgs.msg import Pose2D

from sigmarl.lab.bridge_base import BridgeBase, LabStopped, LabStateTimeout


class LabBridge(BridgeBase):
    def __init__(
        self,
        vehicle_ids,
        period_ms: int,
        world_x_dim: float,
        world_y_dim: float,
        device: str = "cpu",
        namespace: str = "planner",
        state_timeout_s: float = 30.0,
        respawn_timeout_s: float = 90.0,
        command_valid_after_s: float = 0.005,  # consider it a time buffer for the lab latency
        idle_watchdog_periods: float = 3.0,
    ):
        super().__init__(vehicle_ids, period_ms, world_x_dim, world_y_dim, device)
        self.state_timeout_s = state_timeout_s
        self.respawn_timeout_s = respawn_timeout_s
        self.command_valid_after_s = command_valid_after_s
        self._idle_gap_s = (
            idle_watchdog_periods * self.period_s
            if idle_watchdog_periods and idle_watchdog_periods > 0
            else None
        )

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        self._node = rclpy.create_node("sigmarl_lab_bridge", namespace=namespace)

        # State buffer shared with the executor thread.
        self._cond = threading.Condition()
        self._latest_msg = None
        self._latest_seq = 0
        self._latest_arrival = 0.0
        self._consumed_seq = 0
        self._t_apply = time.monotonic()
        self._stop_flag = False
        self._respawn_in_flight = False
        self._closed = False

        # Plain depth-10 default QoS matches the bridge's hlc-side endpoints.
        self._command_pub = self._node.create_publisher(VehicleCommand, "command", 10)
        self._trigger_pub = self._node.create_publisher(
            SystemTrigger, "system_trigger", 10
        )
        self._node.create_subscription(
            VehicleStateList, "vehicle_states", self._on_vehicle_states, 10
        )
        self._node.create_subscription(
            SystemTrigger, "system_trigger", self._on_system_trigger, 10
        )
        self._respawn_client = self._node.create_client(
            RespawnVehicles, "/lab/respawn_vehicles"
        )

        if self._idle_gap_s is not None:
            self._node.create_timer(self.period_s, self._on_watchdog)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True, name="sigmarl-lab-bridge"
        )
        self._spin_thread.start()

        atexit.register(self.close)
        self.send_ready()

    # --- ROS callbacks (executor thread) -----------------------------------

    def _on_vehicle_states(self, msg: VehicleStateList) -> None:
        with self._cond:
            self._latest_msg = msg
            self._latest_seq += 1
            self._latest_arrival = time.monotonic()
            self._cond.notify_all()

    def _on_system_trigger(self, msg: SystemTrigger) -> None:
        if msg.type == SystemTrigger.READY_REQUEST:
            self.send_ready()
        elif msg.type == SystemTrigger.STOP:
            with self._cond:
                self._stop_flag = True
                self._cond.notify_all()

    def _on_watchdog(self) -> None:
        # Hold the fleet stopped whenever the control thread isn't actively
        # stepping (e.g. during a PPO update between collector iterations).
        if self._respawn_in_flight or self._closed:
            return
        if time.monotonic() - self._t_apply > self._idle_gap_s:
            self._publish_stop()

    # --- handshake ---------------------------------------------------------

    def send_ready(self) -> None:
        for vid in self.vehicle_ids:
            msg = SystemTrigger()
            msg.type = SystemTrigger.READY
            msg.stamp = self._node.get_clock().now().to_msg()
            msg.source_id = f"planner_{vid}"
            self._trigger_pub.publish(msg)

    def wait_for_first_state(self) -> None:
        print(
            "[LabBridge] Waiting for the first VehicleStateList. Deploy the "
            "bridge and simulated vehicles in the ControlCenter, then press "
            "START."
        )
        with self._cond:
            while self._latest_seq == 0 and not self._stop_flag:
                self._cond.wait(timeout=1.0)
            if self._stop_flag:
                raise LabStopped("Lab sent STOP before the first state arrived.")
            self._consumed_seq = self._latest_seq
            self._t_apply = time.monotonic()
        print("[LabBridge] Lab is running.")

    # --- BridgeBase interface (control thread) -----------------------------

    def apply_actions(self, target_actions) -> None:
        now = self._node.get_clock().now()
        valid_after = (now + Duration(seconds=self.command_valid_after_s)).to_msg()
        stamp = now.to_msg()
        for a_i, (speed, curvature) in enumerate(
            self.commands_from_targets(target_actions)
        ):
            cmd = VehicleCommand()
            cmd.vehicle_id = int(self.vehicle_ids[a_i])
            cmd.type = VehicleCommand.SPEED_CURVATURE
            cmd.header.frame_id = "map"
            cmd.header.stamp = stamp
            cmd.valid_after_stamp = valid_after
            cmd.speed = float(speed)
            cmd.curvature = float(curvature)
            self._command_pub.publish(cmd)
        self._t_apply = time.monotonic()

    def apply_trajectories(
        self,
        trajectories: torch.Tensor,
        dt: float,
        initial_target_actions: torch.Tensor = None,
    ) -> None:
        """Publish one TRAJECTORY command per vehicle from SigmaRL-frame trajectories."""
        now = self._node.get_clock().now()
        valid_after = (now + Duration(seconds=self.command_valid_after_s)).to_msg()
        stamp = now.to_msg()
        lab_trajs = self.trajectories_to_lab(trajectories)
        for a_i, points in enumerate(lab_trajs):
            cmd = VehicleCommand()
            cmd.vehicle_id = int(self.vehicle_ids[a_i])
            cmd.type = VehicleCommand.TRAJECTORY
            cmd.header.frame_id = "map"
            cmd.header.stamp = stamp
            cmd.valid_after_stamp = valid_after
            for step_i, pt in enumerate(points):
                path_pt = PathPoint()
                path_pt.time = (now + Duration(seconds=step_i * dt)).to_msg()
                path_pt.pose.position.x = pt["x"]
                path_pt.pose.position.y = pt["y"]
                path_pt.pose.position.z = 0.0
                qx, qy, qz, qw = self.quaternion_from_yaw(pt["yaw"])
                path_pt.pose.orientation.x = qx
                path_pt.pose.orientation.y = qy
                path_pt.pose.orientation.z = qz
                path_pt.pose.orientation.w = qw
                path_pt.speed = float(pt["speed"])
                path_pt.distance = float(pt["distance"])
                cmd.path.append(path_pt)
            self._command_pub.publish(cmd)
        self._t_apply = time.monotonic()

    def wait_for_state_list(self, timeout: float = None) -> dict:
        # start_time = time.perf_counter()
        timeout = self.state_timeout_s if timeout is None else timeout
        deadline = time.monotonic() + timeout
        # Only accept a message that arrived after the command became valid,
        # so the snapshot reflects the new command (one bridge tick of
        # latency, matching the old callback-driven node). Compare local
        # monotonic times only — never the bridge's lab-clock stamps.
        min_arrival = self._t_apply + self.command_valid_after_s
        with self._cond:
            while True:
                if self._stop_flag:
                    raise LabStopped("Lab sent STOP during the step.")
                fresh = (
                    self._latest_seq > self._consumed_seq
                    and self._latest_arrival > min_arrival
                )
                #! the reward + all other computations take around 40ms which is extremely long for real time applications
                #! to make sure uncomment the time logic in the helper training before and after the step and maybe_reset in the SIGMARL testing mode
                # if fresh and time.perf_counter() - start_time >= self.period_s - self.command_valid_after_s: # to make sure we wait at least one period before returning the state. Keep it commented out
                if fresh:
                    skipped = self._latest_seq - self._consumed_seq - 1
                    self._consumed_seq = self._latest_seq
                    msg = self._latest_msg
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LabStateTimeout(
                        f"No fresh VehicleStateList within {timeout:.1f}s."
                    )
                self._cond.wait(timeout=remaining)

        if skipped > 0:
            self._node.get_logger().warn(
                f"Skipped {skipped} bridge tick(s): the control step is "
                "slower than the lab period."
            )
        return self._parse_state_list(msg)

    def _parse_state_list(self, msg: VehicleStateList) -> dict:
        by_id = {int(s.vehicle_id): s for s in msg.state_list}
        xs, ys, yaws, speeds, steers = [], [], [], [], []
        for vid in self.vehicle_ids:
            s = by_id.get(vid)
            if s is None:
                # The bridge drops timed-out vehicles right before flipping to
                # EMERGENCY_STOP; treat a missing vehicle as a stop.
                raise LabStopped(
                    f"Vehicle {vid} missing from VehicleStateList (lab timeout?)."
                )
            xs.append(s.pose.position.x)
            ys.append(s.pose.position.y)
            yaws.append(
                self.yaw_from_quaternion(
                    s.pose.orientation.x,
                    s.pose.orientation.y,
                    s.pose.orientation.z,
                    s.pose.orientation.w,
                )
            )
            speeds.append(s.twist.linear.x)
            steers.append(s.steering_servo * self.max_steering)

        return self.states_from_lab(
            torch.tensor(xs, dtype=torch.float32, device=self.device),
            torch.tensor(ys, dtype=torch.float32, device=self.device),
            torch.tensor(yaws, dtype=torch.float32, device=self.device),
            torch.tensor(speeds, dtype=torch.float32, device=self.device),
            torch.tensor(steers, dtype=torch.float32, device=self.device),
        )

    def respawn(self, vehicle_ids, spawn_states_sigmarl) -> None:
        self._respawn_in_flight = True
        try:
            # Stop first: once the bridge starts pausing for the respawn it
            # drops commands, so this must go out before the service call.
            self._publish_stop()
            self._publish_stop()  # send twice to make sure the lab receives it

            poses_sigmarl = spawn_states_sigmarl[:, :3].to(self.device)
            pos_lab, yaw_lab = self.sigmarl_to_lab_pose(
                poses_sigmarl[:, :2], poses_sigmarl[:, 2]
            )
            request = RespawnVehicles.Request()
            request.vehicle_ids = [int(v) for v in vehicle_ids]
            request.poses = [
                Pose2D(
                    x=float(pos_lab[k, 0]),
                    y=float(pos_lab[k, 1]),
                    theta=float(yaw_lab[k]),
                )
                for k in range(len(vehicle_ids))
            ]

            if not self._respawn_client.wait_for_service(timeout_sec=5.0):
                raise LabStopped("/lab/respawn_vehicles service unavailable.")

            done = threading.Event()
            holder = {}

            def _on_response(future):
                holder["future"] = future
                done.set()

            future = self._respawn_client.call_async(request)
            future.add_done_callback(_on_response)

            if not done.wait(timeout=self.respawn_timeout_s):
                raise LabStateTimeout(
                    f"Respawn did not complete within {self.respawn_timeout_s:.0f}s."
                )
            response = holder["future"].result()
            if response is None or not response.success:
                msg = "" if response is None else response.message
                raise LabStopped(f"Respawn rejected by the lab: {msg}")

            self.send_ready()
            # Discard any pre-pause ticks: the next wait must take a snapshot
            # that arrived after the respawn finished.
            with self._cond:
                self._consumed_seq = self._latest_seq
                self._t_apply = time.monotonic()
        finally:
            self._respawn_in_flight = False

    def manual_reset(self, vehicle_ids, reason: str = "") -> dict:
        """Stop the fleet, wait for the operator, return a fresh snapshot.

        Deliberately does *not* set ``_respawn_in_flight``: leaving it False is
        what lets the idle watchdog keep republishing stop every period while
        the human works, so the fleet is actively held at zero speed for the
        whole prompt rather than coasting on its last command.
        """
        self._publish_stop()
        self._publish_stop()  # twice, mirroring respawn()

        self._prompt_manual_reset(vehicle_ids, reason)

        # input() cannot be interrupted by the executor thread, so a lab STOP
        # raised while we were waiting is only observable now.
        with self._cond:
            if self._stop_flag:
                raise LabStopped("Lab sent STOP during the manual reset.")

        self.send_ready()
        # Discard every tick that arrived while the operator was working: the
        # next wait must return a snapshot taken after the placement. The
        # state_timeout_s budget starts here, not before the prompt, so an
        # arbitrarily long placement cannot trip LabStateTimeout.
        with self._cond:
            self._consumed_seq = self._latest_seq
            self._t_apply = time.monotonic()
        return self.wait_for_state_list()

    def stop_vehicles(self) -> None:
        self._publish_stop()

    def _publish_stop(self) -> None:
        now = self._node.get_clock().now()
        stamp = now.to_msg()
        for vid in self.vehicle_ids:
            cmd = VehicleCommand()
            cmd.vehicle_id = int(vid)
            cmd.type = VehicleCommand.SPEED_CURVATURE
            cmd.header.frame_id = "map"
            cmd.header.stamp = stamp
            cmd.valid_after_stamp = stamp
            cmd.speed = 0.0
            cmd.curvature = 0.0
            self._command_pub.publish(cmd)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._publish_stop()
        except Exception:
            pass
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
        except Exception:
            pass
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
