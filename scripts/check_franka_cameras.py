#!/usr/bin/env python3
"""Open both FR3 RealSense streams and prove that they deliver color frames."""

from __future__ import annotations

import argparse
import sys

import pyrealsense2 as rs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist-serial", required=True)
    parser.add_argument("--front-serial", required=True)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.wrist_serial == args.front_serial:
        print("ERROR: wrist and front camera serials must be different", file=sys.stderr)
        return 2
    if args.frames < 1 or args.timeout_ms < 1:
        print("ERROR: --frames and --timeout-ms must be positive", file=sys.stderr)
        return 2

    detected = {
        device.get_info(rs.camera_info.serial_number)
        for device in rs.context().query_devices()
    }
    missing = [
        f"{name}={serial}"
        for name, serial in (
            ("wrist", args.wrist_serial),
            ("front", args.front_serial),
        )
        if serial not in detected
    ]
    if missing:
        print(
            "ERROR: RealSense camera(s) not detected: " + ", ".join(missing),
            file=sys.stderr,
        )
        print("Detected serials: " + (", ".join(sorted(detected)) or "none"), file=sys.stderr)
        return 1

    streams = (
        ("wrist", args.wrist_serial, 640, 480),
        ("front", args.front_serial, 1920, 1080),
    )
    running: list[tuple[str, rs.pipeline]] = []
    try:
        for name, serial, width, height in streams:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
            pipeline.start(config)
            running.append((name, pipeline))

        last_frames: dict[str, tuple[int, int, int]] = {}
        for _ in range(args.frames):
            for name, pipeline in running:
                frames = pipeline.wait_for_frames(timeout_ms=args.timeout_ms)
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError(f"{name} camera returned a frameset without a color frame")
                last_frames[name] = (color.get_width(), color.get_height(), color.get_frame_number())

        for name, _, width, height in streams:
            actual_width, actual_height, frame_number = last_frames[name]
            if (actual_width, actual_height) != (width, height):
                raise RuntimeError(
                    f"{name} camera returned {actual_width}x{actual_height}, expected {width}x{height}"
                )
            if name == "front":
                print(
                    f"camera OK: front native={actual_width}x{actual_height} frame={frame_number} "
                    "-> recorder output=640x480 (full-frame resize, no crop)"
                )
            else:
                print(
                    f"camera OK: {name} native={actual_width}x{actual_height} "
                    f"frame={frame_number} -> recorder output={actual_width}x{actual_height}"
                )
        return 0
    except Exception as exc:
        print(f"ERROR: camera preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        for name, pipeline in reversed(running):
            try:
                pipeline.stop()
            except Exception as exc:
                print(f"WARN: failed to stop {name} camera: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
