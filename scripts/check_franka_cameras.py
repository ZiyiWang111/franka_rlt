#!/usr/bin/env python3
"""Open both FR3 RealSense streams and prove that they deliver color frames."""

from __future__ import annotations

import argparse
import sys

import pyrealsense2 as rs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist-serial")
    parser.add_argument("--front-serial")
    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--front-width", type=int, default=1920)
    parser.add_argument("--front-height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.wrist_serial and not args.front_serial:
        print("ERROR: provide at least one camera serial", file=sys.stderr)
        return 2
    if args.wrist_serial and args.front_serial and args.wrist_serial == args.front_serial:
        print("ERROR: wrist and front camera serials must be different", file=sys.stderr)
        return 2
    dimensions = (
        args.wrist_width,
        args.wrist_height,
        args.front_width,
        args.front_height,
    )
    if args.frames < 1 or args.timeout_ms < 1 or any(value < 1 for value in dimensions):
        print("ERROR: frame count, timeout and camera dimensions must be positive", file=sys.stderr)
        return 2

    detected = {
        device.get_info(rs.camera_info.serial_number)
        for device in rs.context().query_devices()
    }
    missing = [
        f"{name}={serial}"
        for name, serial in (("wrist", args.wrist_serial), ("front", args.front_serial))
        if serial
        if serial not in detected
    ]
    if missing:
        print(
            "ERROR: RealSense camera(s) not detected: " + ", ".join(missing),
            file=sys.stderr,
        )
        print("Detected serials: " + (", ".join(sorted(detected)) or "none"), file=sys.stderr)
        return 1

    streams = tuple(
        (name, serial, width, height)
        for name, serial, width, height in (
            ("wrist", args.wrist_serial, args.wrist_width, args.wrist_height),
            ("front", args.front_serial, args.front_width, args.front_height),
        )
        if serial
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
                processing = (
                    "native, no resize/crop"
                    if (actual_width, actual_height) == (640, 480)
                    else "full-frame resize, no crop"
                )
                print(
                    f"camera OK: front native={actual_width}x{actual_height} frame={frame_number} "
                    f"-> recorder output=640x480 ({processing})"
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
