import argparse
import pickle
import struct
import sys

import cv2


def _read_exact(stream, size: int):
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description="OpenCV teleop preview helper")
    parser.add_argument("--window-name", default="Teleop Camera View")
    parser.add_argument("--window-scale", type=float, default=1.0)
    parser.add_argument("--window-x", type=int, default=0)
    parser.add_argument("--window-y", type=int, default=0)
    args = parser.parse_args(argv)

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    window_initialized = False

    while True:
        header = _read_exact(stdin, 4)
        if header is None:
            break
        payload_size = struct.unpack("!I", header)[0]
        if payload_size == 0:
            break

        payload = _read_exact(stdin, payload_size)
        if payload is None:
            break

        try:
            frame = pickle.loads(payload)
            if args.window_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=args.window_scale,
                    fy=args.window_scale,
                    interpolation=cv2.INTER_AREA,
                )
            if not window_initialized:
                cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(args.window_name, frame.shape[1], frame.shape[0])
                cv2.moveWindow(args.window_name, args.window_x, args.window_y)
                window_initialized = True
            cv2.imshow(args.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            stdout.write(b"ok\n")
            stdout.flush()
        except Exception as exc:  # noqa: BLE001
            stdout.write(f"error:{exc}\n".encode("utf-8", errors="replace"))
            stdout.flush()
            break

    try:
        cv2.destroyWindow(args.window_name)
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
