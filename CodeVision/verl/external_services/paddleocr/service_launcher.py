import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PaddleX OCR serving.")
    parser.add_argument("--pipeline", default="OCR", help="Official pipeline name or local pipeline YAML path.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None, help="Examples: gpu:0, cpu")
    parser.add_argument("--use-hpip", action="store_true", help="Enable PaddleX high-performance inference plugin.")
    parser.add_argument("--hpi-config", default=None, help="Optional HPIP config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = ["paddlex", "--serve", "--pipeline", args.pipeline, "--host", args.host, "--port", str(args.port)]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.use_hpip:
        cmd.append("--use_hpip")
    if args.hpi_config:
        cmd.extend(["--hpi_config", args.hpi_config])
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
