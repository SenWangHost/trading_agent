import argparse
import logging
import os

from dotenv import load_dotenv

from scheduler import run_once, start_scheduler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading-agent",
        description="AI-powered portfolio analyzer",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    sub.add_parser(
        "once",
        help="Fetch portfolio, run one analysis cycle, print report, then exit",
    )
    sub.add_parser(
        "schedule",
        help="Run immediately, then every 30 min during market hours and every 60 min after-hours (Mon–Fri ET)",
    )

    args = parser.parse_args()

    if args.command == "once":
        run_once()
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
