"""Interactive menu and command-line entry point for iDetectFace."""

import argparse
import csv
import sys
from datetime import date
from itertools import groupby
from pathlib import Path

from idetectface.attendance import AttendanceLog
from idetectface.clock import bangladesh_time


def iso_day(value):
    day = date.fromisoformat(value)
    if day.isoformat() != value:
        raise ValueError("Use YYYY-MM-DD.")
    return day


def report(log, day):
    rows = sorted(log.records(day), key=lambda row: row.day, reverse=True)
    if not rows:
        print(f"No attendance records for {day or 'any date'}.")
    for recorded_day, group in groupby(rows, key=lambda row: row.day):
        records = list(group)
        print(f"{recorded_day}  |  {len(records)} present")
        for row in records:
            print(f"  {row.at}  {row.name}")


def run_camera_action(args, log, name=None):
    try:
        from idetectface import vision
    except (ImportError, SystemExit) as error:
        raise RuntimeError("Camera dependencies are missing. Install requirements.txt.") from error
    try:
        directory = args.data_dir / "known_faces"
        if name is None:
            vision.run(directory, log, args.camera, args.threshold, args.every)
        else:
            vision.enroll(directory, args.camera, name)
    except vision.cv2.error as error:
        raise RuntimeError(
            f"OpenCV failed. Check the camera and GUI-enabled OpenCV installation. {error}"
        ) from error


OPERATION_ERRORS = (OSError, ValueError, RuntimeError, csv.Error)


def menu(args, log):
    while True:
        print(
            "\n1. Start face recognition attendance\n2. Add new face\n3. View attendance records\n4. Exit"
        )
        choice = input("Select 1-4: ").strip()
        try:
            if choice == "1":
                run_camera_action(args, log)
            elif choice == "2":
                run_camera_action(args, log, input("Participant name or ID: "))
            elif choice == "3":
                scope = input("1. Today  2. By date  3. All records: ").strip()
                if scope == "1":
                    report(log, bangladesh_time().date())
                elif scope == "2":
                    report(log, iso_day(input("Date (YYYY-MM-DD): ").strip()))
                elif scope == "3":
                    report(log, None)
                else:
                    print("Choose 1, 2 or 3.")
            elif choice == "4":
                return
            else:
                print("Choose a menu option from 1 to 4.")
        except OPERATION_ERRORS as error:
            print(f"Error: {error}", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description="iDetectFace local attendance register")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="folder containing known_faces/ and attendance.csv (default: current folder)",
    )
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="accept a nearest distance strictly below this value (default: 0.5)",
    )
    parser.add_argument("--every", type=int, default=2, help="process every Nth frame (default: 2)")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="start live attendance")
    commands.add_parser("enroll", help="capture one enrollment image").add_argument("name")
    records = commands.add_parser("records", help="display today's attendance")
    scope = records.add_mutually_exclusive_group()
    scope.add_argument("--date", type=iso_day, metavar="YYYY-MM-DD")
    scope.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.every < 1 or args.camera < 0 or not 0 < args.threshold <= 1:
        parser.error("Require --every >= 1, --camera >= 0 and 0 < --threshold <= 1.")
    try:
        log = AttendanceLog(args.data_dir / "attendance.csv")
        if args.command == "records":
            report(log, None if args.all else args.date or bangladesh_time().date())
        elif args.command == "run":
            run_camera_action(args, log)
        elif args.command == "enroll":
            run_camera_action(args, log, args.name)
        else:
            menu(args, log)
    except OPERATION_ERRORS as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
