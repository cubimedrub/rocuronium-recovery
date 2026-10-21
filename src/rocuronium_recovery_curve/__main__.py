import argparse

from rocuronium_recovery_curve import calc, review


def cli() -> argparse.Namespace:
    """
    Parses command line arguments, one subcommand per step.
    """

    parser = argparse.ArgumentParser(
        description="Rocuronium recovery curve analysis.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    calc_parser = subparsers.add_parser(
        "calc",
        help="Calculate recovery curves and summaries from patient Excel files.",
    )
    calc.add_arguments(calc_parser)
    calc_parser.set_defaults(run=calc.run)

    review_parser = subparsers.add_parser(
        "review",
        help="Interactively adjust recovery interval start and end of calculated results.",
    )
    review.add_arguments(review_parser)
    review_parser.set_defaults(run=review.run)

    return parser.parse_args()


def main():
    """
    Software entrypoint
    """

    cli_args = cli()
    cli_args.run(cli_args)


if __name__ == "__main__":
    main()
