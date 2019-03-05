import argparse
import sys


def main():
    run_argument_parser(sys.argv[1:])


def run_argument_parser(args):

    parser = argparse.ArgumentParser(
        description='Graph Minimap',
        prog='graph_minimap',
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100))

    subparsers = parser.add_subparsers(help='Subcommands')
    subparser_map = subparsers.add_parser("map", help="Map reads to graph")
    subparser_map.add_argument("-g", "--graph", help="Numpy graphs")
    subparser_map.add_argument("-i", "--index", help="Numpy minimizer index")
    subparser_map.add_argument("-f", "--fasta", help="Two-line fasta file to map")

    subparser_map.set_defaults(func=map_all())
    subparser_filter.set_defaults(func=run_filter)
    subparser_remove_from_fasta.set_defaults(func=remove_reads_from_fasta)


    if len(args) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)

    if hasattr(args, 'func'):
        args.func(args)
