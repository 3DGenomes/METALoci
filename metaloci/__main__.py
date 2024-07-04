import sys
from argparse import (
    ArgumentParser,
    HelpFormatter,
    RawDescriptionHelpFormatter,
    RawTextHelpFormatter,
)
from importlib.metadata import version

from metaloci.tests import test_tools
from metaloci.tools import figure, layout, ml, prep
from metaloci.utility_scripts import bts, gene_selector, sniffer


def create_parser():
    """
    Create the argument parser for METALoci.
    """
    parser = ArgumentParser(prog="metaloci")
    parser.formatter_class = lambda prog: HelpFormatter(prog, width=120, max_help_position=60)
    subparser = parser.add_subparsers(title="Available programs")
    args_pp = {}

    # Populate subcommands
    args_pp["prep"] = subparser.add_parser("prep", 
                                           description=prep.DESCRIPTION, 
                                           help=prep.HELP, 
                                           add_help=False, 
                                           formatter_class=RawDescriptionHelpFormatter
                                           )
    args_pp["prep"].set_defaults(func=prep.run)
    prep.populate_args(args_pp["prep"])

    args_pp["layout"] = subparser.add_parser("layout", 
                                             description=layout.DESCRIPTION, 
                                             help=layout.HELP, 
                                             add_help=False, 
                                             formatter_class=RawDescriptionHelpFormatter
                                             )
    args_pp["layout"].set_defaults(func=layout.run)
    layout.populate_args(args_pp["layout"])

    args_pp["lm"] = subparser.add_parser("lm", 
                                         description=ml.DESCRIPTION, 
                                         help=ml.HELP, 
                                         add_help=False, 
                                         formatter_class=RawDescriptionHelpFormatter
                                         )
    args_pp["lm"].set_defaults(func=ml.run)
    ml.populate_args(args_pp["lm"])

    args_pp["figure"] = subparser.add_parser("figure", 
                                             description=figure.DESCRIPTION, 
                                             help=figure.HELP, 
                                             add_help=False, 
                                             formatter_class=RawTextHelpFormatter
                                             )
    args_pp["figure"].set_defaults(func=figure.run)
    figure.populate_args(args_pp["figure"])

    args_pp["sniffer"] = subparser.add_parser("sniffer", 
                                              description=sniffer.DESCRIPTION, 
                                              help=sniffer.HELP, 
                                              add_help=False, 
                                              formatter_class=RawDescriptionHelpFormatter
                                              )
    args_pp["sniffer"].set_defaults(func=sniffer.run)
    sniffer.populate_args(args_pp["sniffer"])

    args_pp["bts"] = subparser.add_parser("bts", 
                                          description=bts.DESCRIPTION, 
                                          help=bts.HELP, 
                                          add_help=False, 
                                          formatter_class=RawDescriptionHelpFormatter
                                          )
    args_pp["bts"].set_defaults(func=bts.run)
    bts.populate_args(args_pp["bts"])

    args_pp["gene_selector"] = subparser.add_parser("gene_selector", 
                                                    description=gene_selector.DESCRIPTION, 
                                                    help=gene_selector.HELP, 
                                                    add_help=False, 
                                                    formatter_class=RawDescriptionHelpFormatter
                                                    )
    args_pp["gene_selector"].set_defaults(func=gene_selector.run)
    gene_selector.populate_args(args_pp["gene_selector"])

    return parser

def main(arguments=None):
    """
    Main function to run METALoci CLI.
    """
    if arguments is None:
        arguments = sys.argv

    # Check if running under Sphinx
    if "sphinx-build" in arguments[0]:
        # Return without parsing any arguments
        return

    DESCRIPTION = "METALoci: spatially auto-correlated signals in 3D genomes.\n"

    if len(arguments) > 1:

        subcommand = arguments[1]

        if subcommand in ["version", "--version"]:

            print("METALoci v" + version("metaloci"))

            return

    parser = create_parser()

    if len(arguments) == 1:

        print(DESCRIPTION)

        parser.print_help()

        return

    if len(arguments) == 2 and arguments[1] != "test":

        try:

            parser.parse_args([arguments[1], '--help'])

            return
        
        except KeyError:

            pass

    args = parser.parse_args(arguments[1:])
    args.func(args)

if __name__ == "__main__":

    main()
