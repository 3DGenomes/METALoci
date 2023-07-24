import sys
from argparse import ArgumentParser, HelpFormatter, RawDescriptionHelpFormatter
from importlib.metadata import version

from metaloci.tools import figure, layout, ml, prep


def main(arguments) -> None: 

    DESCRIPTION = "METALoci: spatially auto-correlated signals in 3D genomes.\n"

    if len(arguments) > 1:

        subcommand = arguments[1]

        if subcommand == "version" or subcommand == "--version":

            print("METALoci v" + version("metaloci"))
            return
        
    parser = ArgumentParser()

    parser.formatter_class=lambda prog: HelpFormatter(prog, width=120,
                                                      max_help_position=60)
    
    subparser = parser.add_subparsers(title="Available programs")

    args_pp = {}

    # prep
    args_pp["prep"] = subparser.add_parser("prep",
                                          description=prep.DESCRIPTION,
                                          help=prep.DESCRIPTION,
                                          add_help=False,
                                          formatter_class=RawDescriptionHelpFormatter)
    args_pp["prep"].set_defaults(func=prep.run)
    prep.populate_args(args_pp["prep"])

    # layout
    args_pp["layout"] = subparser.add_parser("layout",
                                          description=layout.DESCRIPTION,
                                          help=layout.DESCRIPTION,
                                          add_help=False,
                                          formatter_class=RawDescriptionHelpFormatter)
    args_pp["layout"].set_defaults(func=layout.run)
    layout.populate_args(args_pp["layout"])

    # ml
    args_pp["lm"] = subparser.add_parser("lm",
                                          description=ml.DESCRIPTION,
                                          help=ml.DESCRIPTION,
                                          add_help=False,
                                          formatter_class=RawDescriptionHelpFormatter)
    args_pp["lm"].set_defaults(func=ml.run)
    ml.populate_args(args_pp["lm"])

    # figure
    args_pp["figure"] = subparser.add_parser("figure",
                                          description=figure.DESCRIPTION,
                                          help=figure.DESCRIPTION,
                                          add_help=False,
                                          formatter_class=RawDescriptionHelpFormatter)
    args_pp["figure"].set_defaults(func=figure.run)
    figure.populate_args(args_pp["figure"])

    if len(arguments) == 1:

        print(DESCRIPTION)
        parser.print_help()
        return

    if len(arguments) == 2:

        try:

            args_pp[arguments[1]].print_help()
            return
        
        except KeyError:

            pass

    args = parser.parse_args(arguments[1:])

    args.func(args)

sys.exit(main(sys.argv))