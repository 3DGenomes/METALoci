"""
Runs the tests for the metaloci commands.
"""

import os
import pathlib
import pickle
import shutil
from subprocess import call
from unittest import TestCase, TestSuite, TextTestRunner

import metaloci
import numpy as np
import pandas as pd


class TestProcess(TestCase):
    """
    Test the metaloci commands.

    Parameters
    ----------
    TestCase : unittest.TestCase
        TestCase class that is used to create new test cases.
    """

    def test_prep_single(self):
        """
        Test the metaloci prep command with a single signal file.
        """

        print("Testing metaloci prep:")
        call(
            f"metaloci prep -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05_"
            f"chr19.mcool -d {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_27me3-merged-sub137930236_"
            f"chr19.bed -r 10000 -s {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/mm10_chrom_sizes.txt "
            "> /dev/null 2>&1",
            shell=True,
        )

        result = round(
            pd.read_csv("test_wd/signal/chr19/chr19_signal.tsv", sep="\t", header=1)
            .iloc[:, 3]
            .sum(),
            0,
        )

        self.assertEqual(result, 5365.0)

    def test_prep_single_header(self):
        """
        Test the metaloci prep command with a single signal file that has a header.
        """

        call(
            f"metaloci prep -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05_"
            f"chr19.mcool -d {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_27me3-merged-sub137930236_"
            f"chr19_header.bed -r 10000 -s {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/mm10_chrom_sizes.txt "
            "> /dev/null 2>&1",
            shell=True,
        )

        result = round(
            pd.read_csv("test_wd/signal/chr19/chr19_signal.tsv", sep="\t", header=1)
            .iloc[:, 3]
            .sum(),
            0,
        )

        self.assertEqual(result, 5365.0)

    def test_prep_multiple(self):
        """
        Test the metaloci prep command with multiple signal files.
        """

        call(
            f"metaloci prep -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05_"
            f"chr19.mcool -d {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_27me3-merged-sub137930236_"
            f"chr19.bed {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_Arid1a-sub17717303_15052021_"
            f"chr19.bed  -r 10000 -s {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/mm10_chrom_sizes.txt "
            "> /dev/null 2>&1",
            shell=True,
        )

        result = {}
        signal = pd.read_csv(
            "test_wd/signal/chr19/chr19_signal.tsv", sep="\t", header=1
        )
        result[1] = round(signal.iloc[:, 3].sum(), 0)
        result[2] = round(signal.iloc[:, 4].sum(), 0)

        self.assertEqual(result[1], 5365.0)
        self.assertEqual(result[2], 5258.0)

    def test_prep_multiple_header(self):
        """
        Test the metaloci prep command with multiple signal files that have headers.
        """

        call(
            f"metaloci prep -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05_"
            f"chr19.mcool -d {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_27me3-merged-sub137930236"
            f"_chr19_header.bed {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/DM_Arid1a-sub17717303_"
            f"15052021_chr19_header.bed  -r 10000 -s {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/"
            "mm10_chrom_sizes.txt > /dev/null 2>&1",
            shell=True,
        )

        result = {}
        signal = pd.read_csv(
            "test_wd/signal/chr19/chr19_signal.tsv", sep="\t", header=1
        )
        result[1] = round(signal.iloc[:, 3].sum(), 0)
        result[2] = round(signal.iloc[:, 4].sum(), 0)

        self.assertEqual(result[1], 5365.0)
        self.assertEqual(result[2], 5258.0)

    def test_prep_bedgraph(self):
        """
        Test the metaloci prep command with a bedgraph signal file (more than one signal per file).
        """

        call(
            f"metaloci prep -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05_"
            f"chr19.mcool -d {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signal/multi_chr19.bed  -r 10000 -s "
            f"{metaloci.__file__.rsplit('/', 1)[0]}/tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
            shell=True,
        )

        result = {}
        signal = pd.read_csv(
            "test_wd/signal/chr19/chr19_signal.tsv", sep="\t", header=1
        )
        result[1] = round(signal.iloc[:, 3].sum(), 0)
        result[2] = round(signal.iloc[:, 4].sum(), 0)

        self.assertEqual(result[1], 5365.0)
        self.assertEqual(result[2], 5258.0)

    def test_layout_singlecore(self):
        """
        Test the metaloci layout command with a single region.
        """

        print("\nTesting metaloci layout:")

        if os.path.exists("test_wd/chr19/chr19_27320000_31340000_200.mlo"):
            os.remove("test_wd/chr19/chr19_27320000_31340000_200.mlo")

        call(
            f"metaloci layout -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05"
            "_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 > /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo", "rb"
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)

            self.assertEqual(round(np.sum(mlobject.kk_distances), 0), 127504.0)

    def test_layout_multicore(self):
        """
        Test the metaloci layout command with multiple regions.
        """

        if os.path.exists("test_wd/chr19"):

            shutil.rmtree("test_wd/chr19")

        call(
            f"metaloci layout -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05"
            f"_chr19.mcool -r 10000 -g {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/regions_test.txt -m -t 3 "
            "-l 9 > /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo", "rb"
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)

            self.assertEqual(round(np.sum(mlobject.kk_distances), 0), 129984.0)

    def test_layout_multicutoff(self):
        """
        Test the metaloci layout command with multiple cutoffs.
        """

        if os.path.exists("test_wd/chr19"):

            shutil.rmtree("test_wd/chr19")

        call(
            f"metaloci layout -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05"
            "_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -o 0.15 0.1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_percentage_0.1500_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_layout_pl(self):
        """
        Test the metaloci layout command with an alternative persistence length.
        """

        if os.path.exists("test_wd/chr19"):

            shutil.rmtree("test_wd/chr19")

        call(
            f"metaloci layout -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05"
            "_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -l 9 -p > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_percentage_0.2000_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_layout_abs(self):
        """
        Test the metaloci layout command with an absolute cutoff value.
        """

        if os.path.exists("test_wd/chr19"):

            shutil.rmtree("test_wd/chr19")

        call(
            f"metaloci layout -w test_wd -c {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/hic/ICE_DM_5kb_eef0283c05"
            "_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -a -o 1.2 -p > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_absolute_1.2000_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_lm_one_signal(self):
        """
        Test the metaloci lm command with a single signal.
        """

        print("\nTesting metaloci lm:")
        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f > "
            "/dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_pval(self):
        """
        Test the metaloci lm command with a single signal and an alternative p-value.
        """

        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f "
            "-v 0.06> /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_agg(self):
        """
        Test the metaloci lm command with a single signal and an aggregate file.
        """

        call(
            f"metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f "
            f"-a {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/aggregate.txt> /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_multisignal(self):
        """
        Test the metaloci lm command with multiple signals.
        """

        call(
            f"metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s "
            f"{metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signals_test.txt -f > /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_multisignal_multiprocess(self):
        """
        Test the metaloci lm command with multiple signals and multiprocessing.
        """
        call(
            f"metaloci lm -w test_wd -g {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/regions_test.txt "
            f"-s {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/signals_test.txt -f -m > /dev/null 2>&1",
            shell=True,
        )

        with open(
            "test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_figure(self):
        """
        Test the metaloci figure command.
        """

        print("\nTesting metaloci figure:")
        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled "
            "> /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_metalocis_only_highlight(self):
        """
        Test the metaloci figure command with the metalocis_only flah for the highlight of the signal plot.
        """

        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -m "
            "> /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_alt_quadrants(self):
        """
        Test the metaloci figure command with aletrnative quadrant to be considered as significant.
        """

        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled "
            "-q 1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_metalocis_only_alt_quadrants(self):
        """
        Test the metaloci figure command with the metalocis_only flah for the highlight of the signal plot and
        aletrnative quadrant to be considered as significant.
        """

        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -m "
            "-q 1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_multiple_regions(self):
        """
        Test the metaloci figure command with multiple regions.
        """

        call(
            f"metaloci figure -w test_wd -g {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/regions_test.txt "
            "-s DM_27me3-merged-sub137930236.IgScaled > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_alt_pval(self):
        """
        Test the metaloci figure command with an alternative p-value.
        """

        call(
            f"metaloci figure -w test_wd -g {metaloci.__file__.rsplit('/', 1)[0]}/tests/data/regions_test.txt "
            "-s DM_27me3-merged-sub137930236.IgScaled -v 0.1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/"
            "chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        shutil.rmtree("test_wd")


def suite_function():
    """
    Define the order of the tests.

    Returns
    -------
    suite : unittest.TestSuite
        Test suite.
    """

    suite = TestSuite()
    suite.addTest(TestProcess("test_prep_single"))
    suite.addTest(TestProcess("test_prep_single_header"))
    suite.addTest(TestProcess("test_prep_multiple"))
    suite.addTest(TestProcess("test_prep_multiple_header"))
    suite.addTest(TestProcess("test_prep_bedgraph"))
    suite.addTest(TestProcess("test_layout_singlecore"))
    suite.addTest(TestProcess("test_layout_multicutoff"))
    suite.addTest(TestProcess("test_layout_pl"))
    suite.addTest(TestProcess("test_layout_abs"))
    suite.addTest(TestProcess("test_layout_multicore"))
    suite.addTest(TestProcess("test_lm_one_signal"))
    suite.addTest(TestProcess("test_lm_pval"))
    suite.addTest(TestProcess("test_lm_agg"))
    suite.addTest(TestProcess("test_lm_multisignal"))
    suite.addTest(TestProcess("test_lm_multisignal_multiprocess"))
    suite.addTest(TestProcess("test_figure"))
    suite.addTest(TestProcess("test_figure_metalocis_only_highlight"))
    suite.addTest(TestProcess("test_figure_alt_quadrants"))
    suite.addTest(TestProcess("test_figure_metalocis_only_alt_quadrants"))
    suite.addTest(TestProcess("test_figure_multiple_regions"))
    suite.addTest(TestProcess("test_figure_alt_pval"))

    return suite


def run(opts):
    """
    Run the test suite.

    Parameters
    ----------
    opts : None
        Placeholder for the command line arguments.
    """

    runner = TextTestRunner(failfast=True)

    runner.run(suite_function())
