from unittest import main, TestCase, TestLoader, TextTestRunner, TestSuite
from subprocess import Popen, PIPE, call
import pandas as pd
from metaloci import mlo
import pickle
import shutil
import os
import numpy as np
import pathlib


class TestProcess(TestCase):
    def test_prep_single(self):
        print("Testing metaloci prep:")

        call(
            "metaloci prep -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -d tests/data/signal/DM_27me3-merged-sub137930236_chr19.bed -r 10000 -s tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
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
        call(
            "metaloci prep -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -d tests/data/signal/DM_27me3-merged-sub137930236_chr19_header.bed -r 10000 -s tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
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
        call(
            "metaloci prep -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -d tests/data/signal/DM_27me3-merged-sub137930236_chr19.bed tests/data/signal/DM_Arid1a-sub17717303_15052021_chr19.bed  -r 10000 -s tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
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
        call(
            "metaloci prep -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -d tests/data/signal/DM_27me3-merged-sub137930236_chr19_header.bed tests/data/signal/DM_Arid1a-sub17717303_15052021_chr19_header.bed  -r 10000 -s tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
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
        call(
            "metaloci prep -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -d tests/data/signal/multi_chr19.bed  -r 10000 -s tests/data/mm10_chrom_sizes.txt > /dev/null 2>&1",
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
        print("\nTesting metaloci layout:")

        if os.path.exists("test_wd/chr19/chr19_27320000_31340000_200.mlo"):
            os.remove("test_wd/chr19/chr19_27320000_31340000_200.mlo")

        call(
            "metaloci layout -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 > /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo", "rb"
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)

            self.assertEqual(round(np.sum(mlobject.kk_distances), 0), 127504.0)

    def test_layout_multicore(self):
        if os.path.exists("test_wd/chr19"):
            shutil.rmtree("test_wd/chr19")

        call(
            "metaloci layout -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -r 10000 -g tests/data/regions_test.txt -m -t 3 -l 9 > /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo", "rb"
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)

            self.assertEqual(round(np.sum(mlobject.kk_distances), 0), 129984.0)

    def test_layout_multicutoff(self):
        if os.path.exists("test_wd/chr19"):
            shutil.rmtree("test_wd/chr19")

        call(
            "metaloci layout -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -o 0.15 0.1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_percentage_0.1500_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_layout_pl(self):
        if os.path.exists("test_wd/chr19"):
            shutil.rmtree("test_wd/chr19")

        call(
            "metaloci layout -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -l 9 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_percentage_0.2000_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_layout_abs(self):
        if os.path.exists("test_wd/chr19"):
            shutil.rmtree("test_wd/chr19")

        call(
            "metaloci layout -w test_wd -c tests/data/hic/ICE_DM_5kb_eef0283c05_chr19.mcool -r 10000 -g chr19:27320000-31340000_200 -a -o 1.2 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/KK/chr19_27320000_31340000_200_absolute_1.2000_KK.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_lm_one_signal(self):
        print("\nTesting metaloci lm:")

        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f > /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_pval(self):
        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f -v 0.06> /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_agg(self):
        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -f -a tests/data/aggregate.txt> /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_multisignal(self):
        call(
            "metaloci lm -w test_wd -g chr19:27320000-31340000_200 -s tests/data/signals_test.txt -f > /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_lm_multisignal_multiprocess(self):
        call(
            "metaloci lm -w test_wd -g tests/data/regions_test.txt -s tests/data/signals_test.txt -f -m > /dev/null 2>&1",
            shell=True,
        )

        with open(
            f"test_wd/chr19/chr19_27320000_31340000_200.mlo",
            "rb",
        ) as mlobject_handler:
            mlobject = pickle.load(mlobject_handler)
            self.assertIsNotNone(mlobject.lmi_info)

    def test_figure(self):
        print("\nTesting metaloci figure:")
        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))

    def test_figure_metalocis_only_highlight(self):
        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -m > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        
    def test_figure_alt_quadrants(self):
        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled  -q 1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))
    
    def test_figure_metalocis_only_alt_quadrants(self):
        call(
            "metaloci figure -w test_wd -g chr19:27320000-31340000_200 -s DM_27me3-merged-sub137930236.IgScaled -m -q 1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))
        
    def test_figure_multiple_regions(self):
        call(
            "metaloci figure -w test_wd -g tests/data/regions_test.txt -s DM_27me3-merged-sub137930236.IgScaled > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))
    
    def test_figure_alt_pval(self):
        call(
            "metaloci figure -w test_wd -g tests/data/regions_test.txt -s DM_27me3-merged-sub137930236.IgScaled -v 0.1 > /dev/null 2>&1",
            shell=True,
        )

        path = pathlib.Path(
            "test_wd/chr19/plots/DM_27me3-merged-sub137930236.IgScaled/27320000_31340000_200/chr19_27320000_31340000_200_10000_DM_27me3-merged-sub137930236.IgScaled.pdf"
        )
        self.assertEqual((str(path), path.is_file()), (str(path), True))


def suite():

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

    runner = TextTestRunner(failfast=True)
    
    runner.run(suite())
    shutil.rmtree("test_wd")