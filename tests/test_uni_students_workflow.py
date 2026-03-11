import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'comps' / 'uni_students' / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from preprocess_utils import load_and_preprocess_data  # noqa: E402
from run_extended_analysis import bootstrap_confidence_interval, generate_extended_eda_report, parse_args  # noqa: E402


def _build_train_frame():
    return pd.DataFrame(
        [
            {
                'id': 1,
                'outcome': 'Withdrawn',
                'code_module': 'EEE',
                'gender': 'M',
                'region': 'North',
                'highest_education': 'A Level or Equivalent',
                'imd_band': '20-30%',
                'age_band': '0-35',
                'disability': 'N',
                'num_assessments': 0,
                'clicks_week1': 2,
                'clicks_week2': 2,
                'clicks_week3': 1,
                'clicks_week4': 0,
                'num_active_days': 2,
                'first_access_day': -5,
                'last_access_day': 3,
                'mean_score': None,
                'max_score': None,
                'min_score': None,
                'mean_days_late': None,
                'mean_score_TMA': None,
                'mean_score_CMA': None,
            },
            {
                'id': 2,
                'outcome': 'Fail',
                'code_module': 'BBB',
                'gender': 'F',
                'region': 'North',
                'highest_education': 'Lower Than A Level',
                'imd_band': '10-20',
                'age_band': '35-55',
                'disability': 'N',
                'num_assessments': 0,
                'clicks_week1': 4,
                'clicks_week2': 1,
                'clicks_week3': 0,
                'clicks_week4': 0,
                'num_active_days': 3,
                'first_access_day': -3,
                'last_access_day': 5,
                'mean_score': None,
                'max_score': None,
                'min_score': None,
                'mean_days_late': None,
                'mean_score_TMA': None,
                'mean_score_CMA': None,
            },
            {
                'id': 3,
                'outcome': 'Pass',
                'code_module': 'AAA',
                'gender': 'F',
                'region': 'East',
                'highest_education': 'HE Qualification',
                'imd_band': '30-40%',
                'age_band': '0-35',
                'disability': 'N',
                'num_assessments': 2,
                'clicks_week1': 10,
                'clicks_week2': 12,
                'clicks_week3': 13,
                'clicks_week4': 14,
                'num_active_days': 12,
                'first_access_day': -12,
                'last_access_day': 27,
                'mean_score': 76,
                'max_score': 90,
                'min_score': 62,
                'mean_days_late': 0,
                'mean_score_TMA': 80,
                'mean_score_CMA': 72,
            },
            {
                'id': 4,
                'outcome': 'Pass',
                'code_module': 'AAA',
                'gender': 'M',
                'region': 'East',
                'highest_education': 'HE Qualification',
                'imd_band': '30-40%',
                'age_band': '0-35',
                'disability': 'N',
                'num_assessments': 2,
                'clicks_week1': 11,
                'clicks_week2': 10,
                'clicks_week3': 12,
                'clicks_week4': 13,
                'num_active_days': 11,
                'first_access_day': -10,
                'last_access_day': 28,
                'mean_score': 80,
                'max_score': 94,
                'min_score': 67,
                'mean_days_late': 0,
                'mean_score_TMA': 84,
                'mean_score_CMA': 75,
            },
            {
                'id': 5,
                'outcome': 'Withdrawn',
                'code_module': 'GGG',
                'gender': 'F',
                'region': 'North',
                'highest_education': 'A Level or Equivalent',
                'imd_band': '20-30%',
                'age_band': '0-35',
                'disability': 'Y',
                'num_assessments': 0,
                'clicks_week1': 1,
                'clicks_week2': 1,
                'clicks_week3': 0,
                'clicks_week4': 0,
                'num_active_days': 1,
                'first_access_day': -1,
                'last_access_day': 1,
                'mean_score': None,
                'max_score': None,
                'min_score': None,
                'mean_days_late': None,
                'mean_score_TMA': None,
                'mean_score_CMA': None,
            },
            {
                'id': 6,
                'outcome': 'Fail',
                'code_module': 'BBB',
                'gender': 'M',
                'region': 'East',
                'highest_education': 'Lower Than A Level',
                'imd_band': '10-20',
                'age_band': '35-55',
                'disability': 'N',
                'num_assessments': 1,
                'clicks_week1': 6,
                'clicks_week2': 3,
                'clicks_week3': 1,
                'clicks_week4': 0,
                'num_active_days': 4,
                'first_access_day': -4,
                'last_access_day': 8,
                'mean_score': 38,
                'max_score': 40,
                'min_score': 36,
                'mean_days_late': 5,
                'mean_score_TMA': 38,
                'mean_score_CMA': None,
            },
        ]
    )


def _build_test_frame():
    return pd.DataFrame(
        [
            {
                'id': 10,
                'code_module': 'ZZZ',
                'gender': 'F',
                'region': 'South',
                'highest_education': 'HE Qualification',
                'imd_band': '30-40%',
                'age_band': '0-35',
                'disability': 'N',
                'num_assessments': 1,
                'clicks_week1': 8,
                'clicks_week2': 8,
                'clicks_week3': 7,
                'clicks_week4': 6,
                'num_active_days': 8,
                'first_access_day': -7,
                'last_access_day': 21,
                'mean_score': 65,
                'max_score': 70,
                'min_score': 60,
                'mean_days_late': 1,
                'mean_score_TMA': 65,
                'mean_score_CMA': None,
            }
        ]
    )


class UniStudentsWorkflowTests(unittest.TestCase):
    def test_preprocessing_adds_engineered_features_without_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / 'train.csv'
            test_path = Path(tmp_dir) / 'test.csv'
            _build_train_frame().to_csv(train_path, index=False)
            _build_test_frame().to_csv(test_path, index=False)

            X_train, y_train, X_test, test_ids = load_and_preprocess_data(
                train_path,
                test_path,
                scale_numeric=False,
                add_engineered_features=True,
            )

        self.assertEqual(list(y_train), ['Withdrawn', 'Fail', 'Pass', 'Pass', 'Withdrawn', 'Fail'])
        self.assertEqual(list(test_ids), [10])
        self.assertListEqual(list(X_train.columns), list(X_test.columns))
        self.assertIn('has_no_assessments_available', X_train.columns)
        self.assertIn('behavioral_score_missing', X_train.columns)
        self.assertNotIn('region_South', X_train.columns)
        self.assertEqual(int(X_train.loc[0, 'has_no_assessments_available']), 1)
        self.assertEqual(int(X_train.loc[0, 'behavioral_score_missing']), 0)
        self.assertEqual(int(X_train.loc[1, 'chose_not_to_submit']), 1)
        self.assertEqual(int(X_train.loc[1, 'behavioral_score_missing']), 1)

    def test_extended_eda_report_contains_key_sections(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            train_path = Path(tmp_dir) / 'train.csv'
            output_path = Path(tmp_dir) / 'eda.md'
            _build_train_frame().to_csv(train_path, index=False)

            generate_extended_eda_report(train_path, output_path)
            report = output_path.read_text(encoding='utf-8')

        self.assertIn('# Extended EDA Summary', report)
        self.assertIn('Outcome Mix by Module', report)
        self.assertIn('Assessment Missingness by Module', report)
        self.assertIn('Modeling Implications', report)
        self.assertIn('AAA', report)
        self.assertIn('EEE', report)

    def test_bootstrap_confidence_interval_is_ordered(self):
        low, high = bootstrap_confidence_interval([0.62, 0.64, 0.66, 0.68, 0.70], n_bootstrap=200)
        self.assertLessEqual(low, high)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_cli_defaults_prefer_fast_xgb_run(self):
        original_argv = sys.argv
        try:
            sys.argv = ['run_extended_analysis.py']
            args = parse_args()
        finally:
            sys.argv = original_argv

        self.assertFalse(args.include_mlp)
        self.assertEqual(args.cv_folds, 3)


if __name__ == '__main__':
    unittest.main()
