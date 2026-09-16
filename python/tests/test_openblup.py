"""Smoke tests for the openblup Python package.

Run with ``python -m pytest python/tests`` (or ``python -m unittest``) after
``maturin develop``.
"""

import os
import tempfile
import unittest

import numpy as np

import openblup
from openblup import MixedModel, Pedigree, compute_a_inverse, compute_g_matrix, to_dense

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "examples")

# Mrode (2005) Example 3.1
MRODE_PED = [
    ("1", None, None),
    ("2", None, None),
    ("3", None, None),
    ("4", "1", None),
    ("5", "3", "2"),
    ("6", "1", "2"),
    ("7", "4", "5"),
    ("8", "3", "6"),
]
MRODE_DATA = {
    "animal": ["4", "5", "6", "7", "8"],
    "sex": ["male", "female", "female", "male", "male"],
    "pwg": [4.5, 2.9, 3.9, 3.5, 5.0],
}


def trial_data():
    return {
        "genotype": ["G1", "G2", "G3"] * 3,
        "rep": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "yield": [10.1, 8.2, 6.0, 12.0, 9.9, 8.1, 11.2, 9.1, 7.0],
    }


class TestPackage(unittest.TestCase):
    def test_version(self):
        self.assertIsInstance(openblup.__version__, str)

    def test_simple_model(self):
        model = MixedModel()
        model.set_data(trial_data())
        model.as_factor("rep")
        model.set_response("yield")
        model.add_fixed("mu + rep")
        model.add_random("genotype")
        result = model.fit()

        self.assertTrue(result.converged)
        self.assertEqual(result.n_obs, 9)
        self.assertEqual(result.n_fixed_params, 3)  # mu + 2 rep contrasts
        vc = result.variance_components()
        self.assertGreater(vc["genotype"], 0.0)
        self.assertGreater(vc["residual"], 0.0)

        blups = result.random_effects()["genotype"]
        levels = result.random_effect_levels()["genotype"]
        by_level = dict(zip(levels, blups))
        self.assertGreater(by_level["G1"], by_level["G2"])
        self.assertGreater(by_level["G2"], by_level["G3"])

        rel = result.reliabilities()["genotype"]
        self.assertTrue(np.all((rel >= 0.0) & (rel <= 1.0)))
        self.assertEqual(result.fixed_effects_cov().shape, (3, 3))
        self.assertEqual(len(result.residuals()), 9)
        terms = {t["term"] for t in result.wald_tests()}
        self.assertEqual(terms, {"mu", "rep"})
        self.assertIn("Variance Components", result.summary())
        self.assertGreater(len(result.iteration_history()), 0)

    def test_missing_response_is_dropped(self):
        data = trial_data()
        data["yield"] = np.array(data["yield"], dtype=float)
        data["yield"][0] = np.nan
        model = MixedModel()
        model.set_data(data)
        model.set_response("yield")
        model.add_fixed("mu")
        model.add_random("genotype")
        result = model.fit()
        self.assertEqual(result.n_obs, 8)

    def test_numeric_random_term_is_coerced(self):
        model = MixedModel()
        model.set_data(trial_data())
        model.set_response("yield")
        model.add_random("rep")
        result = model.fit()
        self.assertEqual(len(result.random_effect_levels()["rep"]), 3)

    def test_pedigree_animal_model(self):
        ped = Pedigree.from_triples(MRODE_PED)
        model = MixedModel()
        model.set_data(MRODE_DATA)
        model.set_response("pwg")
        model.add_fixed("sex")
        model.add_random_pedigree("animal", ped)
        model.set_algorithm("em")
        result = model.fit()

        levels = result.random_effect_levels()["animal"]
        self.assertEqual(sorted(levels), [str(i) for i in range(1, 9)])
        fe = {level: est for (_, level, est, _) in result.fixed_effects()}
        self.assertGreater(fe["male"], fe["female"])

    def test_pedigree_with_explicit_levels_matches_add_random_pedigree(self):
        ped = Pedigree.from_triples(MRODE_PED)
        ainv = compute_a_inverse(ped, as_scipy=False)
        ids = ped.animal_ids()
        self.assertEqual(len(ids), 8)

        m1 = MixedModel()
        m1.set_data(MRODE_DATA)
        m1.set_response("pwg")
        m1.add_fixed("sex")
        m1.add_random("animal", ginverse=ainv, levels=ids)
        m1.set_algorithm("em")
        r1 = m1.fit()

        m2 = MixedModel()
        m2.set_data(MRODE_DATA)
        m2.set_response("pwg")
        m2.add_fixed("sex")
        m2.add_random_pedigree("animal", ped)
        m2.set_algorithm("em")
        r2 = m2.fit()

        np.testing.assert_allclose(
            r1.random_effects()["animal"], r2.random_effects()["animal"], rtol=1e-8
        )

    def test_a_inverse_and_inbreeding(self):
        ped = Pedigree.from_triples(
            [("1", None, None), ("2", None, None), ("3", "1", "2"), ("4", "1", "2"), ("5", "3", "4")]
        )
        ainv = compute_a_inverse(ped, as_scipy=False)
        dense = to_dense(ainv)
        self.assertEqual(dense.shape, (5, 5))
        np.testing.assert_allclose(dense, dense.T)
        f = openblup.compute_inbreeding(ped)
        idx = ped.animal_ids().index("5")
        self.assertAlmostEqual(f[idx], 0.25)

        try:
            import scipy.sparse  # noqa: F401
        except ImportError:
            return
        sp = compute_a_inverse(ped)
        self.assertEqual(sp.shape, (5, 5))
        np.testing.assert_allclose(sp.toarray(), dense)

    def test_g_matrix(self):
        markers = np.array(
            [[0, 1, 2, 1, 0], [2, 1, 0, 1, 2], [1, 1, 1, 1, 1], [0, 2, 1, 0, 1]], dtype=float
        )
        g = compute_g_matrix(markers)
        self.assertEqual(g.shape, (4, 4))
        np.testing.assert_allclose(g, g.T)
        # Non-contiguous input (a transposed view) must give the same answer.
        g2 = compute_g_matrix(np.asfortranarray(markers))
        np.testing.assert_allclose(g, g2)

    def test_load_csv(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trial.csv")
            with open(path, "w") as f:
                f.write("genotype,rep,yield\n")
                data = trial_data()
                for g, r, y in zip(data["genotype"], data["rep"], data["yield"]):
                    f.write(f"{g},{r},{y}\n")
                f.write("G1,4,NA\n")
            model = MixedModel()
            model.load_csv(path)
            self.assertEqual(model.n_rows(), 10)
            self.assertEqual(model.columns(), ["genotype", "rep", "yield"])
            model.as_factor("rep")
            model.set_response("yield")
            model.add_fixed("mu + rep")
            model.add_random("genotype")
            result = model.fit()
            self.assertEqual(result.n_obs, 9)

    def test_pandas_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        df = pd.DataFrame(trial_data())
        model = MixedModel()
        model.set_dataframe(df)
        model.set_response("yield")
        model.add_fixed("mu")
        model.add_random("genotype")
        result = model.fit()
        self.assertEqual(result.n_obs, 9)

    def test_variance_parameters_and_diagnostics(self):
        model = MixedModel()
        model.set_data(trial_data())
        model.as_factor("rep")
        model.set_response("yield")
        model.add_fixed("mu + rep")
        model.add_random("genotype")
        result = model.fit()

        params = result.variance_parameters()
        self.assertEqual([p["component"] for p in params], ["genotype", "residual"])
        for p in params:
            self.assertEqual(p["structure"], "Identity")
            self.assertEqual(p["name"], "sigma2")
            self.assertGreater(p["value"], 0.0)
            self.assertGreater(p["se"], 0.0)
            self.assertFalse(p["at_boundary"])
        self.assertAlmostEqual(params[0]["value"], result.variance_components()["genotype"])

        diag = result.residual_diagnostics()
        self.assertIsNotNone(diag)
        for key in ("fitted", "conditional", "marginal", "standardized", "leverage", "cooks_distance"):
            self.assertEqual(len(diag[key]), 9)
        np.testing.assert_allclose(diag["conditional"], result.residuals())
        np.testing.assert_allclose(
            diag["fitted"] + diag["conditional"], np.asarray(trial_data()["yield"])
        )
        self.assertTrue(np.all(diag["leverage"] >= 0.0))

    def test_satterthwaite_ddf(self):
        model = MixedModel()
        model.set_data(trial_data())
        model.as_factor("rep")
        model.set_response("yield")
        model.add_fixed("mu + rep")
        model.add_random("genotype")
        result = model.fit()

        contain = {t["term"]: t for t in result.wald_tests()}
        satt = {t["term"]: t for t in result.wald_tests(ddf="satterthwaite")}
        self.assertEqual(set(satt), {"mu", "rep"})
        for term in satt:
            self.assertEqual(contain[term]["ddf_method"], "containment")
            self.assertEqual(satt[term]["ddf_method"], "satterthwaite")
            self.assertAlmostEqual(satt[term]["f_statistic"], contain[term]["f_statistic"])
            self.assertGreater(satt[term]["den_df"], 0.0)
            self.assertLessEqual(satt[term]["den_df"], contain[term]["den_df"] + 1e-9)
            self.assertTrue(0.0 <= satt[term]["p_value"] <= 1.0)
        kr = {t["term"]: t for t in result.wald_tests(ddf="kenward-roger")}
        for term in kr:
            self.assertEqual(kr[term]["ddf_method"], "kenward-roger")
            self.assertGreater(kr[term]["f_statistic"], 0.0)
            self.assertTrue(1.0 <= kr[term]["den_df"] <= contain[term]["den_df"] + 1e-9)
        with self.assertRaises(ValueError):
            result.wald_tests(ddf="kenward")

    def test_cross_validate(self):
        model = MixedModel()
        model.set_data(trial_data())
        model.as_factor("rep")
        model.set_response("yield")
        model.add_fixed("mu + rep")
        model.add_random("genotype")
        cv = model.cross_validate(n_folds=3, seed=42)
        self.assertEqual(cv["n_folds"], 3)
        self.assertEqual(len(cv["folds"]), 3)
        self.assertEqual(sum(f["n_validation"] for f in cv["folds"]), 9)
        self.assertTrue(-1.0 <= cv["accuracy"] <= 1.0)
        self.assertGreaterEqual(cv["msep"], 0.0)
        self.assertGreaterEqual(cv["mae"], 0.0)
        # Same seed -> same folds -> same result.
        again = model.cross_validate(n_folds=3, seed=42)
        self.assertAlmostEqual(cv["msep"], again["msep"])

    def test_ar1_by_ar1_residual(self):
        model = MixedModel()
        model.load_csv(os.path.join(EXAMPLES, "field_trial.csv"))
        model.as_factor("rep")
        model.set_response("yield")
        model.add_fixed("mu + rep")
        model.add_random("genotype")
        model.set_residual_interaction("row", "col", "ar1", "ar1c")
        result = model.fit()

        self.assertTrue(result.converged)
        self.assertEqual(result.n_obs, 17)  # one missing plot
        self.assertEqual(result.n_variance_params, 4)
        names = [(p["component"], p["name"]) for p in result.variance_parameters()]
        self.assertEqual(
            names,
            [
                ("genotype", "sigma2"),
                ("residual", "row.sigma2"),
                ("residual", "row.rho"),
                ("residual", "col.rho"),
            ],
        )
        by_name = {n: v for (_, n), v in zip(names, [p["value"] for p in result.variance_parameters()])}
        self.assertGreater(by_name["row.sigma2"], 0.0)
        self.assertTrue(-1.0 < by_name["row.rho"] < 1.0)
        self.assertTrue(-1.0 < by_name["col.rho"] < 1.0)
        # Structured residuals have no leverage-based diagnostics.
        self.assertIsNone(result.residual_diagnostics())
        # Satterthwaite df are available for structured models as well.
        satt = result.wald_tests(ddf="satterthwaite")
        self.assertEqual(satt[0]["ddf_method"], "satterthwaite")
        self.assertTrue(all(1.0 <= t["den_df"] <= 15.0 + 1e-9 for t in satt))

    def test_factor_analytic_gxe(self):
        model = MixedModel()
        model.load_csv(os.path.join(EXAMPLES, "met_trial.csv"))
        model.set_response("yield")
        model.add_fixed("mu + env")
        model.add_random_interaction("env", "genotype", outer_structure="fa1")
        result = model.fit()

        self.assertTrue(result.converged)
        self.assertEqual(result.n_obs, 240)
        self.assertEqual(result.n_variance_params, 9)  # 4 loadings + 4 psi + residual
        params = result.variance_parameters()
        loadings = [p for p in params if p["name"].startswith("env.lambda")]
        psis = [p for p in params if p["name"].startswith("env.psi")]
        self.assertEqual(len(loadings), 4)
        self.assertEqual(len(psis), 4)
        self.assertTrue(all(p["at_boundary"] is False for p in loadings))
        self.assertTrue(all(p["value"] >= 0.0 for p in psis))
        blups = result.random_effects()["env:genotype"]
        levels = result.random_effect_levels()["env:genotype"]
        self.assertEqual(len(blups), 4 * 30)
        self.assertTrue(all(":" in lvl for lvl in levels))
        self.assertIn("env.lambda_1_1", result.summary())

    def test_structure_errors(self):
        model = MixedModel()
        model.set_data(trial_data())
        with self.assertRaises(ValueError):
            model.add_random("genotype", structure="nope")
        ped = Pedigree.from_triples(MRODE_PED)
        ainv = compute_a_inverse(ped, as_scipy=False)
        with self.assertRaises(ValueError):
            model.add_random("genotype", ginverse=ainv, levels=ped.animal_ids(), structure="ar1")

    def test_errors_are_value_errors(self):
        model = MixedModel()
        with self.assertRaises(ValueError):
            model.fit()
        model.set_data(trial_data())
        model.set_response("nope")
        with self.assertRaises(ValueError):
            model.fit()


if __name__ == "__main__":
    unittest.main()
