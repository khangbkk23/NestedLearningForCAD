"""Stdlib-only structural checks: invoke directly, no project conftest or ML."""
import ast
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

class Structure(unittest.TestCase):
    def test_files_compile(self):
        for name in ('scripts/episode_diagnostic.py','tests/episode_reference.py'):
            compile((ROOT/name).read_text(encoding='utf-8-sig'),name,'exec')

    def test_reference_has_no_production_derivative_dependency(self):
        tree=ast.parse((ROOT/'tests/episode_reference.py').read_text(encoding='utf-8-sig'))
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        for name in ('reference_vjp','reference_forward','native_policy','policy_gradient'):
            node=functions[name]
            imported=[n for n in ast.walk(node) if isinstance(n,(ast.Import,ast.ImportFrom))]
            self.assertEqual(imported,[])
            calls={getattr(n.func,'id',None) for n in ast.walk(node) if isinstance(n,ast.Call)}
            self.assertFalse(calls & {'conditional_vjp','functional_step','forward_primals','_tensor_primals'})

    def test_runner_has_no_test_cohort_observer_or_scorer(self):
        tree=ast.parse((ROOT/'scripts/episode_diagnostic.py').read_text(encoding='utf-8-sig'))
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x.endswith('.scoring') or '.metrics' in x for x in imports))
        names={getattr(n.func,'id',None) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(names & {'observe','prepare','arm_trial'})

    def test_prospective_thresholds_and_direction_count(self):
        text=(ROOT/'tests/episode_reference.py').read_text(encoding='utf-8-sig')
        self.assertIn('manual_seed(31901)',text)
        self.assertIn('range(8)',text)
        self.assertIn('2e-8+2e-4',text)

if __name__=='__main__':unittest.main()
