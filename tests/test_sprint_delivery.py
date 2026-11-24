import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from project_cnn_mammo.protocols import run_sprint
from project_cnn_mammo.training_run import development_manifest


class SprintDeliveryTests(unittest.TestCase):
    def test_all_notebooks_have_compilable_cells(self):
        root = Path(__file__).resolve().parents[1] / 'sprints'
        files = list(root.glob('sprint-*/*/*.ipynb'))
        self.assertEqual(len(files), 15)
        self.assertEqual({(p.parent.parent.name, p.parent.name) for p in files},
                         {(f'sprint-{n:02}', platform) for n in range(7, 12)
                          for platform in ('colab', 'kaggle', 'modal')})
        for path in files:
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type'] == 'code':
                    compile(cell['source'], str(path), 'exec')

    def test_final_test_split_is_refused_before_opening_images(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'manifest.csv').write_text('sample_key,patient_key,path,source,label,split\n'
                                               's,p,never-open.png,fixture,0,test\n')
            with self.assertRaisesRegex(ValueError, 'teste final'):
                development_manifest(root)

    def test_gate_writes_validation_coverage(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sprint10_val.csv').write_text('sample_key,split,p_cnn,s_anom,unet_conf\n'
                                                   'a,val,0.1,0.9,0.9\nb,val,0.8,0.2,0.9\n')
            run_sprint(10, root, root / 'out', smoke=False)
            summary = json.loads((root / 'out' / 'summary.json').read_text())
            self.assertEqual(summary['automatic_coverage'], 0.5)
            self.assertEqual(summary['routes']['RESCUE_ANOMALY'], 1)
