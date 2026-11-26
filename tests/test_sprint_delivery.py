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

    def test_cloud_notebooks_bundle_code_and_zip_only_current_run(self):
        root = Path(__file__).resolve().parents[1] / 'sprints'
        for platform in ('colab', 'kaggle'):
            notebook = root / 'sprint-10' / platform / f'sprint_10_{platform}.ipynb'
            cells = [c['source'] for c in json.loads(notebook.read_text())['cells'] if c['cell_type'] == 'code']
            with TemporaryDirectory() as folder:
                output = Path(folder)
                (output / 'old-results.zip').touch()
                state = {'DATA': output, 'OUTPUT': output, 'ARM': ''}
                exec(cells[1], state)  # bootstrap embutido, sem depender da plataforma
                self.assertTrue((state['PROJECT'] / 'configs' / 'sprints.toml').is_file())
                self.addCleanup(state['CODE_DIR'].cleanup)
                (output / 'sprint10_val.csv').write_text('sample_key,split,p_cnn,s_anom,unet_conf\n'
                                                       'a,val,0.1,0.9,0.9\n')
                from unittest.mock import Mock, patch
                from zipfile import ZipFile
                colab = Mock()
                google = Mock(colab=colab)
                with patch.dict('sys.modules', {'google': google, 'google.colab': colab}):
                    exec(cells[-1], state)
                if platform == 'colab':
                    colab.files.download.assert_called_once_with(str(state['archive']))
                archives = list(output.glob('sprint-10-*.zip'))
                self.assertEqual(len(archives), 1)
                with ZipFile(archives[0]) as archive:
                    self.assertIn('summary.json', archive.namelist())
                    self.assertNotIn('old-results.zip', archive.namelist())
                self.assertFalse(any(path.is_dir() for path in output.iterdir()))

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
