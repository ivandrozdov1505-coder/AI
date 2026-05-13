import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import types

# Create mock modules
mock_torch = types.ModuleType("torch")
mock_torch.nn = types.ModuleType("torch.nn")
mock_torch.optim = types.ModuleType("torch.optim")
mock_torch.utils = types.ModuleType("torch.utils")
mock_torch.utils.data = types.ModuleType("torch.utils.data")

# Add some basic things to mock modules
mock_torch.load = MagicMock()
mock_torch.save = MagicMock()
mock_torch.device = MagicMock
mock_torch.nn.Module = MagicMock
mock_torch.optim.Optimizer = MagicMock
mock_torch.utils.data.Dataset = MagicMock
mock_torch.utils.data.DataLoader = MagicMock

# Put them in sys.modules
sys.modules["torch"] = mock_torch
sys.modules["torch.nn"] = mock_torch.nn
sys.modules["torch.optim"] = mock_torch.optim
sys.modules["torch.utils"] = mock_torch.utils
sys.modules["torch.utils.data"] = mock_torch.utils.data

from ai_trainer.core.checkpoint_manager import CheckpointManager

class TestCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.checkpoint_dir = "test_checkpoints"
        self.max_keep = 5

    @patch('pathlib.Path.mkdir')
    def test_init_basic(self, mock_mkdir):
        """Test basic initialization of CheckpointManager"""
        with patch.object(CheckpointManager, '_scan_existing_checkpoints') as mock_scan:
            manager = CheckpointManager(self.checkpoint_dir, self.max_keep)

            self.assertEqual(manager.checkpoint_dir, Path(self.checkpoint_dir))
            self.assertEqual(manager.max_keep, self.max_keep)
            self.assertEqual(manager.checkpoint_history, [])
            self.assertEqual(manager.best_metric, float('inf'))
            self.assertIsNone(manager.best_checkpoint_path)

            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_scan.assert_called_once()

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.glob')
    def test_init_scan_no_files(self, mock_glob, mock_mkdir):
        """Test initialization when no existing checkpoints are found"""
        mock_glob.return_value = []

        manager = CheckpointManager(self.checkpoint_dir, self.max_keep)

        self.assertEqual(manager.checkpoint_history, [])
        mock_glob.assert_called_once_with("checkpoint_*.pt")

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.glob')
    def test_init_scan_with_files(self, mock_glob, mock_mkdir):
        """Test initialization with existing checkpoint files"""
        path1 = MagicMock(spec=Path)
        path1.name = "checkpoint_1.pt"
        path2 = MagicMock(spec=Path)
        path2.name = "checkpoint_best.pt"

        mock_glob.return_value = [path1, path2]

        checkpoint1 = {'epoch': 1, 'step': 100, 'best_metric': 0.5, 'timestamp': 1000.0}
        checkpoint_best = {'epoch': 2, 'step': 200, 'best_metric': 0.1, 'timestamp': 2000.0}

        def mock_load(path, map_location=None):
            if path == path1: return checkpoint1
            if path == path2: return checkpoint_best
            return {}

        mock_torch.load.side_effect = mock_load

        manager = CheckpointManager(self.checkpoint_dir, self.max_keep)

        self.assertEqual(len(manager.checkpoint_history), 2)
        # Verify sorting (by timestamp)
        self.assertEqual(manager.checkpoint_history[0]['timestamp'], 1000.0)
        self.assertEqual(manager.checkpoint_history[1]['timestamp'], 2000.0)

        # Verify best checkpoint update
        self.assertEqual(manager.best_checkpoint_path, path2)
        self.assertEqual(manager.best_metric, 0.1)

    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.glob')
    @patch('ai_trainer.core.checkpoint_manager.logger')
    def test_init_scan_error(self, mock_logger, mock_glob, mock_mkdir):
        """Test initialization when some checkpoints fail to load"""
        path1 = MagicMock(spec=Path)
        path1.name = "checkpoint_corrupt.pt"
        path2 = MagicMock(spec=Path)
        path2.name = "checkpoint_valid.pt"

        mock_glob.return_value = [path1, path2]

        def mock_load(path, map_location=None):
            if path == path1: raise Exception("Load error")
            return {'epoch': 1, 'timestamp': 1000.0}

        mock_torch.load.side_effect = mock_load

        manager = CheckpointManager(self.checkpoint_dir, self.max_keep)

        self.assertEqual(len(manager.checkpoint_history), 1)
        self.assertEqual(manager.checkpoint_history[0]['epoch'], 1)
        mock_logger.warning.assert_called()

if __name__ == '__main__':
    unittest.main()
