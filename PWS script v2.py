from ultralytics import YOLO
import os
from pathlib import Path
import yaml
from roboflow import Roboflow
import torch
import gc
import logging
import argparse
import platform
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training.log'),
        logging.StreamHandler()
    ]
)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train tumor detection model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str,
                        help='Specific checkpoint to resume from (optional)')
    return parser.parse_args()


class HardwareConfig:
    def __init__(self):
        self.device, self.device_name = self._detect_device()
        self.config = self._get_optimal_config()
        self._log_system_info()

    def _detect_device(self):
        """Detect available hardware and return optimal device."""
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            device = 'mps'
            device_name = 'Apple Silicon'
        else:
            device = 'cpu'
            device_name = 'CPU'
        return device, device_name

    def _get_optimal_config(self):
        """Get optimal training configuration based on hardware."""
        base_config = {
            'device': self.device,
            'imgsz': 640,
            'optimizer': 'AdamW',
            'cos_lr': True,
            'patience': 25,
            'epochs': 100,
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.1,

            'mosaic': 0.3,
            'mixup': 0.0,
            'copy_paste': 0.0,
            'degrees': 10,

            'cache': 'disk',
            'lr0': 0.0005,
            'lrf': 0.0001,
            'warmup_epochs': 6,
            'warmup_momentum': 0.8,
            'weight_decay': 0.03,
        }

        # Hardware-specific configurations
        if self.device == 'cuda':
            cuda_config = {
                'batch': 14,
                'workers': 4,

            }
            base_config.update(cuda_config)

        elif self.device == 'mps':
            mps_config = {
                'batch': 8,
                'workers': 3,
            }
            base_config.update(mps_config)

        else:  # CPU fallback
            cpu_config = {
                'batch': 4,
                'workers': 2,
            }
            base_config.update(cpu_config)

        return base_config

    def _log_system_info(self):
        """Log detailed system information."""
        memory = psutil.virtual_memory()
        logging.info("\n=== System Information ===")
        logging.info(f"Device: {self.device_name}")
        logging.info(f"Platform: {platform.platform()}")
        logging.info(f"Python: {platform.python_version()}")
        logging.info(f"PyTorch: {torch.__version__}")
        logging.info(f"Available Memory: {memory.available / (1024**3):.1f}GB")

        if self.device == 'cuda':
            logging.info(f"CUDA Version: {torch.version.cuda}")
            logging.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")

        logging.info("\n=== Training Configuration ===")
        for key, value in self.config.items():
            logging.info(f"{key}: {value}")
        logging.info("========================\n")


class TumorDetectionTrainer:
    def __init__(self, resume=False, checkpoint=None):
        self.work_dir = Path(os.getcwd())
        self.dataset_dir = self.work_dir / 'datasets' / 'Tumor-Otak-3'
        self.hardware = HardwareConfig()
        self.resume = resume
        self.checkpoint = checkpoint

    def clear_memory(self):
        """Clear memory based on device type."""
        gc.collect()
        if self.hardware.device == 'cuda':
            torch.cuda.empty_cache()
        elif self.hardware.device == 'mps':
            torch.mps.empty_cache()

    def setup_dataset(self):
        """
        Downloads the dataset from Roboflow.
        """
        try:
            if not self.dataset_dir.exists():
                logging.info("Downloading dataset...")
                rf = Roboflow(api_key='mjlzWzF2MLR8VQMK6qq6')
                project = rf.workspace(
                    "gunadarma-university").project("tumor-otak")
                version = project.version(3)
                version.download("yolov8", location=str(self.dataset_dir))
                logging.info("Dataset downloaded successfully")
            return True
        except Exception as e:
            logging.error(f"Dataset setup failed: {e}")
            return False

    def update_yaml_config(self):
        """
        Updates the YAML configuration file with the correct paths.
        """
        try:
            data_yaml_path = self.dataset_dir / "data.yaml"

            with open(data_yaml_path, "r") as f:
                data = yaml.safe_load(f)

            # Update paths with absolute paths
            data.update({
                "train": str(self.dataset_dir / "train" / "images"),
                "val": str(self.dataset_dir / "valid" / "images"),
                "test": str(self.dataset_dir / "test" / "images"),
                "nc": 4,
                "names": ['GLIOMA', 'MENINGIOMA', 'NON GLIOMA', 'PITUITARY']
            })

            # Validate paths
            for key, path in data.items():
                if key in ['train', 'val', 'test']:
                    if not os.path.exists(path):
                        logging.warning(f"Path does not exist: {path}")

            with open(data_yaml_path, "w") as f:
                yaml.safe_dump(data, f)

            logging.info("YAML configuration updated successfully")
            return data_yaml_path
        except Exception as e:
            logging.error(f"YAML configuration failed: {e}")
            return None

    def find_latest_checkpoint(self):
        """
        Finds the latest checkpoint in the runs directory.
        """
        runs_dir = self.work_dir / 'runs' / 'detect'
        if not runs_dir.exists():
            return None

        train_runs = [d for d in runs_dir.iterdir() if d.is_dir()
                      and d.name.startswith('train')]
        if not train_runs:
            return None

        latest_run = max(train_runs, key=lambda x: x.stat().st_mtime)
        weights_dir = latest_run / 'weights'
        if not weights_dir.exists():
            return None

        weights_files = list(weights_dir.glob('*.pt'))
        if not weights_files:
            return None

        return str(max(weights_files, key=lambda x: x.stat().st_mtime))

    def train_model(self, data_yaml_path):
        """
        Trains the model.
        """
        try:
            self.clear_memory()
            model_path = self.checkpoint if self.checkpoint else 'yolov8s.pt'
            model = YOLO(model_path)

            # Get hardware-optimized training args
            training_args = self.hardware.config

            # Add dataset-specific args
            training_args.update({
                'data': str(data_yaml_path),
                'resume': self.resume,
            })

            results = model.train(**training_args)
            return results

        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error("Memory error occurred")
                self.clear_memory()

                # Try to recover with smaller batch size
                training_args['batch'] = training_args['batch'] // 2
                logging.info(
                    f"Retrying with batch size: {training_args['batch']}")

                results = model.train(**training_args)
                return results

            logging.error(f"Training error: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error during training: {e}")
            return None


def main():
    args = parse_args()
    trainer = TumorDetectionTrainer(
        resume=args.resume,
        checkpoint=args.checkpoint
    )

    if not trainer.setup_dataset():
        return

    data_yaml_path = trainer.update_yaml_config()
    if not data_yaml_path:
        return

    results = trainer.train_model(data_yaml_path)

    if results:
        logging.info("Training completed successfully")
    else:
        logging.warning("Training failed or was interrupted")


if __name__ == "__main__":
    main()
