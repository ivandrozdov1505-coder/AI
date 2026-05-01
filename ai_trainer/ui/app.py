"""
Основное Gradio приложение AI Trainer Platform
"""

import gradio as gr
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import threading

from utils.gpu_utils import GPUManager, get_device_info, check_cuda
from utils.logger import setup_logger, get_ui_logs, clear_ui_logs
from utils.config_loader import ConfigLoader
from core.data_manager import DataManager
from core.model_manager import ModelManager
from trainers.supervised_trainer import SupervisedTrainer
from trainers.genetic_trainer import GeneticTrainer

logger = logging.getLogger(__name__)


class AIApp:
    def __init__(self):
        self.config = None
        self.data_manager = None
        self.model_manager = None
        self.trainer = None
        self.gpu_manager = GPUManager()
        self.training_thread = None
        self.is_training = False
        
        setup_logger(name="ai_trainer", level="INFO", log_dir="./logs", console=False, file=True)
    
    def load_config(self, config_path: str) -> str:
        try:
            loader = ConfigLoader()
            if config_path and Path(config_path).exists():
                self.config = loader.load(config_path)
            else:
                self.config = loader.create_default("./configs/default_config.yaml")
            return f"Конфигурация загружена: {len(self.config)} секций"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def load_data(self, paths: str, file_types: str) -> str:
        try:
            if not self.config: self.config = {}
            self.data_manager = DataManager(self.config)
            path_list = [p.strip() for p in paths.split(",") if p.strip()]
            types_list = [t.strip() for t in file_types.split(",") if t.strip()]
            count = self.data_manager.load_files(path_list, types_list)
            stats = self.data_manager.get_statistics()
            return f"Загружено файлов: {count}\nВсего записей: {stats['total_raw']}"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def preprocess_data(self, train_split: float, val_split: float) -> str:
        try:
            if not self.data_manager: return "Сначала загрузите данные"
            self.data_manager.preprocess()
            self.data_manager.split(train_split, val_split, 1.0 - train_split - val_split)
            self.data_manager.build_vocab()
            stats = self.data_manager.get_statistics()
            return f"Предобработка завершена\nTrain: {stats['splits'].get('train', 0)}\nVal: {stats['splits'].get('val', 0)}"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def create_model(self, architecture: str, hidden_size: int, num_layers: int, num_classes: int) -> str:
        try:
            if not self.config: self.config = {}
            device = self.gpu_manager.get_device()
            self.model_manager = ModelManager(self.config, device)
            vocab_size = len(self.data_manager.vocab) if self.data_manager and self.data_manager.vocab else 32000
            params = {"hidden_size": hidden_size, "num_layers": num_layers, "num_classes": num_classes, "vocab_size": vocab_size}
            model = self.model_manager.create_model(architecture, params)
            mem = self.model_manager.estimate_memory(batch_size=32)
            return f"Модель создана: {architecture}\nПамять: ~{mem['model_mb']} MB\nПараметров: {sum(p.numel() for p in model.parameters()):,}"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def start_training(self, epochs: int, batch_size: int, learning_rate: float, training_type: str) -> str:
        if self.is_training: return "Обучение уже запущено"
        try:
            if not self.data_manager or not self.model_manager: return "Подготовьте данные и модель"
            device = self.gpu_manager.get_device()
            
            from torch.utils.data import DataLoader, TensorDataset
            X_train, y_train = torch.randn(100, 512), torch.randint(0, 2, (100,))
            X_val, y_val = torch.randn(20, 512), torch.randint(0, 2, (20,))
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
            
            model = self.model_manager.get_model()
            if not self.config: self.config = {}
            self.config["training"] = {"learning_rate": learning_rate, "optimizer": "adamw"}
            
            if training_type == "genetic":
                self.trainer = GeneticTrainer(model, device, self.config, train_loader, val_loader, "./checkpoints", lambda l, m: logger.log(getattr(logging, l), m))
            else:
                self.trainer = SupervisedTrainer(model, device, self.config, train_loader, val_loader, "./checkpoints", lambda l, m: logger.log(getattr(logging, l), m))
            
            self.is_training = True
            def train_thread():
                try: self.trainer.fit(num_epochs=epochs)
                except Exception as e: logger.error(f"Ошибка: {e}")
                finally: self.is_training = False
            
            self.training_thread = threading.Thread(target=train_thread, daemon=True)
            self.training_thread.start()
            return f"Обучение запущено: {epochs} эпох, batch_size={batch_size}"
        except Exception as e:
            self.is_training = False
            return f"Ошибка: {str(e)}"
    
    def stop_training(self, mode: str = "soft") -> str:
        if not self.is_training: return "Обучение не запущено"
        try:
            if self.trainer:
                from core.trainer_base import StopMode
                self.trainer.request_stop(StopMode.SOFT if mode == "soft" else StopMode.HARD)
            return f"Остановка ({mode}) запрошена"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def get_training_status(self) -> Dict[str, Any]:
        if not self.trainer: return {"status": "Не запущено"}
        state = self.trainer.get_state()
        return {"status": "Запущено" if state["is_running"] else "Завершено", "epoch": state["current_epoch"]}
    
    def get_logs(self, limit: int = 50) -> str:
        logs = get_ui_logs(limit=limit)
        if not logs: return "Нет логов"
        return "\n".join([f"[{l['timestamp']}] {l['level']}: {l['message']}" for l in logs[-limit:]])
    
    def get_gpu_info(self) -> str:
        info = get_device_info()
        cuda_info = check_cuda()
        text = f"Устройство: {info}\n"
        if cuda_info["available"]:
            text += f"CUDA доступна (версия {cuda_info['version']})"
        else:
            text += "CUDA недоступна, используется CPU"
        return text


app_instance = AIApp()


def create_app() -> gr.Blocks:
    with gr.Blocks(title="AI Trainer Platform", theme=gr.themes.Soft()) as app:
        gr.Markdown("# AI Trainer Platform")
        
        with gr.Tabs():
            with gr.TabItem("Подготовка"):
                config_file = gr.File(label="Конфигурация", file_types=[".yaml", ".json"])
                load_config_btn = gr.Button("Загрузить конфиг", variant="primary")
                config_status = gr.Textbox(label="Статус", interactive=False)
                
                gpu_info_box = gr.Textbox(label="GPU", value=app_instance.get_gpu_info(), interactive=False)
                
                data_paths = gr.Textbox(label="Пути к данным", value="./data")
                file_types = gr.Textbox(label="Типы файлов", value=".txt, .csv, .json")
                load_data_btn = gr.Button("Загрузить данные", variant="primary")
                data_status = gr.Textbox(label="Статус", interactive=False)
                
                train_split = gr.Slider(0.5, 0.9, value=0.8, label="Train")
                val_split = gr.Slider(0.05, 0.25, value=0.1, label="Val")
                preprocess_btn = gr.Button("Предобработать", variant="primary")
                preprocess_status = gr.Textbox(label="Результат", interactive=False)
                
                load_config_btn.click(fn=lambda x: app_instance.load_config(x.name if x else ""), inputs=[config_file], outputs=[config_status])
                load_data_btn.click(fn=lambda p, t: app_instance.load_data(p, t), inputs=[data_paths, file_types], outputs=[data_status])
                preprocess_btn.click(fn=lambda t, v: app_instance.preprocess_data(t, v), inputs=[train_split, val_split], outputs=[preprocess_status])
            
            with gr.TabItem("Обучение"):
                architecture = gr.Dropdown(choices=["transformer", "cnn", "rnn", "mlp"], value="mlp", label="Архитектура")
                hidden_size = gr.Slider(64, 2048, value=512, label="Hidden size")
                num_layers = gr.Slider(1, 12, value=4, label="Слои")
                num_classes = gr.Slider(2, 100, value=2, label="Классы")
                create_model_btn = gr.Button("Создать модель", variant="primary")
                model_status = gr.Textbox(label="Статус", interactive=False)
                
                training_type = gr.Dropdown(choices=["supervised", "genetic"], value="supervised", label="Тип")
                epochs = gr.Number(value=10, label="Эпохи")
                batch_size = gr.Slider(8, 256, value=32, label="Batch size")
                learning_rate = gr.Number(value=0.0001, label="LR")
                
                start_btn = gr.Button("Запустить", variant="primary")
                stop_soft_btn = gr.Button("Стоп (мягкий)", variant="stop")
                stop_hard_btn = gr.Button("Стоп (жесткий)", variant="stop")
                train_status = gr.Textbox(label="Статус", interactive=False)
                
                create_model_btn.click(fn=lambda a, h, n, c: app_instance.create_model(a, h, n, c), inputs=[architecture, hidden_size, num_layers, num_classes], outputs=[model_status])
                start_btn.click(fn=lambda e, b, l, t: app_instance.start_training(int(e), int(b), l, t), inputs=[epochs, batch_size, learning_rate, training_type], outputs=[train_status])
                stop_soft_btn.click(fn=lambda: app_instance.stop_training("soft"), outputs=[train_status])
                stop_hard_btn.click(fn=lambda: app_instance.stop_training("hard"), outputs=[train_status])
            
            with gr.TabItem("Мониторинг"):
                status_box = gr.JSON(label="Статус")
                logs_box = gr.Textbox(label="Логи", lines=20, interactive=False)
                refresh_btn = gr.Button("Обновить")
                clear_btn = gr.Button("Очистить")
                
                refresh_btn.click(fn=lambda: (app_instance.get_training_status(), app_instance.get_logs()), outputs=[status_box, logs_box])
                clear_btn.click(fn=lambda: (clear_ui_logs(), "Логи очищены"), outputs=[logs_box])
        
        app.load(fn=app_instance.get_training_status, outputs=[status_box], every=2)
    
    return app


def launch_app(server_name: str = "0.0.0.0", server_port: int = 7860):
    app = create_app()
    app.queue()
    app.launch(server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    launch_app()
