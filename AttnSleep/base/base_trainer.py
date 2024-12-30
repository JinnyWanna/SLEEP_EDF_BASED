import torch
from abc import abstractmethod
from numpy import inf
import numpy as np

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, fold_id, device=None): #device 인자 추가
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # 수정: 디바이스 설정 시 MPS까지 고려하도록 변경
        self.device, device_ids = self._prepare_device(config['n_gpu'], device) # 이 부분에 device 추가
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            # 모델을 DataParallel로 래핑 (멀티 GPU 사용 시)
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.fold_id = fold_id

        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        if config.resume is not None:
            # 체크포인트 로드
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch, total_epochs):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        all_outs = []
        all_trgs = []

        for epoch in range(self.start_epoch, self.epochs + 1):
            # 학습 논리를 호출 (_train_epoch은 Trainer에서 구현됨)
            result, epoch_outs, epoch_trgs = self._train_epoch(epoch, self.epochs)

            # 결과 로그 저장
            log = {'epoch': epoch}
            log.update(result)
            all_outs.extend(epoch_outs)
            all_trgs.extend(epoch_trgs)
            # 화면에 로그 출력
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # 성능 평가 및 체크포인트 저장
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Warning: Metric '{self.mnt_metric}' is not found. "
                                        "Model performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"Validation performance didn't improve for {self.early_stop} epochs. "
                                     "Training stops.")
                    break

            if epoch % self.save_period == 0:
                # 체크포인트 저장
                self._save_checkpoint(epoch, save_best=best)

        # 결과 저장
        outs_name = "outs_" + str(self.fold_id)
        trgs_name = "trgs_" + str(self.fold_id)
        np.save(self.config._save_dir / outs_name, all_outs)
        np.save(self.config._save_dir / trgs_name, all_trgs)

        if self.fold_id == self.config["data_loader"]["args"]["num_folds"] - 1:
            self._calc_metrics()

    def _prepare_device(self, n_gpu_use, device=None):
        """
        Setup device (GPU/MPS/CPU) and move model to the configured device
        """
        # 수정: device 인자가 명시적으로 전달된 경우 그대로 사용
        if device:
            return device, []

        # 수정: MPS 지원 추가
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.logger.info("Using Apple Silicon GPU (MPS)")  # 로깅 메시지 추가
            return torch.device('mps'), []
        elif torch.cuda.is_available() and n_gpu_use > 0:
            self.logger.info("Using CUDA GPU")  # CUDA 사용 로깅 메시지 추가
            n_gpu = torch.cuda.device_count()
            n_gpu_use = min(n_gpu_use, n_gpu)
            device_ids = list(range(n_gpu_use))
            return torch.device(f'cuda:{device_ids[0]}'), device_ids
        else:
            self.logger.info("Using CPU")  # CPU 사용 로깅 메시지 추가
            return torch.device('cpu'), []

    def _save_checkpoint(self, epoch, save_best=True):
        """
        Saving checkpoints
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")  # 체크포인트 저장 로그
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")  # 체크포인트 로드 로그
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # 모델 상태 로드
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # Optimizer 상태 로드
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")

    def _calc_metrics(self):
        from sklearn.metrics import classification_report
        from sklearn.metrics import cohen_kappa_score
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        import pandas as pd

        n_folds = self.config["data_loader"]["args"]["num_folds"]
        all_outs = []
        all_trgs = []

        for i in range(n_folds):
            # 결과 파일 로드
            outs = np.load(self.config._save_dir / f"outs_{i}.npy")
            trgs = np.load(self.config._save_dir / f"trgs_{i}.npy")
            all_outs.extend(outs)
            all_trgs.extend(trgs)

        # 성능 평가
        all_trgs = np.array(all_trgs).astype(int)
        all_outs = np.array(all_outs).astype(int)

        r = classification_report(all_trgs, all_outs, digits=6, output_dict=True)
        cm = confusion_matrix(all_trgs, all_outs)
        df = pd.DataFrame(r)
        df["cohen"] = cohen_kappa_score(all_trgs, all_outs)
        df["accuracy"] = accuracy_score(all_trgs, all_outs)
        df = df * 100
        file_name = self.config["name"] + "_classification_report.xlsx"
        report_save_path = self.config._save_dir / file_name
        df.to_excel(report_save_path)

        cm_file_name = self.config["name"] + "_confusion_matrix.torch"
        cm_save_path = self.config._save_dir / cm_file_name
        torch.save(cm, cm_save_path)
