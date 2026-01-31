import os
import torch
import torch.optim as optim
import torch.nn as nn
from abc import abstractmethod
from typing import Optional, Dict, Any
from tensorboardX import SummaryWriter


class BaseTrainer(object):
    """提供常见训练行为的基础训练器。
        所有自定义训练器应该是该类的子类。
    """
    def __init__(self, cfg):
        self.cfg = cfg

        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.clock = TrainClock()
        self.batch_size = cfg.batch_size
        
        # 初始化网络（由子类在 build_net 中设置）
        self.net: Optional[nn.Module] = None

        # 构建网络
        self.build_net(cfg)

        # 设置损失函数
        self.set_loss_function()

        # 设置优化器
        self.set_optimizer(cfg)

        # 设置 TensorBoard 写入器
        self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))

    @abstractmethod
    def build_net(self, cfg):
        raise NotImplementedError

    def set_loss_function(self):
        """设置训练中使用的损失函数"""
        pass

    def set_optimizer(self, cfg):
        """设置训练中使用的优化器和学习率调度器"""
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)#type: ignore
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, cfg.lr_step_size)

    def save_ckpt(self, name=None):
        """在训练期间保存检查点以供将来恢复"""
        if name is None:
            save_path = os.path.join(self.model_dir, "ckpt_epoch{}.pth".format(self.clock.epoch))
            print("保存检查点周期 {}...".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, "{}.pth".format(name))

        if isinstance(self.net, nn.DataParallel):
            model_state_dict = self.net.module.cpu().state_dict()
        else:
            model_state_dict = self.net.cpu().state_dict()#type: ignore

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)

        self.net.cuda()#type: ignore

    def load_ckpt(self, name=None):
        """从已保存的检查点加载检查点"""
        name = name if name == 'latest' else "ckpt_epoch{}".format(name)
        load_path = os.path.join(self.model_dir, "{}.pth".format(name))
        if not os.path.exists(load_path):
            raise ValueError("检查点 {} 不存在。".format(load_path))

        checkpoint = torch.load(load_path)
        print("从 {} 加载检查点 ...".format(load_path))
        if isinstance(self.net, nn.DataParallel):
            self.net.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.net.load_state_dict(checkpoint['model_state_dict'])#type: ignore
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

    @abstractmethod
    def forward(self, data):
        """网络的前向逻辑"""
        """应返回网络输出、损失（字典）"""
        raise NotImplementedError

    def update_network(self, loss_dict):
        """通过反向传播更新网络"""
        # loss = sum(value for sub_dict in loss_dict.values() for value in sub_dict.values())
        loss = sum(loss_dict.values())
        self.optimizer.zero_grad()
        loss.backward()#type: ignore
        if self.cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)#type: ignore
        self.optimizer.step()

    def update_learning_rate(self):
        """记录和更新学习率"""
        self.train_tb.add_scalar('learning_rate', self.optimizer.param_groups[-1]['lr'], self.clock.epoch)
        self.scheduler.step()

    def record_losses(self, loss_dict, mode='train'):
        """将损失记录到 TensorBoard"""
        losses_values = {k: v.item() for k, v in loss_dict.items()}

        tb = self.train_tb if mode == 'train' else self.val_tb
        for k, v in losses_values.items():
            tb.add_scalar(k, v, self.clock.step)

    def train_func(self, data):
        """一步训练"""
        self.net.train()#type: ignore

        outputs, losses = self.forward(data)

        self.update_network(losses)
        if self.clock.step % 10 == 0:
            self.record_losses(losses, 'train')

        return outputs, losses

    def val_func(self, data):
        """一步验证"""
        self.net.eval()#type: ignore

        with torch.no_grad():
            outputs, losses = self.forward(data)

        self.record_losses(losses, 'validation')

        return outputs, losses

    def visualize_batch(self, data, tb, **kwargs):
        """将可视化结果写入 TensorBoard 写入器"""
        raise NotImplementedError


class TrainClock(object):
    """时钟对象，用于在训练期间跟踪周期和步骤
    """
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']
