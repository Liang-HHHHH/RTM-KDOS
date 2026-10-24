import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList, Predictions)
from ..base_head import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class RTMCCHead(BaseHead):


    def __init__(
            self,
            in_channels: Union[int, Sequence[int]],
            out_channels: int,
            input_size: Tuple[int, int],
            in_featuremap_size: Tuple[int, int],
            simcc_split_ratio: float = 2.0,
            final_layer_kernel_size: int = 1,
            gau_cfg: ConfigType = dict(
                hidden_dims=256,
                s=128,
                expansion_factor=2,
                dropout_rate=0.,
                drop_path=0.,
                act_fn='ReLU',
                use_rel_bias=False,
                pos_enc=False),
            loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
            score_loss: OptConfigType = None,
            decoder: OptConfigType = None,
            init_cfg: OptConfigType = None,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg

        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)

        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # --- SimCC Layers ---
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        self.final_layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=final_layer_kernel_size,
            stride=1,
            padding=final_layer_kernel_size // 2)

        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

        # ============================================================
        #   Visual Scoring Head (Mean+Max + 3-Layer MLP)
        # ============================================================
        if score_loss is not None:
            self.with_score_head = True
            self.score_loss_module = MODELS.build(score_loss)

            # 将原本构建 score_head 的代码缩进到这个 if 块里
            feat_dim = gau_cfg['hidden_dims']
            input_dim = feat_dim * 2
            self.score_head = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.with_score_head = False
            self.score_head = None
            self.score_loss_module = None

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward the network."""
        feats = feats[-1]

        # SimCC Process
        feats = self.final_layer(feats)  # (B, K, H, W)
        feats = torch.flatten(feats, 2)  # (B, K, HW)
        feats = self.mlp(feats)  # (B, K, C)
        gau_feats = self.gau(feats)  # (B, K, C) -> 这是我们要用的特征

        # Coordinate Prediction
        pred_x = self.cls_x(gau_feats)
        pred_y = self.cls_y(gau_feats)

        # ============================================================
        #   Score Prediction Logic
        # ============================================================
        # gau_feats shape: (B, K, C), e.g., (Batch, 3, 256)

        if self.with_score_head:
            mean_feats = torch.mean(gau_feats, dim=1)
            max_feats, _ = torch.max(gau_feats, dim=1)
            score_input = torch.cat([mean_feats, max_feats], dim=1)
            pred_score = self.score_head(score_input)
        else:
            pred_score = None  # 如果关闭，返回 None

        return pred_x, pred_y, pred_score

    def predict(
            self,
            feats: Tuple[Tensor],
            batch_data_samples: OptSampleList,
            test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features."""

        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _batch_pred_x, _batch_pred_y, _batch_pred_score = self.forward(_feats)
            _batch_pred_x_flip, _batch_pred_y_flip, _batch_pred_score_flip = self.forward(_feats_flip)

            _batch_pred_x_flip, _batch_pred_y_flip = flip_vectors(
                _batch_pred_x_flip,
                _batch_pred_y_flip,
                flip_indices=flip_indices)

            batch_pred_x = (_batch_pred_x + _batch_pred_x_flip) * 0.5
            batch_pred_y = (_batch_pred_y + _batch_pred_y_flip) * 0.5
            # Score TTA: 取平均

            if self.with_score_head and _batch_pred_score is not None:
                batch_pred_score = (_batch_pred_score + _batch_pred_score_flip) * 0.5
            else:
                batch_pred_score = None

        else:
            batch_pred_x, batch_pred_y, batch_pred_score = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        # 注入 Score 到结果中
        if batch_pred_score is not None:
            pred_scores_np = to_numpy(batch_pred_score)
            for i, pred_instance in enumerate(preds):
                pred_instance.point_score = pred_scores_np[i]

        if test_cfg.get('output_heatmaps', False):
            # ... (保持原有热图可视化逻辑不变) ...
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for visualization.')
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)
            B, K, _ = batch_pred_x.shape
            x = batch_pred_x.reshape(B, K, 1, -1)
            y = batch_pred_y.reshape(B, K, -1, 1)
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()]
            for pred_instances, pred_x, pred_y in zip(preds, to_numpy(batch_pred_x), to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]
            return preds, pred_fields
        else:
            return preds

    def loss(self, feats: Tuple[Tensor], batch_data_samples: OptSampleList, train_cfg: OptConfigType = {}) -> dict:

        pred_x, pred_y, pred_score = self.forward(feats)

        gt_x = torch.cat([d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples], dim=0)
        gt_y = torch.cat([d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples], dim=0)
        keypoint_weights = torch.cat([d.gt_instance_labels.keypoint_weights for d in batch_data_samples], dim=0)

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        losses = dict()
        losses['loss_kpt'] = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)

        if self.with_score_head and pred_score is not None:
            gt_scores = []
            for d in batch_data_samples:
                if hasattr(d, 'gt_instances') and hasattr(d.gt_instances, 'point_score'):
                    # 确保使用 .float() 避免类型不匹配
                    score = torch.as_tensor(d.gt_instances.point_score, device=pred_score.device).float()
                elif hasattr(d, 'gt_instance_labels') and hasattr(d.gt_instance_labels, 'point_score'):
                    score = torch.as_tensor(d.gt_instance_labels.point_score, device=pred_score.device).float()
                else:
                    score = torch.tensor([-1.0], device=pred_score.device)  # 标记为无效
                gt_scores.append(score)

            gt_scores = torch.cat(gt_scores, dim=0)

            # 过滤无效数据 (假设负数或极小值是无效标记，这里用 > -0.1 通用判断)
            valid_mask = gt_scores > -0.1

            if valid_mask.any():
                pred_valid = pred_score[valid_mask]
                gt_valid = gt_scores[valid_mask].view_as(pred_valid)  # 确保形状一致 (N, 1)

                # 计算损失
                loss_value = self.score_loss_module(pred_valid, gt_valid)
                losses['loss_score'] = loss_value

                # 监控指标
                with torch.no_grad():
                    losses['score_mae'] = torch.abs(pred_valid - gt_valid).mean()
                    losses['score_rmse'] = torch.sqrt(F.mse_loss(pred_valid, gt_valid))
            else:
                losses['loss_score'] = pred_score.sum() * 0.0
                losses['score_mae'] = torch.tensor(0.0, device=pred_score.device)

            # Pose Metrics
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        losses.update(acc_pose=torch.tensor(avg_acc, device=gt_x.device))

        return losses

    @property
    def default_init_cfg(self):
        return [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]