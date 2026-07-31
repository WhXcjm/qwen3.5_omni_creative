import torch
import mindspeed.megatron_adaptor

from mindspeed_mm.models.diffusion.hunyuanvideo_i2v_diffusion import HunyuanVideoI2VDiffusion
from tests.ut.utils import judge_expression


class TestHunyuanVideoI2VFlowMatchScheduler:
    @staticmethod
    def test_training_init():
        train_num_timesteps = 500
        scheduler = HunyuanVideoI2VDiffusion(num_train_timesteps=train_num_timesteps)
        judge_expression(scheduler.training_timesteps == train_num_timesteps)

    @staticmethod
    def test_sample():
        test_num_timesteps = 50
        test_x1 = torch.randn([1, 16, 30, 30, 50])
        scheduler = HunyuanVideoI2VDiffusion(num_train_timesteps=test_num_timesteps)
        t, x0, x1 = scheduler.sample(test_x1)
        judge_expression(t.shape == (1,))
        judge_expression(x0.shape == test_x1.shape)
        judge_expression(x1.shape == test_x1.shape)
        judge_expression(isinstance(t, torch.Tensor))
        judge_expression(isinstance(x0, torch.Tensor))
        judge_expression(isinstance(x1, torch.Tensor))

    @staticmethod
    def test_q_sample():
        test_num_timesteps = 50
        test_x1 = torch.randn([1, 16, 30, 30, 50])
        test_model_kwargs = {"cond_latents": torch.randn([1, 16, 30, 30, 50])}
        scheduler = HunyuanVideoI2VDiffusion(num_train_timesteps=test_num_timesteps)
        xt, ut, input_t = scheduler.q_sample(test_x1, model_kwargs=test_model_kwargs)
        judge_expression(xt.shape == (1, 16, 59, 30, 50))
        judge_expression(ut.shape == (1, 16, 30, 30, 50))
        judge_expression(input_t.shape == (1,))

    @staticmethod
    def test_training_losses():
        test_num_timesteps = 50
        scheduler = HunyuanVideoI2VDiffusion(num_train_timesteps=test_num_timesteps)
        test_model_output = torch.randn([1, 16, 30, 30, 50])
        test_noise = torch.randn_like(test_model_output)
        # canculate loss
        model_output = test_model_output[:, :, 1:, :, :]
        noise = test_noise[:, :, 1:, :, :]
        loss_ = torch.mean(((model_output - noise) ** 2), dim=list(range(1, len(model_output.size()))))
        loss = scheduler.training_losses(model_output=test_model_output, noise=test_noise)
        judge_expression(torch.allclose(loss, loss_))
