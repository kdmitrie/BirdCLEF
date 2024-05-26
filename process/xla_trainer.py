import torch
from ..bird.config import CFG


def get_trainer(base_trainer_class):
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        class ModelTrainerXLA(base_trainer_class):

            def _backward_pass_train(self, model, loss_value):
                """Trains the model and returns an array of loss values"""

                # 1. We reset the gradient values ...
                self.optimizer.zero_grad()

                # 2. ... and calculate the new gradient values
                loss_value.backward()

                if self.max_clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_clip_grad_norm)

                # 3. Then we renew the model parameters
                self.optimizer.step()

                # 4. For TPU
                xm.mark_step()
                # or
                # xm.optimizer_step(self.optimizer, barrier=True)

                return loss_value.detach().cpu().numpy()

        CFG.device = xm.xla_device()
        return ModelTrainerXLA
    except ImportError:
        return base_trainer_class
