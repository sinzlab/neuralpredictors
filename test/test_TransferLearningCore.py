import torch
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from neuralpredictors.layers.cores import TransferLearningCore
from utils_for_tests import compare_vars


def test_detached_mode():
    model = TransferLearningCore(
        input_channels=1,
        tl_model_name="vgg16",
        layers=8,
        pretrained=True,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
    )

    inputs = Variable(torch.randn(128, 128))
    targets = torch.randn_like(model(inputs))
    batch = [inputs, targets]

    params_not_to_train = [par for par in model.features.TransferLearning.named_parameters()]
    params_to_train = [par for par in model.features.OutBatchNorm.named_parameters()]

    compare_variables = partial(
        compare_vars,
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device="cpu",
    )

    compare_variables(vars_change=False, params=params_not_to_train)
    compare_variables(vars_change=True, params=params_to_train)
