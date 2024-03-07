from warnings import warn

import numpy as np
import torch
import torch.nn as nn

from .base import Encoder
from .mean_variance_functions import (
    fitted_zig_mean,
    fitted_zig_variance,
    fitted_zil_mean,
    fitted_zil_variance,
)


class ZeroInflationEncoderBase(Encoder):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        shifter=None,
        modulator=None,
    ):

        super().__init__()
        self.core = core
        self.readout = readout
        self.shifter = shifter
        self.modulator = modulator
        self.offset = offset
        self.zero_thresholds = zero_thresholds

        if not loc_image_dependent:
            if isinstance(zero_thresholds, dict):
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            torch.ones(1, ro.outdims) * np.log(zero_thresholds[data_key]),
                            requires_grad=False,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            elif zero_thresholds is None:
                self.logloc = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(
                            (torch.rand(1, ro.outdims) + 1) * np.log(0.1),
                            requires_grad=True,
                        )
                        for data_key, ro in self.readout.items()
                    }
                )
            else:
                raise ValueError("zero_thresholds should either be of type {data_key: zero_shreshold_value} or None.")

        else:
            if zero_thresholds is not None:
                warn("zero thresholds are set but will be ignored because loc_image_dependent is True")

        if not q_image_dependent:
            self.q = nn.ParameterDict(
                {data_key: nn.Parameter(torch.rand(1, ro.outdims) * 2 - 1) for data_key, ro in self.readout.items()}
            )

    def loc_nl(self, logloc):
        loc = torch.exp(logloc)
        assert not torch.any(loc == 0.0), "loc should not be zero! Because of numerical instability. Check the code!"
        return loc

    def q_nl(self, q):
        return torch.sigmoid(q) * 0.99999 + self.offset

    def forward_base(
        self,
        x,
        data_key=None,
        behavior=None,
        pupil_center=None,
        trial_idx=None,
        shift=None,
        detach_core=False,
        **kwargs
    ):
        batch_size = x.shape[0]

        # get readout outputs
        x = self.core(x)
        if detach_core:
            x = x.detach()

        if self.shifter:
            if pupil_center is None:
                raise ValueError("pupil_center is not given")
            shift = self.shifter[data_key](pupil_center, trial_idx)

        if "sample" in kwargs:
            x = self.readout(x, data_key=data_key, sample=kwargs["sample"], shift=shift)
        else:
            x = self.readout(x, data_key=data_key, shift=shift)

        # keep batch dimension if only one image was passed
        params = []
        for param in x:
            params.append(param[None, ...] if len(param.shape) == 1 else param)
        x = torch.stack(params)

        if self.modulator:
            x = self.modulator[data_key](x, behavior=behavior)

        readout_out_idx = 0
        if "logloc" in dir(self):
            logloc = getattr(self, "logloc")[data_key].repeat(batch_size, 1)
        else:
            logloc = x[readout_out_idx]
            readout_out_idx += 1

        if "q" in dir(self):
            q = getattr(self, "q")[data_key].repeat(batch_size, 1)
        else:
            q = x[readout_out_idx]
            readout_out_idx += 1

        return x, q, logloc, readout_out_idx


class ZIGEncoder(ZeroInflationEncoderBase):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        init_ks=None,
        theta_image_dependent=True,
        k_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-6,
        shifter=None,
        modulator=None,
    ):

        super().__init__(
            core, readout, zero_thresholds, loc_image_dependent, q_image_dependent, offset, shifter, modulator
        )

        if not theta_image_dependent:
            self.logtheta = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

        if not k_image_dependent:
            if isinstance(init_ks, dict):
                self.logk = nn.ParameterDict(
                    {
                        data_key: nn.Parameter(torch.ones(1, ro.outdims) * init_ks[data_key])
                        for data_key, ro in self.readout.items()
                    }
                )
            elif init_ks is None:
                self.logk = nn.ParameterDict(
                    {data_key: nn.Parameter(torch.ones(1, ro.outdims) * 0.0) for data_key, ro in self.readout.items()}
                )
            else:
                raise ValueError("init_ks should either be of type {data_key: init_k_value} or None.")

        else:
            if init_ks is not None:
                warn("init_ks are set but will be ignored because k_image_dependent is True")

    def theta_nl(self, logtheta):
        theta = nn.functional.elu(logtheta) + 1 + self.offset
        return theta

    def k_nl(self, logk):
        if self.zero_thresholds is not None:
            k = nn.functional.elu(logk) + 1 + self.offset
        else:
            k = nn.functional.elu(logk) + 1 + 1.1 + self.offset
        return k

    def forward(
        self, x, data_key, behavior=None, pupil_center=None, trial_idx=None, shift=None, detach_core=False, **kwargs
    ):
        batch_size = x.shape[0]

        if self.modulator:
            if behavior is None:
                raise ValueError("behavior is not given")

        x, q, logloc, readout_out_idx = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior,
            pupil_center=pupil_center,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )

        if "logtheta" in dir(self):
            logtheta = getattr(self, "logtheta")[data_key].repeat(batch_size, 1)
        else:
            logtheta = x[readout_out_idx]
            readout_out_idx += 1

        if "logk" in dir(self):
            logk = getattr(self, "logk")[data_key].repeat(batch_size, 1)
        else:
            logk = x[readout_out_idx]
            readout_out_idx += 1

        return (
            self.theta_nl(logtheta),
            self.k_nl(logk),
            self.loc_nl(logloc),
            self.q_nl(q),
        )

    def predict_mean(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_mean(theta, k, loc, q)

    def predict_variance(self, x, data_key, *args, **kwargs):
        theta, k, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zig_variance(theta, k, loc, q)


class ZILEncoder(ZeroInflationEncoderBase):
    def __init__(
        self,
        core,
        readout,
        zero_thresholds=None,
        mu_image_dependent=True,
        sigma2_image_dependent=True,
        loc_image_dependent=False,
        q_image_dependent=True,
        offset=1.0e-12,
        shifter=None,
        modulator=None,
    ):

        super().__init__(
            core, readout, zero_thresholds, loc_image_dependent, q_image_dependent, offset, shifter, modulator
        )

        if not mu_image_dependent:
            self.mu = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

        if not sigma2_image_dependent:
            self.logsigma2 = nn.ParameterDict(
                {data_key: nn.Parameter(torch.zeros(1, ro.outdims)) for data_key, ro in self.readout.items()}
            )

    def sigma2_nl(self, logsigma2):
        return nn.functional.elu(logsigma2) + 1 + self.offset

    def mu_nl(self, mu):
        return mu

    def forward(
        self, x, data_key, behavior=None, pupil_center=None, trial_idx=None, shift=None, detach_core=False, **kwargs
    ):
        batch_size = x.shape[0]
        x, q, logloc, readout_out_idx = self.forward_base(
            x,
            data_key=data_key,
            behavior=behavior,
            pupil_center=pupil_center,
            trial_idx=trial_idx,
            shift=shift,
            detach_core=detach_core,
            **kwargs
        )

        if "logsigma2" in dir(self):
            logsigma2 = getattr(self, "logsigma2")[data_key].repeat(batch_size, 1)
        else:
            logsigma2 = x[readout_out_idx]
            readout_out_idx += 1

        if "mu" in dir(self):
            mu = getattr(self, "mu")[data_key].repeat(batch_size, 1)
        else:
            mu = x[readout_out_idx]
            readout_out_idx += 1

        return (
            self.mu_nl(mu),
            self.sigma2_nl(logsigma2),
            self.loc_nl(logloc),
            self.q_nl(q),
        )

    def predict_mean(self, x, data_key, *args, **kwargs):
        mu, sigma2, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zil_mean(mu, sigma2, loc, q)

    def predict_variance(self, x, data_key, *args, **kwargs):
        mu, sigma2, loc, q = self.forward(x, *args, data_key=data_key, **kwargs)
        return fitted_zil_variance(mu, sigma2, loc, q)
