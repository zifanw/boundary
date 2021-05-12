try:
    import torch
except:
    pass

try:
    import tensorflow as tf
except:
    pass

import numpy as np
from boundary.attributions import PT_IntegratedGradient
from boundary.attributions import PT_SmoothGradient
from boundary.attributions import PT_SaliencyGradient
from boundary.attributions import PT_DeepLIFT
from boundary.attributions import TF2_IntegratedGradient
from boundary.attributions import TF2_SmoothGradient
from boundary.attributions import TF2_SaliencyGradient

import boundary.default as D

import foolbox as fb
from tqdm import tqdm, trange


def take_closer_bd(x, y, cls_bd, dis2cls_bd, boundary_points, boundary_labels):
    """Compare and return adversarial examples that are closer to the input

    Args:
        x (np.ndarray): Benign inputs
        y (np.ndarray): Labels of benign inputs
        cls_bd (None or np.ndarray): Points on the closest boundary
        dis2cls_bd ([type]): Distance to the closest boundary
        boundary_points ([type]): New points on the closest boundary
        boundary_labels ([type]): Labels of new points on the closest boundary

    Returns:
        (np.ndarray, np.ndarray): Points on the closest boundary and distances
    """
    if cls_bd is None:
        cls_bd = boundary_points
        dis2cls_bd = np.linalg.norm(np.reshape((boundary_points - x),
                                               (x.shape[0], -1)),
                                    axis=-1)
        return cls_bd, dis2cls_bd
    else:
        d = np.linalg.norm(np.reshape((boundary_points - x), (x.shape[0], -1)),
                           axis=-1)
        for i in range(cls_bd.shape[0]):
            if d[i] < dis2cls_bd[i] and y[i] != boundary_labels[i]:
                dis2cls_bd[i] = d[i]
                cls_bd[i] = boundary_points[i]
    return cls_bd, dis2cls_bd


def convert_to_numpy(x):
    if not isinstance(x, np.ndarray):
        return x.numpy()
    else:
        return x


def to_device(x, y, model, device):
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    model = model.to(device)

    return x, y, model


class PytorchModel(object):
    """A keras-like model wrapper for Pytorch models
    """
    def __init__(self, pytorch_model):
        self.model = pytorch_model

    # Overload __call__() method
    def __call__(self, x, batch_size=32, training=False, device='cuda:0'):
        if training:
            self.model = self.model.train()
        else:
            self.model = self.model.eval()

        outputs = []
        for i in range(0, x.shape[0], batch_size):
            out = self.model(x[i:i + batch_size].to(device))
            outputs.append(out.detach().cpu())
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def evaluate(self, x, y, batch_size=32):
        pred = self(x, batch_size=batch_size)
        correct = torch.mean((pred == y).float())
        return correct

    def eval(self):
        return PytorchModel(self.model.eval())

    def to(self, device):
        return PytorchModel(self.model.to(device))

    def cuda(self):
        return PytorchModel(self.model.cuda())


def get_boundary_points(model,
                        x,
                        y_onehot,
                        batch_size=64,
                        pipeline=['pgd'],
                        search_range=['local', 'l2', 0.3, None, 100],
                        clamp=[0, 1],
                        backend='pytorch',
                        device='cuda:0',
                        **kwargs):

    """Find nearby boundary points by running adversarial attacks

    Args:
        model (tf.models.Model or torch.nn.Module): tf.keras model or pytorch model
        x (np.ndarray): Benigh inputs
        y_onehot (np.ndarray): One-hot labels for the benign inputs
        batch_size (int, optional): Batch size. Defaults to 64.
        pipeline (list, optional): A list of adversarial attacks used to find nearby boundaries. Defaults to ['pgd'].
        search_range (list, optional): Parameters shared by all adversarial attacks. Defaults to ['local', 'l2', 0.3, None, 100].
        clamp (list, optional): Data range. Defaults to [0, 1].
        backend (str, optional): Deep learning frame work. It is either 'tf.keras' or 'pytorch'. Defaults to 'pytorch'.
        device (str, optional): GPU device to run the attack. This only matters if the backend is 'pytorch'. Defaults to 'cuda:0'.

    Returns:
        (np.ndarray, np.ndarray): Points on the closest boundary and distances
    """

    bd = None
    dis2cls_bd = np.zeros(x.shape[0]) + 1e16
    if 'pgd' in pipeline:
        print(">>> Start PGD Attack <<<", end='\n', flush=True)
        if backend == 'tf.keras':
            fmodel = fb.TensorFlowModel(model, bounds=(clamp[0], clamp[1]))
            x = tf.constant(x, dtype=tf.float32)
            y_onehot = tf.constant(y_onehot, dtype=tf.int32)
            if isinstance(search_range[2], float):
                if search_range[1] == 'l2':
                    attack = fb.attacks.L2PGD(
                        rel_stepsize=search_range[3] if search_range[3]
                        is not None else 2 * search_range[2] / search_range[4],
                        steps=search_range[4])
                else:
                    attack = fb.attacks.LinfPGD(
                        rel_stepsize=search_range[3] if search_range[3]
                        is not None else 2 * search_range[2] / search_range[4],
                        steps=search_range[4])

                boundary_points = []
                success = 0
                for i in trange(0, x.shape[0], batch_size):
                    batch_x = x[i:i + batch_size]
                    batch_y = y_onehot[i:i + batch_size]

                    _, batch_boundary_points, batch_success = attack(
                        fmodel,
                        batch_x,
                        tf.argmax(batch_y, -1),
                        epsilons=[search_range[2]])

                    boundary_points.append(
                        batch_boundary_points[0].unsqueeze(0))
                    success += np.sum(batch_success)

                boundary_points = tf.concat(boundary_points, axis=0)
                success /= x.shape[0]

                print(
                    f">>> Attacking with EPS={search_range[2]} (norm={search_range[1]}), Success Rate={success} <<<"
                )

            elif isinstance(search_range[2], (list, np.ndarray)):
                boundary_points = []
                success = 0.
                for i in trange(0, x.shape[0], batch_size):

                    batch_x = x[i:i + batch_size]
                    batch_y = y_onehot[i:i + batch_size]

                    batch_boundary_points = None
                    batch_success = None

                    for eps in search_range[2]:
                        if search_range[1] == 'l2':
                            attack = fb.attacks.L2PGD(
                                rel_stepsize=search_range[3] if search_range[3]
                                is not None else 2 * eps / search_range[4],
                                steps=search_range[4])
                        else:
                            attack = fb.attacks.LinfPGD(
                                rel_stepsize=search_range[3] if search_range[3]
                                is not None else 2 * eps / search_range[4],
                                steps=search_range[4])

                        _, c_boundary_points, c_success = attack(
                            fmodel,
                            batch_x,
                            tf.argmax(batch_y, -1),
                            epsilons=[eps])
                        c_boundary_points = c_boundary_points[0].numpy()
                        c_success = tf.cast(c_success[0], tf.int32).numpy()

                        print(
                            f">>> Attacking with EPS={eps} (norm={search_range[1]}), Success Rate={tf.reduce_mean(tf.cast(c_success, tf.float32))} <<<"
                        )

                        if batch_boundary_points is None:
                            batch_boundary_points = c_boundary_points
                            batch_success = c_success
                        else:
                            for i in range(batch_boundary_points.shape[0]):
                                if batch_success[i] == 0 and c_success[i] == 1:
                                    batch_boundary_points[
                                        i] = c_boundary_points[i]
                                    batch_success[i] = c_success[i]

                    boundary_points.append(batch_boundary_points)
                    success += np.sum(batch_success)

                boundary_points = tf.concat(boundary_points, axis=0)
                success /= x.shape[0]

            else:
                raise TypeError(
                    f"Expecting eps as float or list, but got {type(search_range[3])}"
                )

            y_pred = np.argmax(
                model.predict(boundary_points, batch_size=batch_size), -1)

            x = x.numpy()
            y_onehot = y_onehot.numpy()
            boundary_points = boundary_points.numpy()

        elif backend == 'pytorch':
            model.eval()
            x, y_onehot, model = to_device(x, y_onehot, model, device)
            fmodel = fb.PyTorchModel(model, bounds=(clamp[0], clamp[1]))

            model = PytorchModel(model)
            if isinstance(search_range[2], float):
                if search_range[1] == 'l2':
                    attack = fb.attacks.L2PGD(
                        rel_stepsize=search_range[3] if search_range[3]
                        is not None else 2 * search_range[2] / search_range[4],
                        steps=search_range[4])
                else:
                    attack = fb.attacks.LinfPGD(
                        rel_stepsize=search_range[3] if search_range[3]
                        is not None else 2 * search_range[2] / search_range[4],
                        steps=search_range[4])

                boundary_points = []
                success = 0
                for i in trange(0, x.shape[0], batch_size):
                    batch_x = x[i:i + batch_size]
                    batch_y = y_onehot[i:i + batch_size]

                    _, batch_boundary_points, batch_success = attack(
                        fmodel,
                        batch_x,
                        torch.argmax(batch_y, -1),
                        epsilons=[search_range[2]])

                    boundary_points.append(
                        batch_boundary_points[0].unsqueeze(0))
                    success += torch.sum(batch_success.detach())

                boundary_points = torch.cat(boundary_points, dim=0)
                success /= x.shape[0]

                print(
                    f">>> Attacking with EPS={search_range[2]} (norm={search_range[1]}), Success Rate={success.cpu().numpy()} <<<"
                )

            elif isinstance(search_range[2], (list, np.ndarray)):
                boundary_points = []
                success = 0.
                for i in trange(0, x.shape[0], batch_size):

                    batch_x = x[i:i + batch_size]
                    batch_y = y_onehot[i:i + batch_size]

                    batch_boundary_points = None
                    batch_success = None

                    for eps in search_range[2]:
                        if search_range[1] == 'l2':
                            attack = fb.attacks.L2PGD(
                                rel_stepsize=search_range[3] if search_range[3]
                                is not None else 2 * eps / search_range[4],
                                steps=search_range[4])
                        else:
                            attack = fb.attacks.LinfPGD(
                                rel_stepsize=search_range[3] if search_range[3]
                                is not None else 2 * eps / search_range[4],
                                steps=search_range[4])
                        _, c_boundary_points, c_success = attack(
                            fmodel,
                            batch_x,
                            torch.argmax(batch_y, -1),
                            epsilons=[eps])
                        c_boundary_points = c_boundary_points[0]
                        c_success = c_success.squeeze(0)

                        print(
                            f">>> Attacking with EPS={eps} (norm={search_range[1]}), Success Rate={c_success.detach().cpu().numpy().mean()} <<<"
                        )

                        if batch_boundary_points is None:
                            batch_boundary_points = c_boundary_points.detach(
                            ).cpu()
                            batch_success = c_success.detach().cpu()
                        else:
                            for i in range(batch_boundary_points.shape[0]):
                                if batch_success[i] == 0 and c_success[i] == 1:
                                    batch_boundary_points[
                                        i] = c_boundary_points[i]
                                    batch_success[i] = c_success[i]

                    boundary_points.append(batch_boundary_points)
                    success += torch.sum(batch_success.detach()).float()

                boundary_points = torch.cat(boundary_points, dim=0)
                success /= x.shape[0]

            else:
                raise TypeError(
                    f"Expecting eps as float or list, but got {type(search_range[3])}"
                )

            torch.cuda.empty_cache()
            y_pred = model(boundary_points,
                           batch_size=batch_size,
                           training=False,
                           device=device)

            x = x.detach().cpu().numpy()
            y_onehot = y_onehot.detach().cpu().numpy()
            y_pred = y_pred.numpy()
            boundary_points = boundary_points.detach().cpu().numpy()
        
        else:
            raise ValueError(f"Unknow backend: {backend}")

        bd, dis2cls_bd = take_closer_bd(x, np.argmax(y_onehot, -1), bd,
                                        dis2cls_bd, boundary_points,
                                        np.argmax(y_pred, -1))

    if 'cw' in pipeline:
        print(">>> Start CW Attack <<<", end='\n', flush=True)

        if backend == 'tf.keras':
            fmodel = fb.TensorFlowModel(model, bounds=(clamp[0], clamp[1]))
            x = tf.constant(x, dtype=tf.float32)
            y_onehot = tf.constant(y_onehot, dtype=tf.int32)

            attack = fb.attacks.L2CarliniWagnerAttack(
                stepsize=search_range[3] if search_range[3] is not None else
                2 * search_range[2] / search_range[4],
                steps=search_range[4])

            boundary_points = []
            success = 0.
            for i in trange(0, x.shape[0], batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y_onehot[i:i + batch_size]

                _, batch_boundary_points, batch_success = attack(
                    fmodel,
                    batch_x,
                    tf.argmax(batch_y, -1),
                    epsilons=[search_range[2]])
                boundary_points.append(batch_boundary_points[0])
                success += tf.reduce_sum(tf.cast(batch_success, tf.int32))

            boundary_points = tf.concat(boundary_points, axis=0)
            success /= x.shape[0]

            print(
                f">>> Attacking with EPS={search_range[2]} (norm={search_range[1]}), Success Rate={success} <<<"
            )

            y_pred = np.argmax(
                model.predict(boundary_points, batch_size=batch_size), -1)

            x = x.numpy()
            y_onehot = y_onehot.numpy()
            boundary_points = boundary_points.numpy()

        elif backend == 'pytorch':
            model.eval()
            x, y_onehot, model = to_device(x, y_onehot, model, device)
            fmodel = fb.PyTorchModel(model, bounds=(clamp[0], clamp[1]))

            model = PytorchModel(model)
            attack = fb.attacks.L2CarliniWagnerAttack(
                stepsize=search_range[3] if search_range[3] is not None else
                2 * search_range[2] / search_range[4],
                steps=search_range[4])

            boundary_points = []
            success = 0.
            for i in trange(0, x.shape[0], batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y_onehot[i:i + batch_size]

                _, batch_boundary_points, batch_success = attack(
                    fmodel,
                    batch_x,
                    torch.argmax(batch_y, -1),
                    epsilons=[search_range[2]])
                boundary_points.append(batch_boundary_points[0])
                success += torch.sum(batch_success.detach())

            boundary_points = torch.cat(boundary_points, dim=0)
            success /= x.shape[0]

            print(
                f">>> Attacking with EPS={search_range[2]} (norm={search_range[1]}), Success Rate={success.cpu().numpy()} <<<"
            )

            y_pred = model(boundary_points,
                           batch_size=batch_size,
                           training=False,
                           device=device)

            x = x.detach().cpu().numpy()
            y_onehot = y_onehot.detach().cpu().numpy()
            y_pred = y_pred.detach().cpu().numpy()
            boundary_points = boundary_points.detach().cpu().numpy()

        else:
            raise ValueError(f"Unknow backend: {backend}")

        bd, dis2cls_bd = take_closer_bd(x, np.argmax(y_onehot, -1), bd,
                                        dis2cls_bd, boundary_points,
                                        np.argmax(y_pred, -1))

    return convert_to_numpy(bd), dis2cls_bd


class BA_pytorch(object):
    """Wrapper of boundary attributions for Pytorch models
    """
    def __init__(self, attribution, use_boundary=True):

        if attribution in ['BIG', 'big', 'bIG', 'Big', 'Boundary-based_IG']:
            self._attribution_name = 'BIG'
            self._attribution_fn = PT_IntegratedGradient
        elif attribution in ['BSM', 'bsm', 'bSM', 'Bsm', 'Boundary-based_SM']:
            self._attribution_name = 'BSM'
            self._attribution_fn = PT_SaliencyGradient
        elif attribution in ['BSG', 'bsg', 'bSG', 'Bsg', 'Boundary-based_SG']:
            self._attribution_name = 'BSM'
            self._attribution_fn = PT_SmoothGradient
        elif attribution in ['bDeepLIFT', 'BDeepLIFT', 'BDT']:
            self._attribution_name = 'BDT'
            self._attribution_fn = PT_DeepLIFT
        else:
            raise NotImplementedError(
                f"{attribution} is not implemented yet. Please make an Issue on our github: https://github.com/zifanw/boundary"
            )

        self._use_boundary = use_boundary

    def attribute(self,
                  model,
                  x,
                  y_onehot,
                  pipline=None,
                  return_dis=False,
                  **kwargs):
        if not self._use_boundary:
            return self.compute_attr(x, y_onehot, model, self._attribution_fn,
                                     **kwargs)
        else:
            bd, dis2cls_bd = self.boundary_search(model, x, y_onehot, pipline,
                                                  **kwargs)
            if self._attribution_name in ['BSM', 'BSG']:
                attr = self.compute_attr_without_baseline(
                    x, bd, y_onehot, model, self._attribution_fn, **kwargs)
            elif self._attribution_name in ['BIG', 'BDT']:
                attr = self.compute_attr_with_baseline(x, bd, y_onehot, model,
                                                       self._attribution_fn,
                                                       **kwargs)

            if return_dis:
                return attr, dis2cls_bd
            else:
                return attr

    @staticmethod
    def boundary_search(model, x, y_onehot, pipeline, **kwargs):

        batch_size = kwargs['batch_size']
        data_min, data_max = kwargs['data_min'], kwargs['data_max']
        device = kwargs['device']
        backend = 'pytorch' if 'backend' not in kwargs else kwargs['backend']

        if pipeline is None:
            methods = 'PGDs+CW'
            pgd_search_range = D.PGD_SEARCH_RANGE
            cw_search_range = D.CW_SEARCH_RANGE
        else:
            methods = pipeline['methods']

            pgd_search_range = [
                'local', pipeline['norm'], pipeline['pgd_eps'],
                pipeline['pgd_step_size'], pipeline['pgd_max_steps']
            ]

            cw_search_range = [
                'local', pipeline['norm'], pipeline['cw_eps'],
                pipeline['cw_step_size'], pipeline['cw_max_steps']
            ]

        cls_bd = None
        dis2cls_bd = None
        boundary_labels = []

        if 'PGDs' in methods:
            pgd_bd_x, _ = get_boundary_points(model,
                                              x,
                                              y_onehot,
                                              batch_size=batch_size,
                                              pipeline='pgd',
                                              search_range=pgd_search_range,
                                              clamp=[data_min, data_max],
                                              backend=backend,
                                              device=device)
            cls_bd, dis2cls_bd = take_closer_bd(x, y_onehot, cls_bd,
                                                dis2cls_bd, pgd_bd_x, None)

        if 'CW' in methods:
            cw_bd_x, _ = get_boundary_points(model,
                                             x,
                                             y_onehot,
                                             batch_size=batch_size,
                                             pipeline='cw',
                                             search_range=cw_search_range,
                                             clamp=[data_min, data_max],
                                             backend=backend,
                                             device=device)

            for i in trange(0, cw_bd_x.shape[0], batch_size):
                batch_x = cw_bd_x[i:i + batch_size]
                batch_x = torch.tensor(batch_x)
                _, pred = torch.max(
                    model(batch_x.to(device)).detach().cpu().data, 1)
                boundary_labels += list(pred)

            cls_bd, dis2cls_bd = take_closer_bd(x, y_onehot, cls_bd,
                                                dis2cls_bd, cw_bd_x,
                                                boundary_labels)

        return cls_bd, dis2cls_bd

    @staticmethod
    def compute_attr_without_baseline(x, bd, y_onehot, model, method,
                                      **kwargs):
        attr = []
        batch_size = kwargs['batch_size']
        for i in trange(0, x.shape[0], batch_size):
            y_mini_batch = y_onehot[i:i + batch_size]
            bd_mini_batch = bd[i:i + batch_size]
            mini_batch_attr = method(model, bd_mini_batch, y_mini_batch,
                                     **kwargs)
            attr.append(mini_batch_attr)
        attr = np.vstack(attr)
        return attr

    @staticmethod
    def compute_attr_with_baseline(x, bd, y_onehot, model, method, **kwargs):
        attr = []
        batch_size = kwargs['batch_size']
        for i in trange(0, x.shape[0], batch_size):
            x_mini_batch = x[i:i + batch_size]
            y_mini_batch = y_onehot[i:i + batch_size]
            bd_mini_batch = bd[i:i + batch_size]
            mini_batch_attr = method(model,
                                     x_mini_batch,
                                     y_mini_batch,
                                     baseline=bd_mini_batch,
                                     **kwargs)
            attr.append(mini_batch_attr)
        attr = np.vstack(attr)
        return attr

    @staticmethod
    def compute_attr(x, y_onehot, model, method, **kwargs):
        attr = []

        batch_size = kwargs['batch_size']

        for i in trange(0, x.shape[0], batch_size):
            x_mini_batch = x[i:i + batch_size]
            y_mini_batch = y_onehot[i:i + batch_size]
            mini_batch_attr = method(model, x_mini_batch, y_mini_batch,
                                     **kwargs)

            attr.append(mini_batch_attr)
        attr = np.vstack(attr)
        return attr


class BA_tensorflow(BA_pytorch):
    """Wrapper of boundary attributions for tf.keras models
    """
    def __init__(self, attribution, use_boundary=True):
        if attribution in ['BIG', 'big', 'bIG', 'Big', 'Boundary-based_IG']:
            self._attribution_name = 'BIG'
            self._attribution_fn = TF2_IntegratedGradient
        elif attribution in ['BSM', 'bsm', 'bSM', 'Bsm', 'Boundary-based_SM']:
            self._attribution_name = 'BSM'
            self._attribution_fn = TF2_SaliencyGradient
        elif attribution in ['BSG', 'bsg', 'bSG', 'Bsg', 'Boundary-based_SG']:
            self._attribution_name = 'BSM'
            self._attribution_fn = TF2_SmoothGradient
        else:
            raise NotImplementedError(
                f"{attribution} is not implemented yet. Please make a Issue on our github: https://github.com/zifanw/boundary"
            )

        self._use_boundary = use_boundary
