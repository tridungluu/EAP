import os
import torch


def visible_gpu(gpus):
    """
        set visible gpu.

        can be a single id, or a list

        return a list of new gpus ids
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was requested, but no CUDA-compatible device is available.')
    device_count = torch.cuda.device_count()
    invalid = [gpu for gpu in gpus if gpu < 0 or gpu >= device_count]
    if invalid:
        raise RuntimeError(
            'Requested GPU index {} but only GPU indices 0-{} are available.'
            .format(invalid, device_count - 1))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, gpus)))
    return list(range(len(gpus)))


def ngpu(gpus):
    """
        count how many gpus used.
    """
    gpus = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpus)


def occupy_gpu(gpus=None):
    """
        make program appear on nvidia-smi.
    """
    if gpus is None:
        torch.zeros(1).cuda()
    else:
        gpus = [gpus] if isinstance(gpus, int) else list(gpus)
        for g in gpus:
            torch.zeros(1).cuda(g)
