"""Keep completed VAE pixel chunks on CPU during fixed-memory inference."""
from types import MethodType
import torch


@torch.no_grad()
def decode_to_cpu(model, z, scale):
    """Preserve Wan decode order/cache; concatenate completed chunks on CPU."""
    model.clear_cache()
    if isinstance(scale[0], torch.Tensor):
        z = z / scale[1].view(1,model.z_dim,1,1,1) + scale[0].view(1,model.z_dim,1,1,1)
    else:
        z = z / scale[1] + scale[0]
    chunks=[]
    try:
        x=model.conv2(z)
        for index in range(z.shape[2]):
            model._conv_idx=[0]
            chunk=model.decoder(x[:,:,index:index+1],feat_cache=model._feat_map,
                                feat_idx=model._conv_idx)
            chunks.append(chunk.detach().to('cpu'))
            del chunk
        return torch.cat(chunks,dim=2)
    finally:
        model.clear_cache()


def install_cpu_video_output(model):
    model.decode=MethodType(decode_to_cpu,model)
