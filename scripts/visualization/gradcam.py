import torch, torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        layer = dict([*self._named_modules(model)])[target_layer_name]
        self.fmap = None; self.grad = None
        layer.register_forward_hook(self._fwd_hook)

    def _named_modules(self, m, prefix=""):
        for n, mod in m.named_children():
            name = f"{prefix}.{n}" if prefix else n
            yield name, mod
            yield from self._named_modules(mod, name)

    def _fwd_hook(self, mod, inp, out):
        self.fmap = out
        self.grad = None
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, 'grad', g))

    def __call__(self, forward_logits_fn, class_idx=1):
        logits = forward_logits_fn()
        score = logits[:, class_idx].sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)
        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (w * self.fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return logits, cam