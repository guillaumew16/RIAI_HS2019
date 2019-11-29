import torch
import torch.nn as nn

def fgsm_(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""    
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target])
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    
    #perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()
    
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out.detach()

def fgsm_targeted(model, x, target, eps, **kwargs):
    return fgsm_(model, x, target, eps, targeted=True, **kwargs)

def fgsm_untargeted(model, x, label, eps, **kwargs):
    return fgsm_(model, x, label, eps, targeted=False, **kwargs)


def pgd_(model, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None):
    x_min = x - eps
    x_max = x + eps
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None) 
        # as we want to apply the attack as defined
        x = fgsm_(model, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
    #if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x

def pgd_targeted(model, x, target, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, target, k, eps, eps_step, targeted=True, **kwargs)

def pgd_untargeted(model, x, label, k, eps, eps_step, clip_min=None, clip_max=None, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, **kwargs)