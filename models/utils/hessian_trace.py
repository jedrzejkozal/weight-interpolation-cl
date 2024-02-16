import torch
import torch.autograd as autograd


def hessian_trace(model, loss, device, niters):
    """compute approximate Hessian trace
    utilize Hutchinson method
    from repo:
    https://github.com/yashkhasbage25/HTR/blob/master/Regularization/experiments/ht_penalization.py
    from paper:
    A Deeper Look at the Hessian Eigenspectrum of Deep Neural Networks and its Applications to Regularization
    """
    trace = 0.0
    all_gradients = autograd.grad(loss, model.parameters(), create_graph=True)
    for _ in range(niters):
        V_i = [torch.randint_like(p, high=2, device=device, requires_grad=False) for p in model.parameters()]
        for V_ij in V_i:
            V_ij[V_ij == 0] = -1
        Hv = autograd.grad(all_gradients, model.parameters(), V_i, create_graph=True)
        this_trace = 0.0
        for Hv_, V_i_ in zip(Hv, V_i):
            this_trace = this_trace + torch.sum(Hv_ * V_i_).cpu().detach()
        trace += this_trace.item()
    trace = trace / niters
    return trace

    # all_traces = list()

    # for i, p in enumerate(model.parameters()):
    #     trace = 0.0
    #     gradient = autograd.grad(loss, p, create_graph=True)
    #     for _ in range(niters):
    #         V_i = torch.randn_like(p, device=device, requires_grad=False)
    #         Hv = autograd.grad(gradient, [p], [V_i], create_graph=True)
    #         this_trace = 0.0
    #         for Hv_, V_i_ in zip(Hv, V_i):
    #             this_trace = this_trace + torch.sum(Hv_ * V_i_).cpu().detach()
    #         trace += this_trace.item()
    #     trace = trace / niters
    #     all_traces.append(trace.item())
    # return all_traces
