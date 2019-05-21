import torch
from torch.optim import Optimizer

class Truncated(Optimizer):
    """Implements the truncated stochastic optimization method
	from the paper: https://arxiv.org/abs/1903.08619

	Applies the update:
		w_{k+1} = w_k - min(alpha_k,f_k/norm(g_k)**2)*g_k
	where:
		w_k: wieght at iteration k
		alpha_k: standard stepsize at iteration k
		g_k: gradient at iteration k
		f_k: loss at iteration k

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate


    Example:
        >>> optimizer = Truncated(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
		>>> def closure():
		>>> 	loss = loss_fn(model(input), target)
		>>>     loss.backward()
		>>>     return loss
        >>> loss = optimizer.step(closure)
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Truncated, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Truncated, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grad_norm = calc_grad_norm(self.param_groups[0]['params'])
		#grad_norm = calc_grad_norm(self.parameters())

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
				
                #update the learning rate according the to "truncated" rule
                lr_1 = group['lr']
                lr_2 = lr_1
                if grad_norm > 0:
                    #lr_2 = 1.0 * float(loss.item()) / grad_norm
                    lr_2 = 1.0 * float(loss.data[0]) / grad_norm
                lr_curr = min(lr_1, lr_2)

                p.data.add_(-lr_curr, d_p)

        return loss



class TruncatedAdagrad(Optimizer):
    """Implements Truncated-Adagrad optimization method
	from the paper: https://arxiv.org/abs/1903.08619.
		
	Applies the update:
		w_{k+1} = w_k - min(alpha_k,f_k/(g_k.transpose()*inverse(H_k)*g_k)*g_k 
	where:
		w_k: wieght at iteration k
		alpha_k: standard stepsize at iteration k
		g_k: gradient at iteration k
		f_k: loss at iteration k
		H_k: the adagrad matrix, i.e. H_k = 1/k*diag(sum_{i=1}^{k} g_i*g_i.transpose())
	   
	
	Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
		
	Example:
        >>> optimizer = TruncatedAdagrad(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
		>>> def closure():
		>>> 	return loss_fn(model(input), target).backward()
        >>> loss = optimizer.step(closure) 
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
        super(TruncatedAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        grad_normalized_norm = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)

                    grad_square = torch.mul(grad,grad)
                    norm_addition = torch.div(grad_square,std)
                    grad_normalized_norm += torch.sum(norm_addition)

        trunc_lr = clr
        if grad_normalized_norm>0:
            trunc_lr =  1.0 * float(loss.data[0]) / grad_normalized_norm
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                #calculate learning rate
                clr = group['lr'] # / (1 + (state['step'] - 1) * group['lr_decay'])

                clr = min(clr,trunc_lr)

                std = state['sum'].sqrt().add_(1e-10)
                p.data.addcdiv_(-clr, grad, std)
        return loss



def calc_grad_norm(params):
    """ Calculates the total norm of the network gradient
		
	Arguments: 
		params (iterable): parameters of the network
	"""

    total_grad_norm = 0
    for p in params:
        total_grad_norm += p.grad.data.norm(2)**2
    return total_grad_norm
