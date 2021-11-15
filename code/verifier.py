import argparse
import torch
# from code.networks import Normalization
from networks import FullyConnected
from networks import *
from torch import nn

DEVICE = 'cpu'
INPUT_SIZE = 28

class DeepPoly():
    def __init__(self, net, inputs, eps, true_label, back_subs=0):
        assert eps > 0
        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label
        self.back_subs = back_subs
        
        self.layers = [DeepPolyInputLayer(self.eps)]
        prev_layer = None
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                prev_layer = DeepPolyAffineLayer(weights=layer._parameters['weight'].detach(),\
                            bias=layer._parameters['bias'].detach(), prev_layer=prev_layer, back_subs=self.back_subs)
                self.layers.append(prev_layer)
            elif isinstance(layer, torch.nn.Flatten):
                prev_layer = DeepPolyFlattenLayer(prev_layer=prev_layer)
                self.layers.append(prev_layer)
            elif isinstance(layer, Normalization):
                prev_layer = DeepPolyNormalizeLayer(prev_layer=prev_layer)
                self.layers.append(prev_layer)
            elif isinstance(SPU):
                prev_layer = DeepPolySPULayer(prev_layer=prev_layer, back_subs=self.back_subs)
                self.layers.append(prev_layer)
            else:
                raise TypeError('Layer type unknown!')

        self.layers.append(DeepPolyOutputLayer(true_label=self.true_label, prev_layer=prev_layer, back_subs=self.back_subs))
        
        self.transformer = nn.Sequential(*self.layers)

    def verify(self):
        return self.transformer(self.inputs)
         

    
class DeepPolyInputLayer(nn.Module):
    def __init__(self, eps):
        super(DeepPolyInputLayer, self).__init__()
        self.eps = eps

    def forward(self, input:torch.Tensor):
        lower = torch.clamp(input - self.eps, 0., 1.)
        upper = torch.clamp(input + self.eps, 0., 1.)
        self.bounds = torch.stack((lower, upper), 0)
        return self.bounds


class DeepPolyFlattenLayer(nn.Module):
    def __init__(self, prev_layer=None):
        super(DeepPolyFlattenLayer, self).__init__()
        self.prev_layer = prev_layer

    def forward(self, bounds):
        return torch.stack([bounds[0].flatten(), bounds[1].flatten()], 0)

    def _back_sub(self, num_steps, params=None):
        bounds = self.last.back_substitution(num_steps, params=params)
        bounds = torch.stack([bounds[:int(len(bounds)/2)], bounds[int(len(bounds)/2)]], dim=0)
        return bounds
    
    '''These codes are necessary as a property here to get consistency of function calls in backpropagation.'''
    @property
    def intercept(self):
        return self.prev_layer.intercept.flatten()

    @property
    def lower_slope(self):
        return self.prev_layer.lower_slope.flatten()

    @property
    def upper_slope(self):
        return self.prev_layer.upper_slope.flatten() 

    @property
    def bounds(self):
        return torch.stack([self.prev_layer.bounds[0].flatten(), self.prev_layer.bounds[1].flatten()], 0)

        


class DeepPolyNormalizeLayer(nn.Module):
    def __init__(self, prev_layer = None):
        super(DeepPolyNormalizeLayer, self).__init__()
        self.prev_layer = prev_layer
        self.mean = torch.FloatTensor([0.1307])
        self.variance = torch.FloatTensor([0.3081])

    def forward(self, bounds):
        self.bounds = torch.div(bounds - self.mean, self.variance)
        return self.bounds



class DeepPolyAffineLayer(nn.Module):
    def __init__(self, weights, bias:bool=None, prev_layer=None, back_subs:int=0):
        self.weights = weights
        self.bias = bias
        self.prev_layer = prev_layer
        self.back_subs = back_subs

        self.weight_positive = torch.clip(self.weights, min = 0.)
        self.weight_negative = torch.clip(self.weights, max = 0.)

    def forward(self, bounds):
        upper = torch.matmul(self.weight_positive, bounds[1, :]) + torch.matmul(self.weight_negative, bounds[0, :])
        lower = torch.matmul(self.weight_positive, bounds[0, :]) + torch.matmul(self.weight_negative, bounds[1, :])
        self.bounds = torch.stack((lower, upper), 0)
        if self.bias is not None:
            self.bounds += self.bias.reshape(-1, 1)
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)

    def back_substitution(self, num_steps):
        new_bounds = self._back_sub(num_steps)
        idx_lower = new_bounds[0] > self.bounds[0]
        idx_upper = new_bounds[1] < self.bounds[1]
        self.bounds[0, idx_lower] = new_bounds[0, idx_lower]
        self.bounds[1, idx_upper] = new_bounds[1, idx_upper]
        
    def _back_sub(self, num_steps, params = None):
        lower_slope, upper_slope, lower_intercept, upper_intercept = self.weights, self.weights, self.bias, self.bias if params is None else params
        if num_steps > 0 and self.prev_layer.prev_layer.prev_layer is not None:
            new_lower_slope = torch.clamp(lower_slope, min = 0) * self.prev_layer.lower_slope + torch.clamp(lower_slope, max = 0) * self.prev_layer.upper_slope
            new_upper_slope = torch.clamp(upper_slope, min = 0) * self.prev_layer.upper_slope + torch.clamp(upper_slope, max = 0) * self.prev_layer.lower_slope
            new_lower_intercept = lower_intercept + torch.matmul(torch.clamp(lower_slope, max = 0), self.prev_layer.upper_intercept) +\
                 torch.matmul(torch.clamp(lower_slope, min = 0), self.prev_layer.lower_intercept)
            new_upper_intercept = upper_intercept + torch.matmul(torch.clamp(upper_slope, max = 0), self.prev_layer.lower_intercept) +\
                 torch.matmul(torch.clamp(upper_slope, min = 0), self.prev_layer.upper_intercept)

            return self.prev_layer._back_sub(num_steps - 1, params = (new_lower_slope, new_upper_slope, new_lower_intercept, new_upper_intercept))

        else:
            lower = torch.matmul(torch.clamp(lower_slope, min = 0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1]) + lower_intercept
            upper = torch.matmul(torch.clamp(upper_slope, min = 0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, max=0), self.prev_layer.bounds[0]) + upper_intercept
            return torch.satck([lower, upper], 0)




class DeepPolySPULayer(nn.Module):
    def __init__(self, prev_layer = None, back_subs = 0):
        super(DeepPolySPULayer, self).__init__()
        self.prev_layer = prev_layer
        self.back_subs = back_subs

    def forward(self, bounds):
        # Index for all positive case
        idx1 = bounds[0] >= 0
        # Index for cross boundary case
        idx2 = (bounds[1]>0) * (bounds[0]<0)
        # Index for cross boundary with upperbound greater than lower bound
        idx3 = (bounds[1] > - bounds[0]) * idx2
        # Index for all negative case
        idx4 = bounds[1] <= 0

        # Slope of upper bound, lower bound and intercept
        self.upper_slope = torch.zeros_like(bounds[1])
        self.lower_slope = torch.zeros_like(bounds[1])
        self.upper_intercept = torch.zeros_like(bounds[1])
        self.lower_intercept = torch.zeros_like(bounds[1])

        self.bounds = torch.zeros_like(bounds)

        # All positive case
        self.bounds[1, idx1] = bounds[1, idx1]**2 - 0.5
        self.bounds[0, idx1] = (bounds[0, idx1] + bounds[1, idx1]) * (3 * bounds[0, idx1] - bounds[1, idx1]) / 4 - 0.5
        self.upper_slope[idx1] = bounds[1, idx1] + bounds[0, idx1]
        self.lower_slope[idx1] = bounds[1, idx1] + bounds[0, idx1] 
        self.upper_intercept[idx1] = -bounds[1,idx1] * bounds[0, idx1] - 0.5
        self.lower_intercept[idx1] = -0.25 * (bounds[1, idx1] + bounds[0,idx1])**2 - 0.5

        # Cross boundary case (upper smaller than lower)
        exp_l = torch.exp(bounds[0, idx2])
        slope1 = - torch.div( exp_l, (1 + exp_l)**2 )
        slope2 = torch.div(bounds[1, idx2]**2 - 0.5 + torch.div( exp_l, 1 + exp_l ), bounds[1, idx2] - bounds[0, idx2])
        temp_idx = slope1 > slope2
        slope2[temp_idx] = slope1[temp_idx]
        self.upper_slope[idx2] = slope2

        self.lower_slope[idx2] = torch.div(-0.5 + torch.div(exp_l, 1+exp_l), -bounds[0, idx2]) 
        self.upper_intercept[idx2] = - slope2 * bounds[0, idx2] - torch.div(exp_l, 1 + exp_l)
        self.lower_intercept[idx2] = -0.5 * torch.ones_like(bounds[0, idx2])

        self.bounds[0, idx2] = torch.div(-0.5 * bounds[1, idx2] + torch.div(bounds[1, idx2] * exp_l, 1 + exp_l), -bounds[0, idx2]) - 0.5
        upper1 = - torch.div(exp_l, 1 + exp_l)
        upper2 = slope2 * (bounds[1, idx2] - bounds[0, idx2]) + upper1
        temp_idx = upper1 > upper2
        upper2[temp_idx] = upper1[temp_idx]
        self.bounds[1, idx2] = upper2

        # Cross boundary case 2 (upper greater than lower)
        self.bounds[0, idx3] = (bounds[0, idx3] + bounds[1, idx3]) * (3 * bounds[0, idx3] - bounds[1, idx3]) / 4 - 0.5
        exp_l = torch.exp(bounds[0, idx3])
        upper1 = (bounds[1, idx3])**2 - 0.5
        upper2 = torch.div(-exp_l, 1 + exp_l)
        temp_idx = upper1 > upper2
        upper2[temp_idx] = upper1[temp_idx]
        self.bounds[1, idx3] = upper2
        
        self.lower_slope[idx3] = bounds[1, idx3] + bounds[0, idx3]
        self.lower_intercept[idx3] = - 0.25 * (bounds[1, idx3] + bounds[0, idx3])**2 - 0.5

        self.upper_slope[idx3] = torch.div(bounds[1, idx3]**2 - 0.5 + torch.div(exp_l, 1 + exp_l), bounds[1, idx3] - bounds[0, idx3])
        self.upper_intercept[idx3] = - bounds[0, idx3] * self.upper_slope[idx3] - torch.div(exp_l, 1+ exp_l)

        # All negative case
        mid = (bounds[1, idx4] + bounds[0, idx4]) / 2
        exp_mid = torch.exp(mid)
        exp_u = torch.exp(bounds[1, idx4])
        exp_l = torch.exp(bounds[0, idx4])
        self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])

        self.upper_intercept[idx4] = - torch.upper_slope[idx4] * mid + torch.div(exp_mid, 1 + exp_mid)
        self.lower_intercept[idx4] = - torch.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)

        self.bounds[1, idx4] = torch.upper_slope[idx4] * (bounds[0, idx4] - bounds[1, idx4]) / 2 + torch.div(exp_mid, 1+exp_mid)
        self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        if self.back_subs > 0:
            self.back_substitution(self.back_subs)
        
        return self.bounds


    def _back_sub(self, num_steps, params = None):
        if params is None:
            lower_slope = torch.diag(self.lower_slope)
            upper_slope = torch.diag(self.upper_slope)
            lower_intercept = torch.diag(self.lower_intercept)
            upper_intercept = torch.diag(self.upper_intercept)
        else:
            lower_slope, upper_slope, lower_intercept, upper_intercept = params
        
        if num_steps > 0 and self.prev_layer.prev_layer is not None:
            new_lower_slope = torch.matmul(lower_slope, self.prev_layer.weights)
            new_upper_slope = torch.matmul(upper_slope, self.prev_layer.weights)
            new_lower_intercept = lower_intercept + torch.matmul(lower_slope, self.prev_layer.bias)
            new_upper_intercept = upper_intercept + torch.matmul(upper_slope, self.prev_layer.bias)
            return self.prev_layer.back_substitution(num_steps - 1, params = (new_lower_slope, new_upper_slope, new_lower_intercept, new_upper_intercept))
        else:
            lower = torch.matmul(torch.clamp(lower_slope, min=0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1]) + lower_intercept
            upper = torch.matmul(torch.clamp(upper_slope, min=0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, min=0), self.prev_layer.bounds[0]) + upper_intercept
            return torch.stack([lower, upper], 0)


    def back_substitution(self, num_steps):
        new_bounds = self._back_sub(num_steps).reshape(self.bounds.shape)

        idx_lower = new_bounds[0] > self.bounds[0]
        idx_upper = new_bounds[1] < self.bounds[1]
        self.bounds[0, idx_lower] = new_bounds[0, idx_lower]
        self.bounds[1, idx_upper] = new_bounds[1, idx_upper]





class DeepPolyOutputLayer(nn.Module):
    def __init__(self, true_label, prev_layer=None, back_subs=0):
        super(DeepPolyOutputLayer, self).__init__()
        self.prev_layer = prev_layer
        self.true_label = true_label
        self.back_subs = back_subs
        self.n_labels = self.prev_layer.weights.shape[0]

        self.weights1=torch.zeros((self.n_labels-1,self.n_labels))
        self.weights1[:,self.true_label]-=1
        self.weights1[:self.true_label, :self.true_label] += torch.eye(self.true_label)
        self.weights1[self.true_label:, self.true_label + 1:] += torch.eye(self.n_labels - self.true_label - 1)

        self.W1_plus = torch.clamp(self.weights1, min=0)
        self.W1_minus = torch.clamp(self.weights1, max=0)
        


    def forward(self, bounds):
        upper1 = torch.matmul(self.W1_plus, bounds[1]) + torch.matmul(self.W1_minus, bounds[0])
        lower1 = torch.matmul(self.W1_plus, bounds[0]) + torch.matmul(self.W1_minus, bounds[1])
        self.bounds1 = torch.stack([lower1, upper1], 0)
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)
        return self.bounds1


    def _back_sub(self, max_steps):
        Ml, Mu  = self.weights1, self.weights1

        if max_steps > 0 and self.prev_layer.prev_layer is not None:
            Mlnew = torch.matmul(Ml, self.prev_layer.weights) 
            Munew = torch.matmul(Mu, self.prev_layer.weights) 
            bl1new = torch.matmul(Ml, self.prev_layer.bias)
            bu1new = torch.matmul(Mu, self.prev_layer.bias)
            return self.prev_layer._back_sub(max_steps-1, params=(Mlnew, Munew, bl1new, bu1new))
        else:
            lower1 = torch.matmul(torch.clamp(Ml, min=0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(Ml, max=0), self.prev_layer.bounds[1]) 
            upper1 = torch.matmul(torch.clamp(Mu, min=0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(Mu, max=0), self.prev_layer.bounds[0]) 
            return torch.stack([lower1, upper1], 1)


    def back_substitution(self, num_steps):
        new_bounds = self._back_sub(num_steps)
        indl = new_bounds[0] > self.bounds1[0]
        indu = new_bounds[1] < self.bounds1[1]
        self.bounds1[0, indl] = new_bounds[0,indl]
        self.bounds1[1, indu] = new_bounds[1, indu]

        

def analyze(net, inputs, eps, true_label):
    net.eval()
    d = DeepPoly(net, eps, inputs, true_label, back_sub_steps=20)
    b1 = d.verify()
    # b1 upper bound represents y_false_upper - y_true_low, which should be <=0
    if sum(b1[:,1]>0)==0:
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net',
                        type=str,
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net.endswith('fc1'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net.endswith('fc2'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc3'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net.endswith('fc4'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net.endswith('fc5'):
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
