import argparse
import torch
from code.networks import Normalization
from networks import FullyConnected
from torch import nn

DEVICE = 'cpu'
INPUT_SIZE = 28

class DeepPoly():
    def __init__(self, net, inputs, eps, true_label, back_subs=0):
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

    def back_substitution(self, num_steps, params=None):
        pass
        


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
    def __init__(self, weights, bias:Bool=None, prev_layer=None, back_subs:int=0):
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
        pass 


class DeepPolySPULayer(nn.Module):
    def __init__(self, prev_layer = None, back_subs = 0):
        pass
    def forward(self, bounds):
        pass
    def back_substitution(self, num_steps):
        pass




class DeepPolyOutputLayer(nn.Module):
    def __init__(self, true_label, prev_layer=None, back_subs=0):
        super(DeepPolyOutputLayer, self).__init__()
        self.last = prev_layer
        self.true_label = true_label
        self.back_sub_steps = back_subs
        self.n_labels = self.last.weights.shape[0]
        self._set_weights()
        self.W1_plus = torch.clamp(self.weights1, min=0)
        self.W1_minus = torch.clamp(self.weights1, max=0)
        self.W2_plus = torch.clamp(self.weights1, min=0)
        self.W2_minus = torch.clamp(self.weights1, max=0)


    def forward(self, bounds):
        upper1 = torch.matmul(self.W1_plus, bounds[:,1]) + torch.matmul(self.W1_minus, bounds[:,0])
        lower1 = torch.matmul(self.W1_plus, bounds[:,0]) + torch.matmul(self.W1_minus, bounds[:,1])
        self.bounds1 = torch.stack([lower1, upper1], 1)
        if self.back_sub_steps > 0:
            self.back_sub(self.back_sub_steps)
        return self.bounds1


    def _back_sub(self, max_steps):
        Ml1, Mu1  = self.weights1, self.weights1

        if max_steps > 0 and self.last.last is not None:
            Ml1new = torch.matmul(Ml1, self.last.weights) 
            Mu1new = torch.matmul(Mu1, self.last.weights) 
            bl1new = torch.matmul(Ml1, self.last.bias)
            bu1new = torch.matmul(Mu1, self.last.bias)
            return self.last._back_sub(max_steps-1, params=(Ml1new, Mu1new, bl1new, bu1new))
        else:
            lower1 = torch.matmul(torch.clamp(Ml1, min=0), self.last.bounds[:, 0]) + torch.matmul(torch.clamp(Ml1, max=0), self.last.bounds[:, 1]) 
            upper1 = torch.matmul(torch.clamp(Mu1, min=0), self.last.bounds[:, 1]) + torch.matmul(torch.clamp(Mu1, max=0), self.last.bounds[:, 0]) 
            return torch.stack([lower1, upper1], 1)


    def back_substitution(self, num_steps):
        new_bounds = self._back_sub(num_steps)
        indl = new_bounds[:,0] > self.bounds1[:,0]
        indu = new_bounds[:,1] < self.bounds1[:,1]
        self.bounds1[indl, 0] = new_bounds[indl,0]
        self.bounds1[indu, 1] = new_bounds[indu,1]

        
    def _set_weights(self):
        self.weights1=torch.zeros((self.n_labels-1,self.n_labels))
        self.weights1[:,self.true_label]-=1
        self.weights1[:self.true_label, :self.true_label] += torch.eye(self.true_label)
        self.weights1[self.true_label:, self.true_label + 1:] += torch.eye(self.n_labels - self.true_label - 1)


def analyze(net, inputs, eps, true_label):
    net.eval()
    d = DeepPoly(net, eps, inputs, true_label, back_sub_steps=20, only_first=False)
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
