import argparse
import torch
# from code.networks import Normalization
from networks import FullyConnected
from networks import *
from torch import nn
import logging, sys
import pdb

STEP = 0
STEP_SIGN = 1
STEP_SIZE = 0.1
DEVICE = 'cpu'
INPUT_SIZE = 28
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

class DeepPoly():
    def __init__(self, net, inputs, eps, true_label, back_subs=0, useBound=1):
        # assert eps > 0
        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label
        self.back_subs = back_subs
        
        self.layers = [DeepPolyInputLayer(self.eps)]
        prev_layer = None
        for layer in self.net.layers:
            if isinstance(layer, torch.nn.Linear):
                prev_layer = DeepPolyAffineLayer(weights=layer._parameters['weight'].detach(), bias=layer._parameters['bias'].detach(), prev_layer=prev_layer, back_subs=self.back_subs)
                self.layers.append(prev_layer)
            elif isinstance(layer, torch.nn.Flatten):
                prev_layer = DeepPolyFlattenLayer(prev_layer=prev_layer)
                self.layers.append(prev_layer)
            elif isinstance(layer, Normalization):
                prev_layer = DeepPolyNormalizeLayer(prev_layer=prev_layer)
                self.layers.append(prev_layer)
            elif isinstance(layer, SPU):
                if useBound == 1:
                    prev_layer = DeepPolySPULayer(prev_layer=prev_layer, back_subs=self.back_subs)
                elif useBound == 2:
                    prev_layer = DeepPolySPULayerBound2(prev_layer=prev_layer, back_subs=self.back_subs)
                else:
                    prev_layer = DeepPolySPULayerBound3(prev_layer=prev_layer, back_subs=self.back_subs)
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
        
        # pdb.set_trace()
        logging.debug("Now in Input Fowrad")
        lower = torch.clamp(input - self.eps, 0., 1.)
        upper = torch.clamp(input + self.eps, 0., 1.)
        self.bounds = torch.stack([lower, upper], 0)
        return self.bounds


class DeepPolyFlattenLayer(nn.Module):
    def __init__(self, prev_layer=None):
        super(DeepPolyFlattenLayer, self).__init__()
        self.prev_layer = prev_layer

    def forward(self, bounds):
        logging.debug("Now in Flatten forward")
        return torch.stack([bounds[0].flatten(), bounds[1].flatten()], 0)

    def _back_sub(self, num_steps, params=None):
        logging.debug("Now in Flatten _back_sub")
        bounds = self.prev_layer._back_sub(num_steps, params=params)
        bounds = torch.stack([bounds[:int(len(bounds)/2)], bounds[int(len(bounds)/2)]], dim=0)
        return bounds
    
    '''These codes are necessary as a property here to get consistency of function calls in backpropagation.'''
    @property
    def lower_intercept(self):
        return self.prev_layer.lower_intercept.flatten()

    @property
    def upper_intercept(self):
        return self.prev_layer.upper_intercept.flatten()

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
        logging.debug("Now in Normalize forward")
        self.bounds = torch.div(bounds - self.mean, self.variance)
        return self.bounds



class DeepPolyAffineLayer(nn.Module):
    def __init__(self, weights, bias:bool=None, prev_layer=None, back_subs:int=0):
        super(DeepPolyAffineLayer, self).__init__()
        self.weights = weights
        self.bias = bias
        self.prev_layer = prev_layer
        self.back_subs = back_subs

        self.weight_positive = torch.clip(self.weights, min = 0.)
        self.weight_negative = torch.clip(self.weights, max = 0.)

    def forward(self, bounds):
        logging.debug("Now in Affine forward")
        upper = torch.matmul(self.weight_positive, bounds[1, :]) + torch.matmul(self.weight_negative, bounds[0, :])
        lower = torch.matmul(self.weight_positive, bounds[0, :]) + torch.matmul(self.weight_negative, bounds[1, :])
        self.bounds = torch.stack([lower, upper], 0)
        if self.bias is not None:
            self.bounds += self.bias.reshape(1, -1)
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)
        return self.bounds

    def back_substitution(self, num_steps):
        logging.debug("Now in Affine back_substitution")
        new_bounds = self._back_sub(num_steps)
        idx_lower = new_bounds[0] > self.bounds[0]
        idx_upper = new_bounds[1] < self.bounds[1]
        self.bounds[0, idx_lower] = new_bounds[0, idx_lower]
        self.bounds[1, idx_upper] = new_bounds[1, idx_upper]
        
    def _back_sub(self, num_steps, params = None):
        logging.debug("Now in Affine _back_sub")
        if params is None:
            lower_slope, upper_slope, lower_intercept, upper_intercept = self.weights, self.weights, self.bias, self.bias
        else:
            lower_slope, upper_slope, lower_intercept, upper_intercept = params
        # lower_slope, upper_slope, lower_intercept, upper_intercept = self.weights, self.weights, self.bias, self.bias if params is None else params
        # import pdb
        # pdb.set_trace()
        if num_steps > 0 and self.prev_layer.prev_layer.prev_layer is not None:
            new_lower_slope = torch.clamp(lower_slope, min = 0) * self.prev_layer.lower_slope + torch.clamp(lower_slope, max = 0) * self.prev_layer.upper_slope
            new_upper_slope = torch.clamp(upper_slope, min = 0) * self.prev_layer.upper_slope + torch.clamp(upper_slope, max = 0) * self.prev_layer.lower_slope
            new_lower_intercept = lower_intercept + torch.matmul(torch.clamp(lower_slope, max = 0), self.prev_layer.upper_intercept) +\
                 torch.matmul(torch.clamp(lower_slope, min = 0), self.prev_layer.lower_intercept)
            new_upper_intercept = upper_intercept + torch.matmul(torch.clamp(upper_slope, max = 0), self.prev_layer.lower_intercept) +\
                 torch.matmul(torch.clamp(upper_slope, min = 0), self.prev_layer.upper_intercept)
            return self.prev_layer._back_sub(num_steps - 1, params = (new_lower_slope, new_upper_slope, new_lower_intercept, new_upper_intercept))

        else:
            if lower_intercept is not None:
                lower = torch.matmul(torch.clamp(lower_slope, min = 0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1]) + lower_intercept
            else:
                lower = torch.matmul(torch.clamp(lower_slope, min = 0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1])
            if upper_intercept is not None:
                upper = torch.matmul(torch.clamp(upper_slope, min = 0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, max=0), self.prev_layer.bounds[0]) + upper_intercept
            else:
                upper = torch.matmul(torch.clamp(upper_slope, min = 0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, max=0), self.prev_layer.bounds[0])
            return torch.stack([lower, upper], 0)




'''
SPU layer for Bound 1
'''
class DeepPolySPULayer(nn.Module):
    def __init__(self, prev_layer = None, back_subs = 0):
        super(DeepPolySPULayer, self).__init__()
        self.prev_layer = prev_layer
        self.back_subs = back_subs

    def forward(self, bounds):
        logging.debug("Now in SPU forward")
        # Index for all positive case
        idx1 = bounds[0] >= 0
        # Index for cross boundary case
        idx2 = (bounds[1]>=0) * (bounds[0]<0)
        # Index for cross boundary with upperbound greater than lower bound
        idx3 = (bounds[1] >= - bounds[0]) * idx2
        # Index for all negative case
        idx4 = bounds[1] < 0

        # Initiate slope of upper bound, lower bound and intercept
        self.upper_slope = torch.zeros_like(bounds[1])
        self.lower_slope = torch.zeros_like(bounds[1])
        self.upper_intercept = torch.zeros_like(bounds[1])
        self.lower_intercept = torch.zeros_like(bounds[1])

        self.bounds = torch.zeros_like(bounds)

        # All positive case
        # self.bounds[1, idx1] = bounds[1, idx1]**2 - 0.5
        # self.bounds[0, idx1] = (bounds[0, idx1] + bounds[1, idx1]) * (3 * bounds[0, idx1] - bounds[1, idx1]) / 4 - 0.5
        # self.upper_slope[idx1] = bounds[1, idx1] + bounds[0, idx1]
        # self.lower_slope[idx1] = bounds[1, idx1] + bounds[0, idx1] 
        # self.upper_intercept[idx1] = -bounds[1,idx1] * bounds[0, idx1] - 0.5
        # self.lower_intercept[idx1] = -0.25 * (bounds[1, idx1] + bounds[0,idx1])**2 - 0.5
        # upper bound no change
        self.bounds[1, idx1] = bounds[1, idx1]**2 - 0.5
        self.upper_slope[idx1] = bounds[1, idx1] + bounds[0, idx1]
        self.upper_intercept[idx1] = -bounds[1,idx1] * bounds[0, idx1] - 0.5
        # lower bound use step
        self.lower_slope[idx1] = bounds[1, idx1] + bounds[0, idx1] + 2 * STEP
        self.lower_intercept[idx1] = -(bounds[0, idx1] + bounds[1, idx1])**2/4 -(bounds[0, idx1] + bounds[1, idx1]) * STEP - STEP**2 -0.5
        self.bounds[0, idx1] = self.lower_slope[idx1] * bounds[0, idx1] + self.lower_intercept[idx1]


        # Cross boundary case (upper strictly smaller than lower)
        exp_l = torch.exp(bounds[0, idx2])
        slope1 = - torch.div( exp_l, (1 + exp_l)**2 )
        slope2 = torch.div(bounds[1, idx2]**2 - 0.5 + torch.div( exp_l, 1 + exp_l ), bounds[1, idx2] - bounds[0, idx2])
        temp_idx = slope1 > slope2
        slope2[temp_idx] = slope1[temp_idx]
        self.upper_slope[idx2] = slope2
        self.upper_intercept[idx2] = - slope2 * bounds[0, idx2] - torch.div(exp_l, 1 + exp_l)

        self.lower_slope[idx2] = torch.div(-0.5 + torch.div(exp_l, 1+exp_l), -bounds[0, idx2]) 
        self.lower_intercept[idx2] = -0.5 * torch.ones_like(bounds[0, idx2])

        self.bounds[0, idx2] = torch.div(-0.5 * bounds[1, idx2] + torch.div(bounds[1, idx2] * exp_l, 1 + exp_l), -bounds[0, idx2]) - 0.5
        upper1 = - torch.div(exp_l, 1 + exp_l)
        upper2 = slope2 * (bounds[1, idx2] - bounds[0, idx2]) + upper1
        temp_idx = upper1 > upper2
        upper2[temp_idx] = upper1[temp_idx]
        self.bounds[1, idx2] = upper2

        # All negative case
        mid = (bounds[1, idx4] + bounds[0, idx4]) / 2
        exp_mid = torch.exp(mid)
        exp_u = torch.exp(bounds[1, idx4])
        exp_l = torch.exp(bounds[0, idx4])
        self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])

        self.upper_intercept[idx4] = - self.upper_slope[idx4] * mid - torch.div(exp_mid, 1 + exp_mid)
        self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)

        self.bounds[1, idx4] = self.upper_slope[idx4] * (bounds[0, idx4] - bounds[1, idx4]) / 2 - torch.div(exp_mid, 1+exp_mid)
        self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        # add STEP to upper line but unsound occur
        # mid = (bounds[1, idx4] + bounds[0, idx4]) / 2 + STEP
        # exp_mid = torch.exp(mid)

        # self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        # self.upper_intercept[idx4] = - self.upper_slope[idx4] * mid - torch.div(exp_mid, 1 + exp_mid)
        # self.bounds[1, idx4] = self.upper_slope[idx4] * bounds[0, idx4] + self.upper_intercept[idx4]
        # # lower bound no change
        # exp_u = torch.exp(bounds[1, idx4])
        # exp_l = torch.exp(bounds[0, idx4])
        # self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])
        # self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)
        # self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        '''These code are for testing, we are using more tighter upper/lower bound without changing the slope and intercept'''
        # import pdb
        # pdb.set_trace()
        self.bounds[0, idx1] = bounds[0, idx1]**2 - 0.5
        # self.bounds[0, idx2] = -0.5 * torch.ones_like(bounds[0, idx2])
        exp_l = torch.exp(bounds[0, idx4])
        self.bounds[1, idx4] = - torch.div(exp_l, 1+ exp_l)

        # tighter lower bound
        self.bounds[0] = torch.clamp(self.bounds[0], min=-0.5)

        logging.debug("Now in SPU forward")
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)

        return self.bounds

    def _back_sub(self, num_steps, params = None):
        logging.debug("Now in SPU _back_sub")
        if params is None:
            lower_slope = torch.diag(self.lower_slope)
            upper_slope = torch.diag(self.upper_slope)
            lower_intercept = self.lower_intercept
            upper_intercept = self.upper_intercept
        else:
            lower_slope, upper_slope, lower_intercept, upper_intercept = params

        if num_steps > 0 and self.prev_layer.prev_layer is not None:
            new_lower_slope = torch.matmul(lower_slope, self.prev_layer.weights)
            new_upper_slope = torch.matmul(upper_slope, self.prev_layer.weights)
            new_lower_intercept = lower_intercept + torch.matmul(lower_slope, self.prev_layer.bias)
            new_upper_intercept = upper_intercept + torch.matmul(upper_slope, self.prev_layer.bias)
            return self.prev_layer._back_sub(num_steps - 1, params = (new_lower_slope, new_upper_slope, new_lower_intercept, new_upper_intercept))

        else:
            lower = torch.matmul(torch.clamp(lower_slope, min=0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1]) + lower_intercept
            upper = torch.matmul(torch.clamp(upper_slope, min=0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, max=0), self.prev_layer.bounds[0]) + upper_intercept
            return torch.stack([lower, upper], 0)


    def back_substitution(self, num_steps):
        logging.debug("Now in SPU back_substitution")
        new_bounds = self._back_sub(num_steps).reshape(self.bounds.shape)
        idx_lower = new_bounds[0] > self.bounds[0]
        idx_upper = new_bounds[1] < self.bounds[1]
        self.bounds[0, idx_lower] = new_bounds[0, idx_lower]
        self.bounds[1, idx_upper] = new_bounds[1, idx_upper]


'''
SPU layer for Bound 2
'''
class DeepPolySPULayerBound2(DeepPolySPULayer):

    def __init__(self, prev_layer = None, back_subs = 0):
        super(DeepPolySPULayerBound2, self).__init__(prev_layer, back_subs)

    def forward(self, bounds):
        logging.debug("Now in SPU forward")
        # Index for all positive case
        idx1 = bounds[0] >= 0
        # Index for cross boundary case
        idx2 = (bounds[1]>=0) * (bounds[0]<0)
        # Index for cross boundary with upperbound greater than lower bound
        idx3 = (bounds[1] >= - bounds[0]) * idx2
        # Index for all negative case
        idx4 = bounds[1] < 0

        # Initiate slope of upper bound, lower bound and intercept
        self.upper_slope = torch.zeros_like(bounds[1])
        self.lower_slope = torch.zeros_like(bounds[1])
        self.upper_intercept = torch.zeros_like(bounds[1])
        self.lower_intercept = torch.zeros_like(bounds[1])

        self.bounds = torch.zeros_like(bounds)

        # All positive case
        # upper bound no change
        self.bounds[1, idx1] = bounds[1, idx1]**2 - 0.5
        self.upper_slope[idx1] = bounds[1, idx1] + bounds[0, idx1]
        self.upper_intercept[idx1] = -bounds[1,idx1] * bounds[0, idx1] - 0.5
        # lower bound use step
        self.lower_slope[idx1] = bounds[1, idx1] + bounds[0, idx1] + 2 * STEP
        self.lower_intercept[idx1] = -(bounds[0, idx1] + bounds[1, idx1])**2/4 -(bounds[0, idx1] + bounds[1, idx1]) * STEP - STEP**2 -0.5
        self.bounds[0, idx1] = self.lower_slope[idx1] * bounds[0, idx1] + self.lower_intercept[idx1]

        # all case 2 (upper greater than lower)
        exp_l = torch.exp(bounds[0, idx2])
        slope1 = - torch.div( exp_l, (1 + exp_l)**2 )
        slope2 = torch.div(bounds[1, idx2]**2 - 0.5 + torch.div( exp_l, 1 + exp_l ), bounds[1, idx2] - bounds[0, idx2])
        temp_idx = slope1 > slope2
        slope2[temp_idx] = slope1[temp_idx]
        self.upper_slope[idx2] = slope2
        self.upper_intercept[idx2] = - slope2 * bounds[0, idx2] - torch.div(exp_l, 1 + exp_l)

        upper1 = - torch.div(exp_l, 1 + exp_l)
        upper2 = slope2 * (bounds[1, idx2] - bounds[0, idx2]) + upper1
        temp_idx = upper1 > upper2
        upper2[temp_idx] = upper1[temp_idx]
        self.bounds[1, idx2] = upper2
        self.bounds[0, idx2] = torch.minimum((bounds[0, idx2] + bounds[1, idx2]) * (3 * bounds[0, idx2] - bounds[1, idx2]) / 4 - 0.5, (bounds[1, idx2] + bounds[0, idx2]) * (3 * bounds[1, idx2] - bounds[0, idx2]) / 4 - 0.5)
        # exp_l = torch.exp(bounds[0, idx2])
        # upper1 = (bounds[1, idx2])**2 - 0.5
        # upper2 = torch.div(-exp_l, 1 + exp_l)
        # temp_idx = upper1 > upper2
        # upper2[temp_idx] = upper1[temp_idx]
        # self.bounds[1, idx2] = upper2
        lower1 = bounds[1, idx2] + bounds[0, idx2]
        lower2 = torch.div(-0.5 + torch.div(exp_l, 1+exp_l), -bounds[0, idx2]) 
        temp_idx = lower1 > lower2
        lower2[temp_idx] = lower1[temp_idx]
        b1 = - 0.25 * (bounds[1, idx2] + bounds[0, idx2])**2 - 0.5
        b2 = -0.5 * torch.ones_like(bounds[0, idx2])
        b2[temp_idx] = b1[temp_idx]
        self.lower_slope[idx2] = lower2
        self.lower_intercept[idx2] = b2

        # Here should also use the upper slope and intercept as bound1 
        # self.upper_slope[idx2] = torch.div(bounds[1, idx2]**2 - 0.5 + torch.div(exp_l, 1 + exp_l), bounds[1, idx2] - bounds[0, idx2])
        # self.upper_intercept[idx2] = - bounds[0, idx2] * self.upper_slope[idx2] - torch.div(exp_l, 1+ exp_l)
        

        # All negative case
        mid = (bounds[1, idx4] + bounds[0, idx4]) / 2
        exp_mid = torch.exp(mid)
        exp_u = torch.exp(bounds[1, idx4])
        exp_l = torch.exp(bounds[0, idx4])
        self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])

        self.upper_intercept[idx4] = - self.upper_slope[idx4] * mid - torch.div(exp_mid, 1 + exp_mid)
        self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)

        self.bounds[1, idx4] = self.upper_slope[idx4] * (bounds[0, idx4] - bounds[1, idx4]) / 2 - torch.div(exp_mid, 1+exp_mid)
        self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        # add STEP to upper line but unsound occur
        # mid = (bounds[1, idx4] + bounds[0, idx4]) / 2 + STEP
        # exp_mid = torch.exp(mid)

        # self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        # self.upper_intercept[idx4] = - self.upper_slope[idx4] * mid - torch.div(exp_mid, 1 + exp_mid)
        # self.bounds[1, idx4] = self.upper_slope[idx4] * bounds[0, idx4] + self.upper_intercept[idx4]
        # # lower bound no change
        # exp_u = torch.exp(bounds[1, idx4])
        # exp_l = torch.exp(bounds[0, idx4])
        # self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])
        # self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)
        # self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        '''These code are for testing, we are using more tighter upper/lower bound without changing the slope and intercept'''
        # import pdb
        # pdb.set_trace()
        self.bounds[0, idx1] = bounds[0, idx1]**2 - 0.5
        # self.bounds[0, idx2] = -0.5 * torch.ones_like(bounds[0, idx2])
        exp_l = torch.exp(bounds[0, idx4])
        self.bounds[1, idx4] = - torch.div(exp_l, 1+ exp_l)


        # tighter lower bound
        self.bounds[0] = torch.clamp(self.bounds[0], min=-0.5)

        logging.debug("Now in SPU forward")
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)
        # import pdb
        # pdb.set_trace()
        return self.bounds


'''
SPU layer for Bound 3
'''
class DeepPolySPULayerBound3(DeepPolySPULayer):

    def __init__(self, prev_layer = None, back_subs = 0):
        super(DeepPolySPULayerBound3, self).__init__(prev_layer, back_subs)

    def forward(self, bounds):
        logging.debug("Now in SPU forward")
        # Index for all positive case
        idx1 = bounds[0] >= 0
        # Index for cross boundary case
        idx2 = (bounds[1]>=0) * (bounds[0]<0)
        # Index for cross boundary with upperbound greater than lower bound
        idx3 = (bounds[1] >= - bounds[0]) * idx2
        # Index for all negative case
        idx4 = bounds[1] < 0

        # Initiate slope of upper bound, lower bound and intercept
        self.upper_slope = torch.zeros_like(bounds[1])
        self.lower_slope = torch.zeros_like(bounds[1])
        self.upper_intercept = torch.zeros_like(bounds[1])
        self.lower_intercept = torch.zeros_like(bounds[1])

        self.bounds = torch.zeros_like(bounds)

        # All positive case
        # box here not use STEP, stick with box
        self.bounds[1, idx1] = bounds[1, idx1]**2 - 0.5
        self.bounds[0, idx1] = bounds[0, idx1]**2 - 0.5
        self.upper_slope[idx1] = bounds[1, idx1] + bounds[0, idx1]
        self.lower_slope[idx1] = torch.zeros_like(bounds[0, idx1])
        self.upper_intercept[idx1] = -bounds[1,idx1] * bounds[0, idx1] - 0.5
        self.lower_intercept[idx1] = bounds[0, idx1]**2 - 0.5

        # Cross boundary case (upper strictly smaller than lower)
        exp_l = torch.exp(bounds[0, idx2])
        slope1 = - torch.div( exp_l, (1 + exp_l)**2 )
        slope2 = torch.div(bounds[1, idx2]**2 - 0.5 + torch.div( exp_l, 1 + exp_l ), bounds[1, idx2] - bounds[0, idx2])
        temp_idx = slope1 > slope2
        slope2[temp_idx] = slope1[temp_idx]
        self.upper_slope[idx2] = slope2
        self.upper_intercept[idx2] = - slope2 * bounds[0, idx2] - torch.div(exp_l, 1 + exp_l)

        upper1 = - torch.div(exp_l, 1 + exp_l)
        upper2 = slope2 * (bounds[1, idx2] - bounds[0, idx2]) + upper1
        temp_idx = upper1 > upper2
        upper2[temp_idx] = upper1[temp_idx]
        self.bounds[1, idx2] = upper2

        self.lower_slope[idx2] = torch.zeros_like(bounds[0, idx2])
        self.lower_intercept[idx2] = -0.5 * torch.ones_like(bounds[0, idx2])
        self.bounds[0, idx2] = -0.5 * torch.ones_like(bounds[0, idx2])


        # All negative case
        # box here not use STEP, stick with box
        mid = (bounds[1, idx4] + bounds[0, idx4]) / 2
        exp_mid = torch.exp(mid)
        exp_u = torch.exp(bounds[1, idx4])
        exp_l = torch.exp(bounds[0, idx4])
        self.upper_slope[idx4] = torch.zeros_like(bounds[0, idx4])
        self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])

        self.upper_intercept[idx4] = - torch.div(exp_l, 1 + exp_l)
        self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)

        self.bounds[1, idx4] = - torch.div(exp_l, 1 + exp_l)
        self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)
        # mid = (bounds[1, idx4] + bounds[0, idx4]) / 2 + STEP
        # exp_mid = torch.exp(mid)

        # self.upper_slope[idx4] = torch.div(-exp_mid, (1 + exp_mid)**2)
        # self.upper_intercept[idx4] = - self.upper_slope[idx4] * mid - torch.div(exp_mid, 1 + exp_mid)
        # self.bounds[1, idx4] = self.upper_slope[idx4] * bounds[0, idx4] + self.upper_intercept[idx4]
        # # lower bound no change
        # exp_u = torch.exp(bounds[1, idx4])
        # exp_l = torch.exp(bounds[0, idx4])
        # self.lower_slope[idx4] = torch.div(-torch.div(exp_u, 1+ exp_u) + torch.div(exp_l, 1+exp_l), bounds[1, idx4] - bounds[0, idx4])
        # self.lower_intercept[idx4] = - self.lower_slope[idx4] * bounds[0, idx4] - torch.div(exp_l, 1+exp_l)
        # self.bounds[0, idx4] = - torch.div(exp_u, 1 + exp_u)

        '''These code are for testing, we are using more tighter upper/lower bound without changing the slope and intercept'''
        # import pdb
        # pdb.set_trace()
        self.bounds[0, idx1] = bounds[0, idx1]**2 - 0.5
        # self.bounds[0, idx2] = -0.5 * torch.ones_like(bounds[0, idx2])
        exp_l = torch.exp(bounds[0, idx4])
        self.bounds[1, idx4] = - torch.div(exp_l, 1+ exp_l)


        # tighter lower bound
        self.bounds[0] = torch.clamp(self.bounds[0], min=-0.5)

        logging.debug("Now in SPU forward")
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)

        return self.bounds



class DeepPolyOutputLayer(nn.Module):
    def __init__(self, true_label, prev_layer=None, back_subs=0):
        super(DeepPolyOutputLayer, self).__init__()
        self.true_label = true_label
        self.prev_layer = prev_layer
        self.back_subs = back_subs
        self.num_label = self.prev_layer.weights.shape[0]

        self.weights1=torch.zeros((self.num_label-1,self.num_label))
        self.weights1[:,self.true_label]-=1
        self.weights1[:self.true_label, :self.true_label] += torch.eye(self.true_label)
        self.weights1[self.true_label:, self.true_label + 1:] += torch.eye(self.num_label - self.true_label - 1)

        self.weights_postive = torch.clamp(self.weights1, min=0)
        self.wrights_negative = torch.clamp(self.weights1, max=0)
        

    def forward(self, bounds):
        logging.debug("Now in Output forward")
        upper = torch.matmul(self.weights_postive, bounds[1]) + torch.matmul(self.wrights_negative, bounds[0])
        lower = torch.matmul(self.weights_postive, bounds[0]) + torch.matmul(self.wrights_negative, bounds[1])
        self.bounds_new = torch.stack([lower, upper], 0)
        # import pdb
        # pdb.set_trace()
        if self.back_subs > 0:
            self.back_substitution(self.back_subs)
        return self.bounds_new

    def _back_sub(self, max_steps):
        logging.debug("Now in Output _back_sub")
        lower_slope, upper_slope  = self.weights1, self.weights1

        
        if max_steps > 0 and self.prev_layer.prev_layer is not None:
            new_lower_slope = torch.matmul(lower_slope, self.prev_layer.weights) 
            new_upper_slope = torch.matmul(upper_slope, self.prev_layer.weights) 
            new_lower_intercept = torch.matmul(lower_slope, self.prev_layer.bias)
            new_upper_intercept = torch.matmul(upper_slope, self.prev_layer.bias)
            return self.prev_layer._back_sub(max_steps-1, params=(new_lower_slope, new_upper_slope, new_lower_intercept, new_upper_intercept))
        else:
            lower1 = torch.matmul(torch.clamp(lower_slope, min=0), self.prev_layer.bounds[0]) + torch.matmul(torch.clamp(lower_slope, max=0), self.prev_layer.bounds[1]) 
            upper1 = torch.matmul(torch.clamp(upper_slope, min=0), self.prev_layer.bounds[1]) + torch.matmul(torch.clamp(upper_slope, max=0), self.prev_layer.bounds[0]) 
            return torch.stack([lower1, upper1], 0)

    def back_substitution(self, num_steps):
        logging.debug("Now in SPU back_subustitution")
        new_bounds = self._back_sub(num_steps)
        indl = new_bounds[0] > self.bounds_new[0]
        indu = new_bounds[1] < self.bounds_new[1]
        # import pdb
        # pdb.set_trace()
        self.bounds_new[0, indl] = new_bounds[0,indl]
        self.bounds_new[1, indu] = new_bounds[1, indu]

        

def analyze(net, inputs, eps, true_label):
    import time
    start = time.perf_counter()
    net.eval()
    # import pdb
    # pdb.set_trace()
    # b1 upper bound represents y_false_upper - y_true_low, which should be < 0
    global STEP
    global STEP_SIGN
    STEP = 0
    STEP_SIGN = 1
    while(time.perf_counter() - start < 60):
        d1 = DeepPoly(net, inputs, eps, true_label, 20, 1)
        b1 = d1.verify()
        d2 = DeepPoly(net, inputs, eps, true_label, 20, 2)
        b2 = d2.verify()
        d3 = DeepPoly(net, inputs, eps, true_label, 20, 3)
        b3 = d3.verify()
        if sum(b1[1]>=0)==0 or sum(b2[1]>=0)==0 or sum(b3[1]>=0)==0:
        # if sum(b1[1]>=0)==0 or sum(b2[1]>=0)==0:
        # if sum(b2[1]>=0)==0:
            return True
        else:
            STEP = (STEP + STEP_SIGN*STEP_SIZE) * (-1)
            STEP_SIGN = STEP_SIGN * (-1)
            # print(STEP)
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

    # pdb.set_trace()
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
