# RIAI 2021 Course Project



## Descrition

The goal of the proejct is to implement a verifier of a fully connected neural network. According to the particular specification, our network only contains 4 type of layers: `nn.Linear`(affine layers), `nn.Flatten`(tensor stacking), `Normalization` (normalization layer), and `SPU` (activation layer). Therefore, we need to implement corresponding DeepPoly transformers that allows some convex relaxations to pass through. In addition, we also need an input and output layer that transforms the original input into DeepPoly or the DeepPoly output into verification output. 

In the initial code, we have defined a class called `DeepPoly` which directly mirrors the structure of our target network. The class `DeepPoly` contains the following 6 different layers (all using `nn.Module`):

1. `DeepPolyInputLayer`
This layer takes an image and a bound $`\epsilon`$ as an input. It tranforms the image into its DeepPoly representation of the $`\epsilon`$-ball in $`L^\infty$` space.

2. `DeepPolyAffineLayer`
This layer performs affine transformation on the convex relaxation. The specific implementation is the same as that seen in class. More specific in terms of matrix multiplication, its new transformed upper-bound can be specified as the positive entries of weight matrix `w` multiplied with upper-bound plus negative entries multiplied with lower-bound. The lower bound can be defined vice versa. 

3. `DeepPolyNormalizer`
The normalizer is essentially normalizing (scaling and shifting) the input data. Hence, we only need to perform the exact same scaling and shifting on our bounds.

4. `DeepPolyOutputLayer`
Given a DeepPoly represenatation as the output of the network, we use this final layer to see if we can still get the same output under perturbation. The essence is to check if the output-class has a lower-bound that is higher than all other upper-bounds.

5. `DeepPolyFalttenLayer`
All this method does is to simply transform the shape of the input. Relatively easily we can perform the same operation on the bounds.

6. `DeepPolySPU`
This layer performs transformation of DeepPoly through SPU activation. The specific formalation is calculated and is proved to have the minimal area among all DeepPolies (under the constraint that they are tangent lines or secant lines). 
