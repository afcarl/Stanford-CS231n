import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #input data dims
    (C, H, W) = input_dim
    #conv params
    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    #max pool params
    pool_height = 2
    pool_width = 2
    pool_stride = 2

    #infer output dims of intermediate layers
    H_output_conv = 1 + (H + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv = 1 + (W + 2 * conv_pad - filter_size) / conv_stride
    H_output_pool = 1 + (H_output_conv  - pool_height) / pool_stride
    W_output_pool = 1 + (W_output_conv  - pool_width) / pool_stride

    #conv param init
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    #1st affine init
    self.params['W2'] = weight_scale * np.random.randn(num_filters*H_output_pool*W_output_pool, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)

    #2nd affine init
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #conv-relu-pool layer output
    crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #affine-relu layer output
    ar_out, ar_cache = affine_relu_forward(crp_out, W2, b2)
    #last affine output
    affine_out, affine_cache = affine_forward(ar_out, W3, b3)
    scores = affine_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, daffine_out = softmax_loss(scores, y)
    #add regularization loss
    loss += 0.5*self.reg*np.sum(self.params['W1']**2) + 0.5*self.reg*np.sum(self.params['W2']**2) + 0.5*self.reg*np.sum(self.params['W3']**2)

    #grads
    dloss = 1.0
    daffine_out = daffine_out*dloss
    dar_out, dW3, db3 = affine_backward(daffine_out, affine_cache)
    dcrp_out, dW2, db2 = affine_relu_backward(dar_out, ar_cache)
    dX, dW1, db1 = conv_relu_pool_backward(dcrp_out, crp_cache)

    #add  regulalization to gradients
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['W1'] = dW1 + self.reg*self.params['W1']
    grads['W2'] = dW2 + self.reg*self.params['W2']
    grads['W3'] = dW3 + self.reg*self.params['W3']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class FiveLayerConvNet(object):
  """
  A five-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - conv - relu - [affine - bn - relu]x2 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype

    #input data dims
    (C, H, W) = input_dim
    #conv params
    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    #max pool params
    pool_height = 2
    pool_width = 2
    pool_stride = 2

    #infer output dims of intermediate layers
    H_output_conv1 = 1 + (H + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv1 = 1 + (W + 2 * conv_pad - filter_size) / conv_stride
    H_output_pool1 = 1 + (H_output_conv1  - pool_height) / pool_stride
    W_output_pool1 = 1 + (W_output_conv1  - pool_width) / pool_stride

    H_output_conv2 = 1 + (H_output_pool1 + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv2 = 1 + (W_output_pool1 + 2 * conv_pad - filter_size) / conv_stride

    #conv1 param init
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    #conv2 param init
    self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filters)

    #1st affine init
    self.params['W3'] = weight_scale * np.random.randn(num_filters*H_output_conv2*W_output_conv2, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['gamma3'] = np.ones(hidden_dim)
    self.params['beta3'] = np.zeros(hidden_dim)

    #2nd affine init
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
    self.params['b4'] = np.zeros(hidden_dim)
    self.params['gamma4'] = np.ones(hidden_dim)
    self.params['beta4'] = np.zeros(hidden_dim)

    #3nd affine init
    self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b5'] = np.zeros(num_classes)

    #init bn_params (bn_params[layer_number] = bn params related to BN stage in this layer)
    self.bn_params = {}
    self.bn_params[3] = {'mode': 'train'}
    self.bn_params[4] = {'mode': 'train'}
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)     

  def affine_bn_relu_forward(self, x, w, b, gamma, beta, bn_param):
    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)
    return out, cache

  def affine_bn_relu_backward(self, dout, cache):
    affine_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)
    dx, dw, db = affine_backward(da, affine_cache)
    return dx, dw, db, dgamma, dbeta

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the five-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    W5, b5 = self.params['W5'], self.params['b5']


    #determine mode, for bn_params
    mode = 'test' if y is None else 'train'
    for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #conv1-relu1-pool1 layer output (layer 1)
    crp1_out, crp1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #conv2-relu2 layer output (layer 2)
    cr2_out, cr2_cache = conv_relu_forward(crp1_out, W2, b2, conv_param)

    #affine1-bn1-relu1 layer output (layer 3)
    abr1_out, abr1_cache = self.affine_bn_relu_forward(cr2_out, W3, b3, gamma3, beta3, self.bn_params[3])
    #affine1-bn1-relu1 layer output (layer 4)
    abr2_out, abr2_cache = self.affine_bn_relu_forward(abr1_out, W4, b4, gamma4, beta4, self.bn_params[4])

    #last affine output (layer 5)
    affine_out, affine_cache = affine_forward(abr2_out, W5, b5)
    scores = affine_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, daffine_out = softmax_loss(scores, y)
    #add regularization loss
    loss += (0.5*self.reg*np.sum(self.params['W1']**2) + 0.5*self.reg*np.sum(self.params['W2']**2) + 0.5*self.reg*np.sum(self.params['W3']**2)
        + 0.5*self.reg*np.sum(self.params['W4']**2) + 0.5*self.reg*np.sum(self.params['W5']**2))

    #grads
    dloss = 1.0
    daffine_out = daffine_out*dloss
    #grad layer 5
    dabr2_out, dW5, db5 = affine_backward(daffine_out, affine_cache)
    #grad layer 4
    dabr1_out, dW4, db4, dgamma4, dbeta4 = self.affine_bn_relu_backward(dabr2_out, abr2_cache)
    #grad layer 3
    dcr2_out, dW3, db3, dgamma3, dbeta3 = self.affine_bn_relu_backward(dabr1_out, abr1_cache)
    #grad layer 2
    dcrp1_out, dW2, db2 = conv_relu_backward(dcr2_out, cr2_cache)
    #grad layer 1
    dX, dW1, db1 = conv_relu_pool_backward(dcrp1_out, crp1_cache)

    #add  regulalization to gradients
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['W1'] = dW1 + self.reg*self.params['W1']
    grads['W2'] = dW2 + self.reg*self.params['W2']
    grads['W3'] = dW3 + self.reg*self.params['W3']
    grads['W4'] = dW4 + self.reg*self.params['W4']
    grads['W5'] = dW5 + self.reg*self.params['W5']
    grads["gamma3"] = dgamma3
    grads["gamma4"] = dgamma4
    grads["beta3"] = dbeta3
    grads["beta4"] = dbeta4
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass

class SixLayerConvNet(object):
  """
  A six-layer convolutional network with the following architecture:
  
  [conv - relu - 2x2 max pool]x2 - conv - relu - [affine - bn - relu]x2 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype

    #input data dims
    (C, H, W) = input_dim
    #conv params
    conv_stride = 1
    conv_pad = (filter_size - 1) / 2
    #max pool params
    pool_height = 2
    pool_width = 2
    pool_stride = 2

    #infer output dims of intermediate layers
    H_output_conv1 = 1 + (H + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv1 = 1 + (W + 2 * conv_pad - filter_size) / conv_stride
    H_output_pool1 = 1 + (H_output_conv1  - pool_height) / pool_stride
    W_output_pool1 = 1 + (W_output_conv1  - pool_width) / pool_stride

    H_output_conv2 = 1 + (H_output_pool1 + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv2 = 1 + (W_output_pool1 + 2 * conv_pad - filter_size) / conv_stride
    H_output_pool2 = 1 + (H_output_conv2  - pool_height) / pool_stride
    W_output_pool2 = 1 + (W_output_conv2  - pool_width) / pool_stride

    H_output_conv3 = 1 + (H_output_pool2 + 2 * conv_pad - filter_size) / conv_stride
    W_output_conv3 = 1 + (W_output_pool2 + 2 * conv_pad - filter_size) / conv_stride

    #conv1 param init
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    #conv2 param init
    self.params['W2'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b2'] = np.zeros(num_filters)
    #conv3 param init
    self.params['W3'] = weight_scale * np.random.randn(num_filters, num_filters, filter_size, filter_size)
    self.params['b3'] = np.zeros(num_filters)

    #1st affine init
    self.params['W4'] = weight_scale * np.random.randn(num_filters*H_output_conv3*W_output_conv3, hidden_dim)
    self.params['b4'] = np.zeros(hidden_dim)
    self.params['gamma4'] = np.ones(hidden_dim)
    self.params['beta4'] = np.zeros(hidden_dim)

    #2nd affine init
    self.params['W5'] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
    self.params['b5'] = np.zeros(hidden_dim)
    self.params['gamma5'] = np.ones(hidden_dim)
    self.params['beta5'] = np.zeros(hidden_dim)

    #3nd affine init
    self.params['W6'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b6'] = np.zeros(num_classes)

    #init bn_params (bn_params[layer_number] = bn params related to BN stage in this layer)
    self.bn_params = {}
    self.bn_params[4] = {'mode': 'train'}
    self.bn_params[5] = {'mode': 'train'}
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)     

  def affine_bn_relu_forward(self, x, w, b, gamma, beta, bn_param):
    affine_out, affine_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_out)
    cache = (affine_cache, bn_cache, relu_cache)
    return out, cache

  def affine_bn_relu_backward(self, dout, cache):
    affine_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward_alt(dbn, bn_cache)
    dx, dw, db = affine_backward(da, affine_cache)
    return dx, dw, db, dgamma, dbeta

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the five-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    W5, b5 = self.params['W5'], self.params['b5']
    gamma5, beta5 = self.params['gamma5'], self.params['beta5']
    W6, b6 = self.params['W6'], self.params['b6']


    #determine mode, for bn_params
    mode = 'test' if y is None else 'train'
    for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    #conv1-relu1-pool1 layer output (layer 1)
    crp1_out, crp1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #conv2-relu2-pool2 layer output (layer 2)
    crp2_out, crp2_cache = conv_relu_pool_forward(crp1_out, W2, b2, conv_param, pool_param)
    #conv-relu layer output (layer 3)
    cr3_out, cr3_cache = conv_relu_forward(crp2_out, W3, b3, conv_param)

    #affine1-bn1-relu1 layer output (layer 4)
    abr1_out, abr1_cache = self.affine_bn_relu_forward(cr3_out, W4, b4, gamma4, beta4, self.bn_params[4])
    #affine1-bn1-relu1 layer output (layer 5)
    abr2_out, abr2_cache = self.affine_bn_relu_forward(abr1_out, W5, b5, gamma5, beta5, self.bn_params[5])

    #last affine output (layer 5)
    affine_out, affine_cache = affine_forward(abr2_out, W6, b6)
    scores = affine_out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, daffine_out = softmax_loss(scores, y)
    #add regularization loss
    loss += (0.5*self.reg*np.sum(self.params['W1']**2) + 0.5*self.reg*np.sum(self.params['W2']**2) + 0.5*self.reg*np.sum(self.params['W3']**2)
        + 0.5*self.reg*np.sum(self.params['W4']**2) + 0.5*self.reg*np.sum(self.params['W5']**2) + 0.5*self.reg*np.sum(self.params['W6']**2))

    #grads
    dloss = 1.0
    daffine_out = daffine_out*dloss
    #grad layer 6
    dabr2_out, dW6, db6 = affine_backward(daffine_out, affine_cache)
    #grad layer 5
    dabr1_out, dW5, db5, dgamma5, dbeta5 = self.affine_bn_relu_backward(dabr2_out, abr2_cache)
    #grad layer 4
    dcr3_out, dW4, db4, dgamma4, dbeta4 = self.affine_bn_relu_backward(dabr1_out, abr1_cache)
    #grad layer 3
    dcrp2_out, dW3, db3 = conv_relu_backward(dcr3_out, cr3_cache)
    #grad layer 2
    dcrp1_out, dW2, db2 = conv_relu_pool_backward(dcrp2_out, crp2_cache)
    #grad layer 1
    dX, dW1, db1 = conv_relu_pool_backward(dcrp1_out, crp1_cache)

    #add  regulalization to gradients
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    grads['b4'] = db4
    grads['b5'] = db5
    grads['b6'] = db6
    grads['W1'] = dW1 + self.reg*self.params['W1']
    grads['W2'] = dW2 + self.reg*self.params['W2']
    grads['W3'] = dW3 + self.reg*self.params['W3']
    grads['W4'] = dW4 + self.reg*self.params['W4']
    grads['W5'] = dW5 + self.reg*self.params['W5']
    grads['W6'] = dW6 + self.reg*self.params['W6']
    grads["gamma4"] = dgamma4
    grads["gamma5"] = dgamma5
    grads["beta4"] = dbeta4
    grads["beta5"] = dbeta5
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
