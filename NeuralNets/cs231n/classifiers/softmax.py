import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    #unnorm log prob
    s_i = X[i,:].dot(W)
    #normalization trick as mentioned in lecture notes
    s_i -= np.max(s_i)

    #loss: loss for current i = -s_y[i] + log( e^{s_0} + e^{s_1} ... )
    loss_sum_i = 0.0
    for s_ij in s_i:
      loss_sum_i += np.exp(s_ij)
    loss += (np.log(loss_sum_i) - s_i[y[i]])

    #gradient
    for j in range(num_classes):
      p_i = np.exp(s_i[j])/loss_sum_i
      dW[:, j] += p_i*X[i, :]
    dW[:, y[i]] -= X[i, :]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5*reg*np.sum(W**2)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  #scores = N*C dimensional matrix
  scores = X.dot(W)
  scores = scores.transpose()
  #normalization trick
  scores -= np.max(scores, axis=0)
  scores = scores.transpose()
  #loss
  correct_scores = scores[range(num_train), y]
  loss = np.mean(-correct_scores + np.log(np.sum(np.exp(scores), axis=1)))
  #gradient
  scores = scores.transpose()
  prob = np.exp(scores)/np.sum(np.exp(scores), axis=0)
  scores = scores.transpose()
  prob = prob.transpose()
  ind = np.zeros(prob.shape)
  ind[range(num_train), y] = 1
  dW = X.transpose().dot((prob-ind))
  dW /= num_train

  #regularize
  loss += 0.5 * reg * np.sum(W**2)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

