Originally from https://github.com/SimBe195/monotonic-rnnt which is a fork of https://github.com/HawkAaron/warp-transducer

# Monotonic RNN-T Loss

## Theory

### Loss function

The monotonic RNN-T loss can be written as

$$L = -\log p(a_1^S \mid x_1^T) = -\log \sum_{y_1^T : a_1^S} p(y_1^T \mid x_1^T) = -\log \sum_{y_1^T : a_1^S} \prod_{t=1}^T p_t(y_t \mid x_1^T, a(y_1^{t-1}))$$

where $S$ is the number of labels, $T$ is the number of time-steps, $a_1^S$ is the ground-truth label sequence, $x_1^T$
are the acoustic features, $y_1^T$ is the set of alignments of $a_1^S$ as a result of inserting blank symbols and $a(
y_1^t)$ is a function that simplifies the history (usually removal of blanks and sometimes truncation to only the one or
two most recent symbols). For simplicity, $x_1^T$ will be omitted from the dependencies from now on.

### Forward-backward

The loss and gradients can be computed using the forward-backward-algorithm. For this, a forward variable

$$\alpha(t, s) = \sum_{y_1^t : a_1^s} \prod_{t'=1}^t p_{t'}(y_{t'} \mid a(y_1^{t'-1}))$$

and a backward variable

$$\beta(t, s) = \sum_{y_t^T : a_s^S} \prod_{t'=t}^T p_{t'}(y_{t'} \mid a(a_1^s || y_{t'}^T))$$

are introduced. These have the property

$$L = -\log \alpha(T, S) = -\log \beta(1, 0)$$

and adhere to the recursive equations

$$\alpha(t, s) = p_t(\epsilon \mid a(a_1^s)) \cdot \alpha(t-1, s) + p_t(a_s \mid a(a_1^{s-1})) \cdot \alpha(t-1, s-1)$$

and

$$\beta(t, s) = p_t(\epsilon \mid a(a_1^s)) \cdot \beta(t+1, s) + p_t(a_{s+1} \mid a(a_1^s)) \cdot \beta(t+1, s+1)$$

(excluding edge cases).

### Gradients

For the gradients it is straightforward to prove that for any $t$

$$p(a_1^S) = \sum_{s=0}^S \alpha(t, s) \cdot \beta(t + 1, s)$$

And thus

$\frac{\partial p(a_1^S)}{\partial p_t(y \mid a(a_1^s))}$

$=\frac{\partial}{\partial p_t(y \mid a_1^s)} \left( \sum_{s'} \alpha(t, s') \cdot \beta(t+1, s') \right)$

$=\frac{\partial}{\partial p_t(y \mid a_1^s)} \left( \sum_{s'} \left( p_t(\epsilon \mid a(a_1^{s'})) \cdot \alpha(t-1, s') +
p_t(a_{s'} \mid a(a_1^{s'-1})) \cdot \alpha(t-1, s'-1)\right) \cdot \beta(t+1, s') \right)$

$= \alpha(t-1, s) \cdot \beta(t+1, s)$ if $y = \epsilon$

$= \alpha(t-1, s) \cdot \beta(t+1, s+1)$ if $y = a_{s+1}$ and

$= 0$ otherwise.

which means for the overall gradient

$\frac{\partial L}{\partial p_t(y \mid a(a_1^s))}$

$= - \frac{\alpha(t-1, s) \cdot \beta(t+1, s)}{p(a_1^S)}$ if $y = \epsilon$

$= - \frac{\alpha(t-1, s) \cdot \beta(t+1, s+1)}{p(a_1^S)}$ if $y = a_{s+1}$

$= 0$ otherwise.

For expressing the derivative directly with respect to the logits $z_1^V$ where
$p_t(y \mid a(a_1^s)) = \frac{e^{z_y}}{\sum_v e^{z_v}}$
we can derive with some calculation that

$\frac{\partial L}{\partial z_y}$

$= - \frac{\alpha(t-1, s) \cdot p(\epsilon \mid a(a_1^s)) \left(\beta(t+1, s) - \beta(t, s) \right)}{p(a_1^S)}$ if $y =\epsilon$

$= - \frac{\alpha(t-1, s) \cdot p(\epsilon \mid a(a_1^s)) \left(\beta(t+1, s+1) - \beta(t, s) \right)}{p(a_1^S)}$ if $y = a_{s+1}$

$= - \frac{\alpha(t-1, s) \cdot p(\epsilon \mid a(a_1^s)) \left(-\beta(t, s)\right)}{p(a_1^S)}$ otherwise

## Example

Assume the following model posteriors $p_t(y \mid a(a_1^s))$ with $T = 4$, $S = 2$ and number of classes $V = 3$ with
blank-index $0$.

    // t = 1
    0.6, 0.3, 0.1,  // s = 0
    0.7, 0.1, 0.2,  // s = 1
    0.5, 0.1, 0.4,  // s = 2

    // t = 2
    0.5, 0.4, 0.1,  // s = 0
    0.5, 0.1, 0.4,  // s = 1
    0.8, 0.1, 0.1,  // s = 2

    // t = 3
    0.4, 0.3, 0.3,  // s = 0
    0.5, 0.1, 0.4,  // s = 1
    0.7, 0.2, 0.1,  // s = 2

    // t = 4
    0.8, 0.1, 0.1,  // s = 0
    0.3, 0.1, 0.6,  // s = 1
    0.8, 0.1, 0.1   // s = 2

For $a_1^S = [1, 2]$ the valid alignments $y_1^T$ are as follows (with "." denoting blank):

- . . 1 2
- . 1 . 2
- . 1 2 .
- 1 . . 2
- 1 . 2 .
- 1 2 . .

The 6 paths have probabilities of

- 0.6 _ 0.5 _ 0.3 \* 0.6 = 0.0540
- 0.6 _ 0.4 _ 0.5 \* 0.6 = 0.0720
- 0.6 _ 0.4 _ 0.4 \* 0.8 = 0.0768
- 0.3 _ 0.5 _ 0.5 \* 0.6 = 0.0450
- 0.3 _ 0.5 _ 0.4 \* 0.8 = 0.0480
- 0.3 _ 0.4 _ 0.7 \* 0.8 = 0.0672

wich sum to a total of 0.363, i.e. -1.0134 in log space

The alphas then are as follows in probability and log space:

- a(0, 0) = 1.0 -> 0.0
- a(1, 0) = 0.6 -> -0.51
- a(1, 1) = 0.3 -> -1.20
- a(2, 0) = 0.5 \* a(1, 0) = 0.3 -> -1.20
- a(2, 1) = 0.5 _ a(1, 1) + 0.4 _ a(1, 0) = 0.39 -> -0.94
- a(2, 2) = 0.4 \* a(1, 1) = 0.12 -> -2.12
- a(3, 1) = 0.5 _ a(2, 1) + 0.3 _ a(2, 0) = 0.285 -> -1.26
- a(3, 2) = 0.7 _ a(2, 2) + 0.4 _ a(2, 1) = 0.24 -> -1.43
- a(4, 2) = 0.8 _ a(3, 2) + 0.6 _ a(3, 1) = 0.363 -> -1.01

And the betas are as follows in probability and log space:

- b(5, 2) = 1.0 -> 0.0
- b(4, 2) = 0.8 -> -0.22
- b(4, 1) = 0.6 -> -0.51
- b(3, 2) = 0.7 \* b(4, 2) = 0.56 -> -0.58
- b(3, 1) = 0.5 _ b(4, 1) + 0.4 _ b(4, 2) = 0.62 -> -0.48
- b(3, 0) = 0.3 \* b(4, 1) = 0.18 -> -1.71
- b(2, 1) = 0.5 _ b(3, 1) + 0.4 _ b(3, 2) = 0.534 -> -0.63
- b(2, 0) = 0.5 _ b(3, 0) + 0.4 _ b(3, 1) = 0.338 -> -1.08
- b(1, 0) = 0.6 _ b(2, 0) + 0.3 _ b(2, 1) = 0.363 -> -1.01

As we can see $\alpha(4, 2) = -1.01 = \beta(0, 1)$ is the overall log-likelihood.

Now, the gradients with respect to all the logits can be computed as

    // t = 1
    0.04, -0.14, 0.1,  // s = 0
    0.0, 0.0, 0.0,  // s = 1
    0.0, 0.0, 0.0,  // s = 2

    // t = 2
    0.13, -0.19, 0.06,  // s = 0
    -0.04, 0.04, -0.01,  // s = 1
    0.0, 0.0, 0.0,  // s = 2

    // t = 3
    0.06, -0.1, 0.04,  // s = 0
    0.01, 0.07, -0.08,  // s = 1
    -0.06, 0.04, 0.02,  // s = 2

    // t = 4
    0.0, 0.0, 0.0,  // s = 0
    0.14, 0.05, -0.19,  // s = 1
    -0.11, 0.05, 0.05   // s = 2
