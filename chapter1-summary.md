# Summary: Chapter 1, "Using neural nets to recognize handwritten digits"

**Source:** Michael Nielsen, *Neural Networks and Deep Learning*, Chapter 1
(http://neuralnetworksanddeeplearning.com/chap1.html)

**Scope:** From the start of the chapter through the end of "Learning with gradient descent." The implementation section and "Toward deep learning" are not covered.

**Generated with:** Claude (Anthropic), from the full text of the assigned sections.

---

## Introduction

Humans read handwritten digits without effort, but only because a huge amount of visual processing happens unconsciously. The primary visual cortex alone has about 140 million neurons, and several more visual areas sit on top of it. The difficulty shows up as soon as you try to write a program for the task: rules like "a 9 is a loop on top of a vertical stroke" collapse under exceptions and special cases.

Neural networks take a different approach. Instead of hand-writing rules, you give the system many labeled examples (training examples) and let it infer the rules on its own. More examples generally means better accuracy.

The chapter's goal is a short program (74 lines, no neural network libraries) that learns to recognize digits with over 96% accuracy. Handwriting recognition is used as the running example because it is hard enough to be interesting but not so hard that it needs a huge solution. Along the way the chapter introduces two kinds of artificial neuron (the perceptron and the sigmoid neuron) and the standard learning algorithm, stochastic gradient descent, with an emphasis on why things work rather than only how.

## Perceptrons

A perceptron is an early artificial neuron, developed by Frank Rosenblatt in the 1950s and 60s, building on work by McCulloch and Pitts. It takes several binary inputs and produces one binary output.

- Each input $x_j$ has a **weight** $w_j$, a real number saying how much that input matters.
- The output is 1 if the weighted sum $\sum_j w_j x_j$ is above a **threshold**, and 0 otherwise.

**Intuition: weighing evidence.** The chapter's example is deciding whether to go to a cheese festival based on three yes/no factors: weather, whether your partner will come, and whether public transit is nearby. With weights 6, 2, 2 and a threshold of 5, only the weather matters. Drop the threshold to 3 and you also go when both of the other two conditions hold. Changing weights and thresholds gives different decision-making models.

**Layers.** Perceptrons in a first layer make simple decisions from the raw inputs. Perceptrons in later layers weigh the results of earlier layers, so they can make more complex and abstract decisions. A perceptron still has one output; drawing several output arrows only means that output feeds several other neurons.

**Bias notation.** Two simplifications are used for the rest of the book:

- Write the weighted sum as a dot product, $w \cdot x$.
- Move the threshold to the other side and call it the **bias**, $b = -\text{threshold}$.

The rule becomes: output 1 if $w \cdot x + b > 0$, otherwise 0. The bias measures how easy it is to get the neuron to fire. A large positive bias makes firing easy; a very negative one makes it hard.

**Perceptrons as logic gates.** A perceptron with two inputs, weights of $-2$ each, and bias 3 computes NAND. Since NAND is universal for computation, networks of perceptrons can compute any logical function. The chapter demonstrates this by building a circuit that adds two bits (a sum bit and a carry bit) out of NAND perceptrons. It also notes that the "input layer" neurons are not really perceptrons; they are just units defined to output the input values.

**Why this matters.** Universality is reassuring (perceptron networks are as powerful as any computer) but on its own a little disappointing, since it makes perceptrons look like just another kind of NAND gate. The real payoff is that we can design **learning algorithms** that tune the weights and biases automatically from data, with no programmer laying out the circuit by hand.

## Sigmoid neurons

**The problem with perceptrons.** Learning requires that a small change to a weight or bias produces a small change in the output. Then you can nudge the network step by step toward better behavior. Perceptrons break this: a tiny change can flip an output from 0 to 1, and that flip can change the rest of the network in complicated, hard-to-control ways.

**The fix.** A sigmoid neuron looks like a perceptron (inputs, weights, a bias) but:

- Inputs can be any value between 0 and 1, not only 0 or 1.
- The output is $\sigma(w \cdot x + b)$, where the **sigmoid function** (also called the logistic function) is

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Relationship to the perceptron.** Let $z = w \cdot x + b$.

- When $z$ is large and positive, $\sigma(z) \approx 1$.
- When $z$ is very negative, $\sigma(z) \approx 0$.
- Only for moderate $z$ does the behavior differ much from a perceptron.

So the sigmoid is a smoothed-out step function, and a sigmoid neuron is a smoothed-out perceptron. The smoothness is what matters, not the exact formula. Because $\sigma$ is smooth, the change in output is approximately a *linear* function of the changes in the weights and bias:

$$\Delta \text{output} \approx \sum_j \frac{\partial\, \text{output}}{\partial w_j} \Delta w_j + \frac{\partial\, \text{output}}{\partial b} \Delta b$$

That linearity makes it easy to pick small changes that move the output in a desired direction.

**Other points.**

- Other **activation functions** $f(w \cdot x + b)$ can be used. The sigmoid is chosen mainly because exponentials differentiate cleanly, which simplifies the algebra later.
- Outputs are real numbers between 0 and 1. When a yes/no answer is needed, a convention is used, such as treating an output of at least 0.5 as "yes."

**Exercises in this section** ask you to show that (1) scaling all weights and biases of a perceptron network by a positive constant $c$ does not change its behavior, and (2) a sigmoid network with weights and biases scaled by $c$ behaves exactly like the perceptron network as $c \to \infty$, as long as no neuron has $w \cdot x + b = 0$.

## The architecture of neural networks

**Terminology.**

- **Input layer:** the leftmost layer, made of input neurons.
- **Output layer:** the rightmost layer, made of output neurons.
- **Hidden layer:** any layer in between. "Hidden" only means "not an input or an output."
- Networks with multiple layers are sometimes called **multilayer perceptrons (MLPs)** for historical reasons, even though they are built from sigmoid neurons. The book avoids the term.

**Designing the layers.** Input and output layers are usually straightforward. For example, to decide whether a 64 by 64 greyscale image shows a 9, use 4,096 input neurons (one per pixel, intensities scaled to 0–1) and one output neuron (above 0.5 means "it's a 9"). Designing hidden layers is more of an art. There are no simple rules, only heuristics that researchers have built up, such as how to trade the number of hidden layers against training time.

**Feedforward vs. recurrent.**

- **Feedforward networks:** each layer's output feeds only the next layer. No loops, so information never flows backward. This is what the book focuses on.
- **Recurrent neural networks:** allow feedback loops. Neurons fire for a limited time and stimulate other neurons later, so a neuron's output affects its input only at a later time. They are closer to how brains work and may solve some problems feedforward networks find hard, but (as of the book's writing) their learning algorithms were less powerful.

## A simple network to classify handwritten digits

**Two sub-problems.** Recognizing a string of digits splits into:

1. **Segmentation:** breaking an image of several digits into separate single-digit images.
2. **Classification:** identifying each individual digit.

The book focuses on classification. Segmentation becomes manageable once you have a good classifier: try many candidate segmentations and score each by how confident the classifier is on every piece. Low confidence suggests a bad cut.

**The network.** Three layers:

| Layer | Size | Role |
|---|---|---|
| Input | 784 neurons | One per pixel of a 28 × 28 greyscale image (0.0 = white, 1.0 = black) |
| Hidden | $n$ neurons | Size is experimented with; the illustration uses $n = 15$ |
| Output | 10 neurons | Numbered 0–9; the neuron with the highest activation is the network's guess |

**Why 10 outputs instead of 4?** Four binary outputs could encode ten digits ($2^4 = 16$), so 10 looks wasteful. The main justification is empirical: the 10-output version learns better on this problem. The heuristic explanation is that hidden neurons may learn to detect component shapes of digits (for instance, four pieces that together form a 0). An output neuron for "0" can then simply weigh evidence from those shape detectors. With a 4-bit encoding, an output neuron would have to compute something like "the most significant bit of the digit," which has no natural connection to the shapes. Nielsen is clear that this is a heuristic and that nothing forces the network to work this way.

**Exercise in this section:** add a fourth layer that converts the 10-neuron output into a 4-bit binary representation, and find weights and biases for it.

## Learning with gradient descent

### The data: MNIST

- 60,000 training images and 10,000 test images, all 28 × 28 greyscale, with correct labels.
- Training images come from 250 people (half US Census Bureau employees, half high school students). Test images come from a *different* 250 people, so test accuracy reflects handwriting the network has never seen.
- Each input $x$ is treated as a 784-dimensional vector. The desired output $y(x)$ is a 10-dimensional vector with a 1 in the position of the correct digit and 0 elsewhere. For a 6: $y(x) = (0,0,0,0,0,0,1,0,0,0)^T$.

### The cost function

To measure how well the weights and biases are doing, define the **quadratic cost** (also called mean squared error, MSE):

$$C(w, b) = \frac{1}{2n} \sum_x \| y(x) - a \|^2$$

where $w$ is all the weights, $b$ all the biases, $n$ the number of training inputs, and $a$ the network's output vector for input $x$.

- $C$ is never negative.
- $C \approx 0$ exactly when the network's outputs are close to the desired outputs for all training inputs.
- Training therefore means finding weights and biases that make $C$ as small as possible.

**Why not just maximize the number of correct classifications?** That count is not a smooth function of the weights and biases. Most small changes do not change it at all, so it gives no signal about which way to adjust. A smooth cost does. The quadratic form is somewhat arbitrary and the book revisits the choice later, but it is fine for learning the basics.

### Gradient descent in general

Set aside neural networks and consider minimizing any function $C(v)$ of many variables. Solving for the minimum analytically with calculus is hopeless when there are many variables (the largest networks have billions).

**The analogy.** Picture $C$ as a valley and imagine a ball rolling downhill. We do not simulate real physics. We invent our own "law of motion" that guarantees the ball always moves downhill.

**The math.** For a small move $\Delta v$:

$$\Delta C \approx \nabla C \cdot \Delta v$$

where the **gradient** $\nabla C$ is the vector of partial derivatives of $C$. If we choose

$$\Delta v = -\eta \nabla C$$

with $\eta$ a small positive number called the **learning rate**, then $\Delta C \approx -\eta \|\nabla C\|^2 \le 0$. The cost always decreases (to within the approximation). The update rule is:

$$v \rightarrow v' = v - \eta \nabla C$$

Repeating this over and over is the gradient descent algorithm: compute the gradient, step in the opposite direction, repeat.

**Notes on the algorithm.**

- **Choosing $\eta$:** too large and the linear approximation breaks down, so the cost may go up. Too small and learning is very slow. In practice $\eta$ is often varied.
- Unlike a real ball, there is no momentum. The rule is simply "go down, right now."
- It works the same way for any number of variables.
- It does not always find the global minimum, but in practice it often works very well.
- It is optimal in a specific sense: among all steps of a fixed small size, the step along $-\nabla C$ reduces $C$ the most. (Proving this via the Cauchy–Schwarz inequality is an exercise.)
- Variants that mimic a physical ball more closely need second partial derivatives, which is expensive: a million variables means on the order of a trillion second derivatives.

### Applying it to neural networks

Replace the generic variables with the weights and biases:

$$w_k \rightarrow w_k' = w_k - \eta \frac{\partial C}{\partial w_k}, \qquad b_l \rightarrow b_l' = b_l - \eta \frac{\partial C}{\partial b_l}$$

**The practical problem.** The cost is an average of per-example costs, $C = \frac{1}{n}\sum_x C_x$. Computing the true gradient means computing $\nabla C_x$ for every training input and averaging, which is slow when $n$ is large.

### Stochastic gradient descent (SGD)

**Idea:** estimate the gradient from a small random sample instead of the whole training set.

- Pick a random **mini-batch** of $m$ training inputs, $X_1, \dots, X_m$.
- The average gradient over the mini-batch approximates the true gradient:

$$\nabla C \approx \frac{1}{m} \sum_{j=1}^{m} \nabla C_{X_j}$$

- Update the weights and biases using that estimate:

$$w_k \rightarrow w_k - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial w_k}, \qquad b_l \rightarrow b_l - \frac{\eta}{m} \sum_j \frac{\partial C_{X_j}}{\partial b_l}$$

- Pick another mini-batch and repeat. One full pass through the training inputs is called an **epoch**. Then a new epoch begins.

**Analogy:** SGD is like political polling. Sampling a small group is far cheaper than a full election. With $n = 60{,}000$ and $m = 10$, estimating the gradient is 6,000 times faster. The estimate is noisy, but that is fine, because all that is needed is a direction that generally reduces $C$.

**Convention note:** some authors drop the $\frac{1}{n}$ in the cost or the $\frac{1}{m}$ in the update. This is equivalent to rescaling $\eta$, but it matters when comparing results across different work.

**Online learning** (an exercise): the extreme case of a mini-batch size of 1, where the network updates after every single training input. The exercise asks for one advantage and one disadvantage compared with a mini-batch of about 20.

### Thinking in high dimensions

The cost is a surface in a space with a huge number of dimensions, which nobody can visualize. That is not a barrier. Even professional mathematicians mostly cannot picture four dimensions. They rely on other representations, like the algebraic expression for $\Delta C$ used above, to reason about what is going on.

---

## Key terms

| Term | Meaning |
|---|---|
| Perceptron | Neuron with binary inputs and output; fires if $w \cdot x + b > 0$ |
| Weight | Real number expressing how much an input matters |
| Bias | Negative of the threshold; how easily a neuron fires |
| Sigmoid neuron | Neuron whose output is $\sigma(w \cdot x + b)$, a smooth value between 0 and 1 |
| Activation function | The function applied to $w \cdot x + b$ (here, the sigmoid) |
| Hidden layer | Any layer that is neither input nor output |
| Feedforward network | Network with no loops; information flows one direction |
| Recurrent network | Network that allows feedback loops over time |
| MNIST | Dataset of 60,000 training and 10,000 test images of handwritten digits |
| Cost function | Measure of how far the network's outputs are from the desired outputs (also "loss" or "objective") |
| Gradient | Vector of partial derivatives of the cost |
| Learning rate ($\eta$) | Step size used in gradient descent |
| Gradient descent | Repeatedly stepping opposite the gradient to reduce the cost |
| Stochastic gradient descent | Gradient descent using a gradient estimated from a random mini-batch |
| Mini-batch | Small random sample of training inputs used for one update |
| Epoch | One full pass through the training data |
| Online learning | SGD with a mini-batch size of 1 |

## The chapter in five sentences

1. Hand-written rules fail at digit recognition, so we let a network learn the rules from labeled examples.
2. Perceptrons weigh evidence and can compute any logical function, but their all-or-nothing output makes gradual learning impractical.
3. Sigmoid neurons smooth out the perceptron so that small changes in weights and biases cause small changes in output, which is what makes learning possible.
4. A 784-$n$-10 feedforward network is trained by defining a smooth cost function that measures its error on MNIST and minimizing that cost.
5. Gradient descent minimizes the cost by repeatedly stepping against the gradient, and stochastic gradient descent makes this fast by estimating the gradient from small random mini-batches.
