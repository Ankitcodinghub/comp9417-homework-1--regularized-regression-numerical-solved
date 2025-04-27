# comp9417-homework-1--regularized-regression-numerical-solved
**TO GET THIS SOLUTION VISIT:** [COMP9417 Homework 1- Regularized Regression & Numerical Solved](https://www.ankitcodinghub.com/product/comp9417-machine-learning-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;121019&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;2&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (2 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP9417 Homework 1- Regularized Regression \u0026amp; Numerical Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (2 votes)    </div>
    </div>
Homework 1: Regularized Regression &amp; Numerical

Optimization

Introduction In this homework we will explore some algorithms for gradient based optimization. These algorithms have been crucial to the development of machine learning in the last few decades. The most famous example is the backpropagation algorithm used in deep learning, which is in fact just an application of a simple algorithm known as (stochastic) gradient descent. We will first implement gradient descent from scratch on a deterministic problem (no data), and then extend our implementation to solve a real world regression problem.

• Question 1 b): 1 mark

• Question 1 c): 1 mark

• Question 1 e): 1 mark

• Question 1 f): 1 mark

• Question 1 g): 1 mark

What to Submit

• A single PDF file which contains solutions to each question. For each question, provide your solution in the form of text and requested plots. For some questions you will be requested to provide screen shots of code used to generate your answer — only include these when they are explicitly asked for.

• .py file(s) containing all code you used for the project, which should be provided in a separate .zip file. This code must match the code provided in the report.

1

• You cannot submit a Jupyter notebook; this will receive a mark of zero. This does not stop you from developing your code in a notebook and then copying it into a .py file though, or using a tool such as nbconvert or similar.

• We will set up a Moodle forum for questions about this homework. Please read the existing questions before posting new questions. Please do some basic research online before posting questions. Please only post clarification questions. Any questions deemed to be fishing for answers will be ignored and/or deleted.

• Please check Moodle announcements for updates to this spec. It is your responsibility to check for announcements about the spec.

• Please complete your homework on your own, do not discuss your solution with other people in the course. General discussion of the problems is fine, but you must write out your own solution and acknowledge if you discussed any of the problems in your submission (including their name(s) and zID).

When and Where to Submit

• Submission must be done through Moodle, no exceptions.

Question 1. Gradient Based Optimization

The general framework for a gradient method for finding a minimizer of a function f : Rn → R is defined by

x(k+1) = x(k) − αk∇f(xk), k = 0,1,2,…, (1)

where αk &gt; 0 is known as the step size, or learning rate. Consider the following simple example of√

minimizing the function g(x) = 2 x3 + 1. We first note that g0(x) = 3×2(x3 + 1)−1/2. We then need to choose a starting value of x, say x(0) = 1. Let’s also take the step size to be constant, αk = α = 0.1. Then we have the following iterations:

x(1) = x(0) − 0.1 × 3(x(0))2((x(0))3 + 1)−1/2 = 0.7878679656440357 x(2) = x(1) − 0.1 × 3(x(1))2((x(1))3 + 1)−1/2 = 0.6352617090300827

x(3) = 0.5272505146487477

…

and this continues until we terminate the algorithm (as a quick exercise for your own benefit, code this up and compare it to the true minimum of the function which is x∗ = −1). This idea works for functions that have vector valued inputs, which is often the case in machine learning. For example, when we minimize a loss function we do so with respect to a weight vector, β. When we take the stepsize to be constant at each iteration, this algorithm is known as gradient descent. For the entirety of this question, do not use any existing implementations of gradient methods, doing so will result in an automatic mark of zero for the entire question. (a) Consider the following optimisation problem:

min f(x), x∈Rn

where

,

and where A ∈ Rm×n, b ∈ Rm are defined as

,

and γ is a positive constant. Run gradient descent on f using a step size of α = 0.1 and γ = 0.2 and starting point of x(0) = (1,1,1,1). You will need to terminate the algorithm when the following condition is met: k∇f(x(k))k2 &lt; 0.001. In your answer, clearly write down the version of the gradient steps (1) for this problem. Also, print out the first 5 and last 5 values of x(k), clearly indicating the value of k, in the form:

k = 0, x(k) = [1,1,1,1]

k = 1, x(k) = ···

k = 2, x(k) = ···

…

What to submit: an equation outlining the explicit gradient update, a print out of the first 5 (k = 5 inclusive) and last 5 rows of your iterations. Use the round function to round your numbers to 4 decimal places. Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

(b) In the previous part, we used the termination condition k∇f(x(k))k2 &lt; 0.001. What do you think this condition means in terms of convergence of the algorithm to a minimizer of f ? How would making the right hand side smaller (say 0.0001) instead, change the output of the algorithm? Explain.

What to submit: some commentary.

(c) Although we used gradient descent to find the minimizer of f in part (a), we can also use calculus to solve the problem directly. Show that the minimizer of f is

xˆ = (ATA + γI)−1ATb,

where I is the 4 × 4 identity matrix. What is the exact numerical value of xˆ for the problem in (a) and how does it compare to your result from gradient descent? What to submit: your working out.

(d) In this question we’ll investigate the step-size parameter in more depth. Run the gradient descent algorithm 9 times, each time with a different step-size, ranging over:

α ∈ {0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.02,0.1,0.15},

and with γ = 0.2. For each choice of α, plot the difference: kx(k) − xˆk2 over all steps k = 1,2,…, and where xˆ is the true solution from part (c). Use the same termination condition as before (k∇f(x(k))k2 &lt; 0.001) with an additional termination constraint of k &lt; 10000 (the algorithm terminates after 10,000 steps at most). Present your results as a 3 × 3 grid plot (one subplot for each α), and on each subplot also plot the line y = 0.001 in red. Comment on your results. What effect does changing the step-size have? What would you expect as you take the step-size to be increasingly large (α = 10 for example). What to submit: a single plot, some commentary. Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

In the next few parts, we will use gradient methods explored above to solve a real machine learning problem. Consider the CarSeats data provided in CarSeats.csv. It contains 400 observations with each observation describing child car seats for sale at one of 400 stores. The features in the data set are outlined below:

• Sales: Unit sales (in thousands) at each location

• CompPrice: Price charged by competitor at each location

• Income: Local income level (in thousands of dollars)

• Advertising: advertising budget (in thousands of dollars)

• Population: local population size (in thousands)

• Price: price charged by store at each site

• ShelveLoc: A categorical variable with Bad, Good and Medium describing the quality of the shelf location of the car seat

• Age: Average age of the local population

• Education: Education level at each location

• Urban A categorical variable with levels No and Yes to describe whether the store is in an urban location or in a rural one

• US: A categorical variable with levels No and Yes to describe whether the store is in the US or not.

The target variable is Sales. The goal is to learn to predict the amount of Sales as a function of a subset of the above features. We will do so by running Ridge Regression (Ridge) which is defined as follows

βˆRidge = argmin ,

where β ∈ Rp,X ∈ Rn×p,y ∈ Rn and φ &gt; 0.

(e) We first need to preprocess the data. Remove all categorical features. Then use sklearn.preprocessing.StandardScaler to standardize the remaining features. Print out the mean and variance of each of the standardized features. Next, center the target variable (subtract its mean). Finally, create a training set from the first half of the resulting dataset, and a test set from the remaining half and call these objects X train, X test, Y train and Y test. Print out the first and last rows of each of these.

What to submit: a print out of the means and variances of features, a print out of the first and last rows of the 4 requested objects, and some commentary. Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

(f) Explain why standardization of the features is necessary for ridge regression. What issues might you run into if you used Ridge without first standardizing? What to submit: some commentary.

(g) It should be obvious that a closed form expression for βˆRidge exists. Write down the closed form expression, and compute the exact numerical value on the training dataset with φ = 0.5. What to submit: Your working, and a print out of the value of the ridge solution based on (X train, Y train). Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

We will now solve the ridge problem but using numerical techniques. As noted in the lectures, there are a few variants of gradient descent that we will briefly outline here. Recall that in gradient descent our update rule is

β(k+1) = β(k) − αk∇L(β(k)), k = 0,1,2,…,

where L(β) is the loss function that we are trying to minimize. In machine learning, it is often the case that the loss function takes the form

,

i.e. the loss is an average of n functions that we have lablled Li. It then follows that the gradient is also an average of the form

.

We can now define some popular variants of gradient descent .

(i) Gradient Descent (GD) (also referred to as batch gradient descent): here we use the full gradient, as in we take the average over all n terms, so our update rule is:

(ii) Stochastic Gradient Descent (SGD): instead of considering all n terms, at the k-th step we choose an index ik randomly from {1,…,n}, and update

β(k+1) = β(k) − αk∇Lik(β(k)), k = 0,1,2,….

Here, we are approximating the full gradient ∇L(β) using ∇Lik(β).

(iii) Mini-Batch Gradient Descent: GD (using all terms) and SGD (using a single term) represents the two possible extremes. In mini-batch GD we choose batches of size 1 &lt; B &lt; n randomly at each step, call their indices {ik1,ik2,…,ikB}, and then we update

so we are still approximating the full gradient but using more than a single element as is done in SGD.

(h) The ridge regression loss is

.

Show that we can write

,

and identify the functions L1(β),…,Ln(β). Further, show that

∇Li(β) = −2xi(yi − xTi β) + 2φβ, i = 1,…,n.

What to submit: your working.

(i) In this question, you will implement (batch) GD from scratch to solve the ridge regression problem. Use an initial estimate β(0) = 1p (the vector of ones), and φ = 0.5 and run the algorithm for 1000 epochs (an epoch is one pass over the entire data, so a single GD step). Repeat this for the following step sizes:

α ∈ {0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01}

To monitor the performance of the algorithm, we will plot the value

∆(k) = L(β(k)) − L(βˆ),

where βˆ is the true (closed form) ridge solution derived earlier. Present your results in a 3 × 3 grid plot, with each subplot showing the progression of ∆(k) when running GD with a specific step-size. State which step-size you think is best and let β(K) denote the estimator achieved when running GD with that choice of step size. Report the following:

(i) The train MSE:

(ii) The test MSE:

What to submit: a single plot, the train and test MSE requested. Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

(j) We will now implement SGD from scratch to solve the ridge regression problem. Use an initial estimate β(0) = 1p (the vector of ones) and φ = 0.5 and run the algorithm for 5 epochs (this means a total of 5n updates of β, where n is the size of the training set). Repeat this for the following step sizes:

α ∈ {0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.006,0.02}

Present an analogous 3 × 3 grid plot as in the previous question. Instead of choosing an index randomly at each step of SGD, we will cycle through the observations in the order they are stored in X train to ensure consistent results. Report the best step-size choice and the corresponding train and test MSEs. In some cases you might observe that the value of ∆(k) jumps up and down, and this is not something you would have seen using batch GD. Why do you think this might be happening?

What to submit: a single plot, the train and test MSE requested and some commentary. Include a screen shot of any code used for this section and a copy of your python code in solutions.py.

(k) Based on your GD and SGD results, which algorithm do you prefer? When is it a better idea to use GD? When is it a better idea to use SGD?
