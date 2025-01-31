Introduction

The Automatic Identification System (AIS) transmits a ship’s position so that other ships are aware of its location. The International Maritime Organization and other regulatory bodies require large ships, including many commercial fishing vessels, to broadcast their position with AIS to avoid collisions.

By using deep learning to develop better prediction algorithms, a system with more precision and control can be built. This system would enable better monitoring and control of ships at sea.

For our purposes, we will use AIS data. The goal is to abstract the real-world dynamics of vessel movement into a time-series problem, utilizing past positional and kinematic data to predict future trajectories.


Theoretical Framework

Trajectory Representations

A vessel’s state at time t is represented as a feature vector:

x_t = (\phi_t, \lambda_t, SOG_t, COG_t, f_t)

where:

  = Latitude at time t,

  = Longitude at time t,

  = Speed Over Ground at time t (in knots),

  = Course Over Ground at time t (in degrees),

  = Additional features (e.g., heading, timestamp, vessel type).

The sequence of observations over a historical time window of size N forms the input:

X = \{ x_{t-N+1}, x_{t-N+2}, \dots, x_{t} \}

Output Sequence

The task is to predict a future sequence of vessel positions over a horizon of T time steps:

X = \{ (\hat{\phi}_{t+1}, \hat{\lambda}_{t+1}), (\hat{\phi}_{t+2}, \hat{\lambda}_{t+2}), \dots, (\hat{\phi}_{t+T}, \hat{\lambda}_{t+T}) \}

Mapping Function

The relationship between the input and output sequences is a function , parameterized by :

\hat{X} = f_\theta(X)

where  represents the predicted sequence.

Constraints

The predicted positions must satisfy the following constraints:

Spatial Bounds

\phi \in [-90, 90], \lambda \in [-180, 180].

Movement Consistency: The movement must respect the physical limits of speed and acceleration:

|\Delta \phi_{t+k}|, |\Delta \lambda_{t+k}| \leq v_{max} \cdot \Delta_t,

where  is the maximum realistic speed and  is the interval between observations.

Architecture Design

The model architecture is Long Short-Term Memory (LSTM). LSTM is a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data by addressing the vanishing gradient problem.

An LSTM unit consists of a cell state , an input gate , a forget gate , and an output gate . The computations for an LSTM cell at time step  are:

f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)

i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)

\tilde{C}_t = \tanh(W_C x_t + U_C h_{t-1} + b_C)

C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t

o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)

h_t = o_t \odot \tanh(C_t)

where  is the sigmoid activation function,  is the hyperbolic tangent function,  denotes element-wise multiplication, and  are learnable weight matrices and biases.

This LSTM model is designed to predict the next overall step/state (latitude, longitude, speed, course, and heading) based on historical time series data of the ship.

Model Architecture

Input Layer: Represents a window of past ship movements. The shape is (SEQ_LENGTH, num_features), where SEQ_LENGTH is the number of timesteps and num_features is the number of features.

First LSTM Layer: Contains 128 units, allowing the network to learn long-term dependencies. Uses tanh activation, which helps in handling negative values and prevents large values from exploding during training.

Second LSTM Layer: Has 64 units, reducing dimensionality. This layer uses ReLU activation, which allows for faster convergence and better performance compared to a second tanh.

Output Layer: Uses linear activation to output a 5-dimensional vector (latitude, longitude, speed, course, heading).

Optimizer
The model was built with the Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimizer, no other optimizers were tested mostly due to time constraints. The Nadam optimizer was selected for its adaptive learning rate adjustments and momentum-based updates to facilitate smoother optimization.

The Nadam learning rate was set to $0.001$ to have better performance while avoiding overshooting. In addition, a gradient clipping set to 1.0 was added to prevent exploding gradients, as previous models suffered from it.

Loss Function

The model must optimize a loss function that balances spatial and geospatial accuracy:

L = \alpha \cdot MSE + \beta \cdot d_{hav}

where:

Mean Squared Error (MSE):

MSE = \frac{1}{T} \sum_{k=1} ^{T} [ (\hat{\phi}_{t+k} - {\phi}_{t+k})^2 + (\hat{\lambda}_{t+k} - {\lambda}_{t+k})^2 ]

Haversine Distance:

d_{hav} = 2 \times R \times \arcsin \left( \sqrt{\sin^2(\frac{\hat{\phi}-\phi}{2})+\cos(\phi)\cos(\hat{\phi})\sin^2(\frac{\hat{\lambda}-\lambda}{2})} \right)

where  is the Earth’s radius. After experimentation, the parameters were set as  since they provided the best performance.

Evaluation Metrics

The model was evaluated using the following metrics:

Mean Haversine Distance

Cumulative Geospatial Error

Directional Consistency

Data Preprocessing

The data preprocessing consisted of the following steps:

The dataset was filtered to retain only the features of interest.

Instead of interpolation, data was subset to one sample per hour by selecting the closest observation to the hour, ensuring a consistent time frame.

Missing values were removed to prevent null evaluation metrics.

The dataset was split into training, validation, and testing subsets.

Normalizing the data resulted in worse performance, so it was skipped. Since the data values were of similar magnitudes, normalization was unnecessary. Further experiments with different normalization techniques could be explored.

Dataset Splitting

The dataset was divided as follows:

Training: 60% of the data

Validation: 20% of the data

Test: 20% of the data

Training Strategy

The training strategy incorporated batch processing, epochs, early stopping, and validation data:

Batches: A batch size of 16 was chosen for stable gradient updates. Given the training data size, a smaller batch size (16 or 32) was optimal, with 16 performing better.

Epochs & Early Stopping: A low number of epochs (20) was used to avoid overfitting. Early stopping (patience of 5) was implemented to reduce computational costs.

Validation Data: Validation data was used to monitor performance on unseen data, ensuring generalization and preventing overfitting.


Evaluation and Analysis
Evaluation
Evaluation on the test data:
• The combined loss for the test data is 1638.1918355494176
• The harvesine loss for the test data is 761.7693
• The geospatial loss for the test data is 6276979.0
• The Consistency loss for the test data is 105.77431
Since the test data consists of 8240 samples the performance is looking pretty good. Since for the averaged
metrics it keeps at least 1 magnitude from the data size. But further optimization could be done.
2.13 Evaluation visualized
Figures 2 and 3, show a visualization from the comparisons between true coordinates and predicted coordinates.
There is also a fourth figure only available in the notebook which fits the request for a coordinate level comparison.
The conclusion one can draw from this figures is that direction wise the model is accurate but further work is
to be made to increase the overall performance and minimize error in distance.
2.14 Analysis
The model is a first attempt at the problem. It lacks more development to achieve a physics and math based
rigor that would boost the confidence in the results. But overall the performance is decent and could be used as a
foundation for later work.
An approach I personally think could be investigated is a custom standardization to prevent the errors of standard
scaling in which the model performed worse and lacked a sense of ”direction” (it tended to be constant in either
latitude or longitude thus making it less physics based and just mathematically accurate).



