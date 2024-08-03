# Package overview: `stream_viz`

Welcome to our comprehensive data analysis and drift detection package! This library is designed to provide robust tools for encoding data, detecting various types of drifts, visualizing feature changes over time, and handling missing data with advanced visualization techniques. Whether you're dealing with normal data, data with missing values, or strategy data, our package has specialized encoders and detectors to meet your needs. Here’s a detailed look at the key components and features of our package:

### Key Features
- **Data Encoding**: Efficient and flexible data encoding mechanisms for various data types.
- **Drift Detection**: Robust algorithms for detecting real concept drift and feature drift in streaming data.
- **Visualization**: Advanced visualization techniques for analyzing feature
changes, data velocity, learning strategies and missingness patterns.
Here’s a detailed look at the key components and features of our package:

_Note_: For more detailed explanation or implementation details, please check our [User Guide](tutorial/UserGuide.ipynb)

---

## Data Encoder Implementations

Our library includes three distinct implementations for data encoders, each tailored for specific types of data:

1. **`NormalDataEncoder`**:
    - Designed for normal data, such as *cfpdss*, without any missing values.


2. **`MissingDataEncoder`**:
   - Suitable for normal data, like *cfpdss*, that may contain missing values.

3. **`KappaStrategyDataEncoder`**:
   - Tailored for strategy data, such as *experiments*, without any missing values.

#### Inheritance and Common Methods

All the above implementations inherit from the `DataEncoder` abstract class, ensuring they adhere to a common interface. The abstract class enforces the implementation of the following methods:

- **`read_csv_data(filepath_or_buffer, *args, **kwargs)`**:
  - Reads CSV data from the specified file path and stores it in the `original_data` class attribute.
  - Internally utilizes the pandas `read_csv` method.

- **`encode_data()`**:
  - Encodes the data according to the defined process and stores the result in the `encoded_data` class attribute.

#### Special Considerations for Cfpdss Data

For `cfpdss`-related data encoders, there is an additional contract to handle the separation of the target variable from the rest of the data. Thus, the encoded data is stored in separate class attributes:

- **`X_encoded_data`**: Stores the features (input data).
- **`y_encoded_data`**: Stores the target variable.

Therefore, when working with `cfpdss` data encoders, use `X_encoded_data` and `y_encoded_data` instead of the `encoded_data` attribute mentioned above.

---

## Plotter Contract

All implementations that involve plotting graphs or figures must adhere to the **`Plotter`** contract. This contract enforces the use of a **`plot`** method to generate the figure according to the class's specific logic.

#### Key Benefits

- **Consistency**: Standardizing the method name across all plotting classes ensures that users don't need to remember different method names for various implementations. Each subclass must implement the `plot` method, making the API predictable and straightforward.
- **Ease of Use**: Users can invoke the plot method with a consistent syntax: `'your_class_object'.plot(*args, **kwargs)`. This uniformity simplifies the process of generating plots and enhances the overall user experience.
- **Maintainability**: Adhering to a common interface makes it easier to maintain and extend the library. All plotting classes follow the same method signature, streamlining both development and debugging.

#### Example Usage

Here’s an example demonstrating how to use the `plot` method with an instance of a class implementing the Plotter```python:

```python
# Assuming 'YourClass' is a subclass implementing the Plotter contract
your_class_instance = YourClass()
your_class_instance.plot(kw_arg_1=1, kw_arg_2="test")

```

---

## Drift Detection

For effective drift detection, we utilize implementations of two abstract classes:

1. **`DriftDetector(Plotter)`**:
    - **`update`**: This method is called at each stream time point from a `Streamer` implementation. It must update the drift mechanism with a new data point at each call. This is essential for keeping the detector current with incoming data.
    - **`detect_drift`**: This abstract method should implement the core logic of the drift detection mechanism. Different detection techniques will have their unique implementations of this method.
    - **`plot`**: Plots the relevant figure for the drifts of the implementation. Visualization is key to understanding and analyzing detected drifts.

2. **`Streamer`**:
    - **Constructor**: Takes the drift detection mechanism (following the above contract) as input. This ensures that the streamer is linked with the appropriate drift detection logic.
    - **`stream_data(X_dataframe, y_df)`**: This method takes the whole dataset as input and starts streaming the data. During streaming, it sequentially processes the data points.
    - **Streaming Logic**: Within this logic, the streamer calls the `update` method of the `DriftDetector` contract, passing the relevant data at each stream time point. This interaction is crucial for real-time drift detection.

### Key Classes

- **RealConceptDriftDetector**: This class implements the `DriftDetector` for detecting real concept drifts. It leverages specific algorithms tailored to identifying changes in the conditional probability distribution \(p(y|X)\).
- **DataStreamer**: This class implements the `Streamer` and handles the streaming of data. It ensures that data is fed into the drift detector in a sequential and timely manner.

### Note

- For real concept drift detection, we currently have the `RealConceptDriftDetector` class.
- For streaming the data, we use the `DataStreamer` class.

By following these contracts and using the provided classes, we can systematically detect and visualize drifts in streaming data, allowing for timely interventions and adjustments in our models and systems.

---

## 1.  Real Concept Drift

Real concept drift refers to changes in the conditional probability distribution $p(y|X)$, which affects the decision boundaries or the target concept. For example, a user's interest in news articles may shift from dwelling houses to holiday homes over time.

#### McDiarmid Drift Detection Method (MDDM)

The **McDiarmid Drift Detection Method (MDDM)** is a technique used to detect real concept drifts. It applies McDiarmid’s inequality and utilizes a sliding window approach to monitor changes in data streams. Key features of MDDM include:

- **Sliding Window Approach**: MDDM processes data in a sliding window to continuously evaluate changes and detect drifts.

- **Weighting Scheme for Elements in the Window**:
  - **Arithmetic Weighting**:
    - Weights are defined as $w_{i} = 1 + (i + d)$, where $d \geq 0$ is the difference between two consecutive weights. This scheme increases weights linearly with the index of the element.
  - **Geometric Weighting**:
    - Weights are defined as $w_{i} = r^{(i−1)}$, where $r \geq 1$ is the ratio between two consecutive weights. This scheme increases weights exponentially.
  - **Euler Weighting**:
    - Weights are defined as $w_{i} = r^{(i−1)}$ with $r = e^{λ}$, where $λ \geq 0$. This scheme also increases weights exponentially but uses the base of the natural logarithm.

These weighting schemes help assign different levels of importance to the elements within the sliding window for sensitive drift detection.

For further details, please refer
<a href="https://arxiv.org/pdf/1710.02030">Pesaranghader, A., Viktor, H.L., & Paquet, E. (2017). McDiarmid Drift Detection Methods for Evolving Data Streams. 2018 International Joint Conference on Neural Networks (IJCNN), 1-9.</a>

---

## 2. Feature Drift Detection

### FeatureDriftDetector Class

The `FeatureDriftDetector` class is designed to monitor and detect feature drift in streaming data using the __Kolmogorov-Smirnov (KS)__ test and __Population Stability Index__ Test (PSI test). This class can handle numerical features and categorical features, and provides functionality to identify sudden, linear, and gradual drifts in the data. Drift detection is crucial for maintaining the reliability and accuracy of machine learning models, especially in dynamic environments where data distributions can change over time.

#### Parameters

- **features_list** (`List[str]`):
  A list of feature names to monitor for drift. It is essential to specify the features of interest to ensure targeted monitoring and efficient detection.

- **window_size** (`int`, optional):
  The size of the window to use for drift detection. The default value is 300. The window size determines the number of recent data points considered for drift analysis at any given time.

- **ks_test_pval** (`float`, optional):
  The p-value threshold for the Kolmogorov-Smirnov test. The default value is 0.001. This threshold helps in determining the statistical significance of the detected drift, ensuring that only meaningful changes are flagged.

- **gap_size** (`int`, optional):
  The size of the gap between segments when computing gradual drift. The default value is 50. This parameter helps in distinguishing gradual changes from sudden shifts by introducing a gap between the two halves of the data used in the KS test.

- **p_val_threshold** (`float`, optional):
  The p-value threshold for gradual drift detection. The default value is 0.0001. This threshold is used to determine the significance of gradual changes in the data, providing an additional layer of drift detection sensitivity.

- **psi_threshold** (`float`, optional):
  The psi threshold for the Population Stability Index. The default value is 0.12. This threshold helps in determining the significance of the detected drift, ensuring that meaningfu changes are detected or flagged.

#### Drift Detection Methodology

Drift detection in the `FeatureDriftDetector` is performed using the Kolmogorov-Smirnov (KS) test for numerical features and Population Stability Index Test (PSI test) for categorical features, which compares the distributions of feature values within a sliding window of data. The following steps outline the drift detection process:

1. **Window Data Segmentation**:
    - The feature values within the current window are split into two halves: the first half and the second half.
    - Additionally, for detecting gradual drift, the data is further segmented into overlapping parts by introducing a gap between segments. Specifically, the first half of the data ends before the gap, and the second half of the data starts after the gap. This method ensures that gradual changes are not masked by immediate changes around the midpoint of the window.

2. **Kolmogorov-Smirnov Test**:
    - The KS test is applied for numerical features to compare the distributions of the first and second halves.
    - If the p-value from the KS test is below the specified threshold (`ks_test_pval`), it indicates a significant difference between the two halves, suggesting potential drift.

3. **Population Stability Index Test**:
    - The PSI test is applied for categorical features to compare the distributions of the first and second halves.
    - If the psi value from the PSI test is above the specified threshold (`psi_threshold`), it indicates a significant difference between two halves, indicating potential drift.

3. **Gradual Drift Detection**:
    - Another KS test and PSI test are conducted on the overlapping parts of the data to detect gradual drift. The overlapping parts are defined by excluding the gap in the middle of the window.
    - If the p-value from this test is below the gradual drift threshold (`p_val_threshold`), it suggests gradual changes in the data distribution.
    - Similarly, if the psi value from this test is above the psi threshold, it indicates gradual drift in data distribution.

4. **Drift Type Identification**:
    - If drift is detected, the type of drift is determined based on the mean difference between the halves:
        - **Sudden Drift**: Identified if the mean difference is significant and exceeds the standard deviation of the window data.
        - **Linear Drift**: Identified if the mean difference is positive but does not exceed the standard deviation.
        - **Gradual Drift**: Identified if the gradual drift test detects a significant difference.

---

## 3. Velocity Plots

To analyze velocity, we provide two implementations under the `Velocity(Plotter)` contract:

1. **`StackedBarChart`**:
    - **Purpose**: Plots a stacked bar chart for a given single __categorical__ feature, for each time period.
    - **Details**: Each time period consists of several time points, allowing you to observe changes in the distribution of categorical values over time.

2. **`RollingMeanStds`**:
    - **Purpose**: Plots rolling/moving means and standard deviations given a window size for __numerical__ features.
    - **Details**: Multiple numerical features can be given as input. This plot generates sub-plots for each feature with a common x-axis, facilitating better comparative visual analysis across features.

3. **`StreamGraph`**:
    - **Purpose**: Plots a stream graph for a given single __categorical__ feature, for each time period/ window size.
    - **Details**: Each time period consists of 50 time points, allowing you to observe changes in the distribution of categorical values over time.

### Combined Implementation

The above classes are combined into a single class, **`FeatureVelocity`**, for ease of use, ensuring a unified approach to plotting velocity.

- **`FeatureVelocity`**:
    - **Purpose**: Simplifies the process by allowing you to call a single class for velocity plots.
    - **Functionality**: This implementation takes the feature name(s) and differentiates between numerical and categorical features based on the metadata of encoders. It then internally calls the respective class object (`StackedBarChart` for categorical features and `RollingMeanStds` for numerical features).

### Usage

- **Standalone Approach**: Users have the option to use `StackedBarChart` or `RollingMeanStds` individually if they wish to focus on a specific type of feature.
- **Combined Approach**: Alternatively, users can employ the `FeatureVelocity` class to handle both categorical and numerical features in a streamlined manner.
- **Standalone Approach**: Users can the `StreamGraph` class to plot stream graph for categorical features.

### Key Features

- **Categorical Feature Analysis**: Visualize the distribution changes of categorical features over time with stacked bar charts.
- **Numerical Feature Analysis**: Monitor trends and variations in numerical features using rolling means and standard deviations.
- **Unified Interface**: The `FeatureVelocity` class provides a convenient way to generate both types of plots, enhancing flexibility and ease of use.

By leveraging these implementations, users can perform a comprehensive analysis of feature velocities, gaining valuable insights into how features change over time.


### Stream Graphs
- A stream graph shows how categories in a feature change over time, with the vertical axis representing the magnitude of each category at each time point.
- The baseline is not fixed, allowing for negative values on the Y-axis.
- The stream graph is used typically used for categorical data, it can be used for numerical data provided there are bins for the numerical data (This is not implemented in the current version).
- `StreamGraph` class implements this.

---

## 4. Missingness Plots

In addition to the `Plotter` contract, we have another abstract class named **`InteractivePlot`**. By inheriting this contract, you can create user-friendly interactive plots, allowing you to tune the parameters using buttons, sliders, or other UI elements. This enhances the capability for rigorous visual analysis.

### Implementation Requirements

- **`_add_interactive_plot(*args, **kwargs)`** : Any class implementing this contract must define the `_add_interactive_plot` abstract method to add custom inputs to the interactive plot.
- **`display(*args, **kwargs)`** : You can further customize the interactive plot by overriding the `display` method for specific logic.
- **Feature Display**: By default, the plot displays only the first three features of the dataset, but this can be adjusted through the UI elements.

Additionally, any class implementing the `InteractivePlot` contract must also implement the `Plotter` class or at least the `plot` method inherited from `Plotter`. This ensures consistency and usability across different types of plots.

### Usage Instructions

- **Normal Plot**: Call the `plot` method on your object for a standard plot.
- **Interactive Plot**: Call the `display` method for an interactive plot. This dual functionality lets users choose their preferred visualization method.

### Key Features

- **User-Friendly Interactivity**: Enhance plots with buttons, sliders, and other UI elements for detailed visual analysis.
- **Customizable Inputs**: Implement the `_add_interactive_plot` method to tailor inputs to your plot’s needs.
- **Default Feature Display**: Initially displays the first three features of the dataset, with UI options to adjust.
- **Dual Plotting Functionality**: Easily switch between the standard `plot` method and the interactive `display` method based on user preference.

This framework provides a comprehensive solution for creating both static and interactive plots, enhancing flexibility and ease of use for various data visualization needs.

**_Note_**: As of now, only `HeatMapPlotter` implements this interface.

### Scatter Plot
**_Note_**: The user should upload both the complete and incomplete datasets to generate this visualization.
- The ScatterPlotter class has two methods: plot_numerical and plot_categorical.
- The plot_numerical method handles numerical features, displaying all time points on the X-axis.
- The plot_categorical method handles categorical features, adjusting missing values to 0.5 after one-hot encoding and limiting the X-axis to the first 100 time points.
- Implementation Class : `ScatterPlotter`

### Data Missingness HeatMap Plot (Additional/Optional)

**MAR (Missing at Random):**
- The missing data are related to some of the observed data but not to the missing data itself.
- For example, consider two variables: age and income. Income might be missing more frequently for very young people, but the missingness is not due to income itself.

**Achieved by:**
1. **Binned Numerical Features**: Numerical features are binned with the help of decision trees to categorize them into discrete intervals.
2. **Missing Value Indicators**: Extra columns are added to the dataset to indicate the presence of missing values (is_na_col).
3. **Chi-Square Test**: A Chi-Square test is used to check for dependency between features from non-missing data and the missingness of corresponding features. This test compares the distribution of categorical variables.
    - **Null Hypothesis**: There is no relationship between the given two variables.
    - **Significance Level**: Assumed to be 0.05. If the p-value is greater than the significance level, we fail to reject the null hypothesis, indicating that the data is not MAR.

**Implemented by:**
- **`MarHeatMap`**: This class computes and visualizes the association between categorical columns and missing indicator columns using the chi-square test. You need to provide the missing and non-missing data and call the `plot` method with relevant parameters.

---

## 5. Learning Strategies Plot

### Strategy Plot
When evaluating various models or strategies on each batch of your dataset, it is common to visualize their performance in a plot. However, when the number of strategies increases, the plot can become cluttered, making it challenging to identify the best-performing strategy at a glance. Below is an example of such a cluttered plot, where the lines representing different strategies overlap, creating visual chaos.

The cluttered visualization requires manual effort to discern which strategy is performing best in each batch, leading to potential oversight of key insights


### Proposed Strategy Plot
The proposed strategy plot addresses this issue by clearly indicating the winning strategy at each batch. Additionally, it shows the margin or difference by which the winning strategy outperforms the second-best strategy. This enhancement not only simplifies the comparison but also highlights the most effective strategy over time, making it easier to identify the overall winner.


- **Winner Indication**: The plot clearly marks the winning strategy at each batch.
- **Margin Highlight**: It also shows the margin or difference between the winner and the second-best strategy, providing a clear quantitative measure of performance
>
This refined visualization aids in quickly spotting trends and making informed decisions without the need for manual inspection
