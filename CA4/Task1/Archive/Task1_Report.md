# Task 1 - MLP on California Housing

## 1. Dataset Loading

The dataset was loaded from `../california_housing.csv` without modifying the original file. It contains 20640 samples, 8 input features, and the target `MedHouseVal`.

All columns are numeric and the total number of missing values is 0. The target is measured in hundreds of thousands of dollars. A notable dataset property is top-coding: 992 rows (4.81%) have `MedHouseVal` greater than or equal to 5.0, so the model cannot observe the real values above that cap.

## 2. Exploratory Data Analysis

The EDA artifacts are saved in `figures/` and the tabular summaries are saved as CSV files in this folder.

Main observations:

- `MedInc` has the strongest positive linear relationship with median house value, so income should be one of the most useful predictors.
- `Latitude` and `Longitude` matter because house prices have clear geographic structure, especially around coastal/high-demand regions.
- Several count and ratio variables are highly skewed, especially occupancy, population, and room-related variables. This supports scaling and log/rate feature engineering.
- Outliers are present in occupancy, population, and room variables. MLPs trained with MSE can be sensitive to these points, so validation monitoring and error analysis are important.
- The target cap near 5.0 can make expensive districts hard to fit: multiple true high-end values are collapsed to the same observed target.

Top absolute correlations with `MedHouseVal`:

| Feature | Correlation with target |
| --- | --- |
| MedInc | 0.6881 |
| AveRooms | 0.1519 |
| Latitude | -0.1442 |
| HouseAge | 0.1056 |
| AveBedrms | -0.0467 |
| Longitude | -0.0460 |
| Population | -0.0246 |
| AveOccup | -0.0237 |

Most skewed variables:

| Feature | Skewness |
| --- | --- |
| AveOccup | 97.6396 |
| AveBedrms | 31.3170 |
| AveRooms | 20.6979 |
| Population | 4.9359 |
| MedInc | 1.6467 |

Largest IQR outlier counts:

| Feature | IQR outlier count |
| --- | --- |
| AveBedrms | 1424 |
| Population | 1196 |
| MedHouseVal | 1071 |
| AveOccup | 711 |
| MedInc | 681 |

Strongly correlated original feature pairs:

| Feature 1 | Feature 2 | Correlation |
| --- | --- | --- |
| AveRooms | AveBedrms | 0.8476 |
| Latitude | Longitude | -0.9247 |

Key plots:

- `figures/eda_histograms.png`
- `figures/eda_boxplots.png`
- `figures/eda_correlation_heatmap.png`
- `figures/eda_feature_target_relationships.png`

## 3. Data Preparation

The data was split once at the row-index level and reused for both models:

| Split | Rows | Percent |
| --- | --- | --- |
| train | 14448 | 70.00 |
| validation | 3096 | 15.00 |
| test | 3096 | 15.00 |

For each model, `StandardScaler` was fitted only on the training features. The validation and test sets were transformed using the training scaler to avoid data leakage. The scaled arrays were converted to PyTorch tensors through `TensorDataset` and `DataLoader`.

## 4. Baseline MLP Model

The baseline model uses only the original 8 input features. It inherits from `torch.nn.Module` and uses fully connected layers with ReLU activations and dropout:

- Architecture: input -> 128 -> 64 -> 32 -> output
- Loss: Mean Squared Error
- Optimizer: Adam
- Batch size: 1024
- Weight decay: 0.0001
- Early stopping patience: 25
- Best checkpoint: `models\baseline_best.pt`

Training and validation losses are plotted in `figures/training_curve_baseline.png`.

## 5. Feature Engineering

The enhanced model adds domain-inspired features:

| Engineered feature |
| --- |
| RoomsPerPerson |
| BedroomsPerRoom |
| RoomsPerBedroom |
| IncomePerOccupant |
| PopulationLog |
| AveOccupLog |
| AveRoomsLog |
| LatitudeLongitude |
| LatitudeSquared |
| LongitudeSquared |
| DistanceToLosAngeles |
| DistanceToSanFrancisco |
| ClosestMajorCityDistance |

These features expose ratios, log-compressed skewed variables, nonlinear geographic structure, and approximate distances to Los Angeles and San Francisco. They are intended to help the MLP learn useful tabular relationships without needing to infer every interaction from raw columns alone.

## 6. Enhanced MLP Model

The enhanced model uses the 8 original features plus the engineered features. It also inherits from `torch.nn.Module`:

- Architecture: input -> 160 -> 96 -> 48 -> output
- Loss: Mean Squared Error
- Optimizer: Adam
- Batch size: 1024
- Weight decay: 0.0001
- Early stopping patience: 25
- Best checkpoint: `models\enhanced_best.pt`

Training and validation losses are plotted in `figures/training_curve_enhanced.png`.

## 7. Model Evaluation and Comparison

Test-set results:

| Model | MAE | RMSE | R2 | Best epoch | Epochs run |
| --- | --- | --- | --- | --- | --- |
| Baseline | 0.3445 | 0.4970 | 0.8132 | 260 | 260 |
| Enhanced | 0.3304 | 0.4866 | 0.8209 | 197 | 222 |

The best-performing model by RMSE is **Enhanced**. The enhanced model changed RMSE by 0.0104 and R2 by 0.0077 relative to the baseline.

Interpretation:

- Lower MAE/RMSE means fewer average and large prediction errors.
- R2 measures the fraction of test-set target variance explained by the model.
- The validation-loss plots in `figures/model_comparison_validation_loss.png` show convergence behavior and whether one model becomes unstable or overfits faster.
- In this run, feature engineering improved test RMSE. Even when the gain is modest, engineered features make relevant ratios and nonlinear geography easier for the network to use.

## 8. Error Analysis

The error analysis uses the **Enhanced** model. Plots are saved as:

- `figures/error_actual_vs_predicted.png`
- `figures/error_residual_distribution.png`
- `figures/error_residuals_by_actual.png`

Summary:

- Mean residual: 0.0050
- Median absolute error: 0.2191
- 90th percentile absolute error: 0.7451
- 95th percentile absolute error: 1.0277
- Test rows at the 5.0 target cap: 141
- Worst individual predictions are saved in `worst_test_predictions.csv`.

The hardest cases tend to be high-value districts, capped target values, and points with unusual occupancy, room, or geographic patterns. Large residuals are expected there because the model only sees aggregate district statistics, not direct indicators such as school quality, coastline proximity, local zoning, crime rates, or employment-center access.

## 9. Conclusion and Future Work

This workflow covered loading, EDA, leakage-safe preprocessing, baseline MLP training, manual feature engineering, enhanced MLP training, test evaluation, and error analysis. Standardization was necessary because feature scales differ substantially. Early stopping and checkpointing helped select the best validation model instead of the last epoch.

Can feature engineering still improve deep learning models on structured tabular datasets? Yes. MLPs can learn nonlinear interactions, but tabular datasets are often small enough and structured enough that human-designed ratios, transforms, and geographic features can still improve generalization or training stability. In this experiment, the empirical comparison above is the deciding evidence: the engineered model improved RMSE and R2.

Future improvements include wider hyperparameter search, batch normalization, Huber loss for outlier robustness, target transformation or explicit handling of the 5.0 cap, geographic clustering, regularization tuning, ensembles, and comparison against strong tabular baselines such as gradient-boosted trees.
