### League of Legends Match Outcome Predictor

### Project Overview
    This project builds a machine learning–based prediction system to determine the outcome of a League of Legends (LoL) match (Win / Loss) using historical match statistics and team performance features.

    The model is implemented from scratch using PyTorch, applying logistic regression to a structured esports dataset. The project demonstrates the complete ML workflow—from data preprocessing to training, evaluation, optimization, and   
    interpretation.


### Objectives

    By completing this project, the following goals are achieved:
    
    Load and preprocess structured esports match data
    
    Perform feature scaling and train-test splitting
    
    Implement logistic regression using PyTorch
    
    Train and optimize the model using gradient descent
    
    Evaluate model performance using:
    
    Accuracy
    
    Precision
    
    Recall
    
    F1-score
    
    Interpret model coefficients for feature importance
    
    Save and reload trained models for reuse


 ### Dataset
    
    Source: Public League of Legends match statistics (e.g., Kaggle or Riot Games API)
    
    Format: CSV
    
    Target variable:
    
    win → Binary label (0 = Loss, 1 = Win)
    
    ```test
    Sample Structure
    league_of_legends_data_large.csv
    ├── kills
    ├── deaths
    ├── assists
    ├── gold_earned
    ├── towers_destroyed
    ├── dragons_taken
    ├── barons_taken
    ├── ...
    └── win
    
    ⚠️ Dataset URL is not included to respect licensing and usage terms.
    ```



### Model Architecture

    Logistic Regression (Binary Classification)
    
    Implemented using PyTorch
    
    Activation function: Sigmoid
    
    Loss function: Binary Cross Entropy (BCE)
    
    Optimizer: Stochastic Gradient Descent (SGD)
    
    Input Features → Linear Layer → Sigmoid → Win / Loss

### Training Strategy

### Baseline Training

    Optimizer: SGD
    
    Learning Rate: 0.001
    
    Epochs: 1000
    
    Feature Scaling: StandardScaler

### Optimized Training

    Momentum: 0.9
    
    Weight Decay: 0.01 (regularization)
    
    Improved convergence and generalization

### Hyperparameter Tuning

    Tested learning rates: [0.01, 0.05, 0.1]
    
    Selected best model based on test accuracy

### Evaluation Metrics

    The model is evaluated on unseen test data using:
    
    Accuracy
    
    Precision
    
    Recall
    
    F1-score
    
    Classification Report
    
    Labels:
    0 → Loss
    1 → Win


### Feature Importance

    Feature importance is derived from logistic regression weights:
    
    Positive weights → Increase win probability
    
    Negative weights → Decrease win probability
    
    This allows interpretability and insight into which in-game statistics most strongly influence match outcomes.

### Model Persistence

    The trained model is saved and reloaded using:
    
    torch.save(model.state_dict(), "league_match_predictor.pth")
    
    
    This enables:
    
    Model reuse
    
    Deployment readiness
    
    Reproducibility


 ### Repository Structure
 
 ```test
  league-of-legends-match-predictor/
  │
  ├── league_match_predictor.ipynb
  ├── league_match_predictor.pth
  ├── league_of_legends_data_large.csv
  ├── README.md
```

 
 ### Technologies Used

    Python
    
    PyTorch
    
    Pandas
    
    NumPy
    
    Scikit-learn
    
    Matplotlib

### Real-World Applications

    Esports match outcome prediction
    
    Competitive gaming analytics
    
    Strategy optimization for teams
    
    Data-driven decision support in esports

###  Notes

This project is an original implementation based on independent learning and experimentation

No proprietary APIs or credentials are included


### Author

### Amar Kumar
### AI / Machine Learning Enthusiast
