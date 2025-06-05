
## Regression

Regression is a supervised learning task where the model predicts continuous outcomes.

![[Pasted image 20250515113321.png]]

>"There are not necessalry good or bad mails, but there is a likelyhood if a mail is good or bad."
>
### Support Vector Regression (SVR)

SVR is an extension of SVM for regression problems. It tries to fit the best line within a threshold (epsilon) that captures most data points.

- **Linear SVR**: Fits a straight line to the data, minimizing the error within the epsilon margin.
    
- **Non-Linear SVR**: Uses kernel functions to capture complex relationships.
    

> [!CITE] Support Vector Machine - Wikipedia  
> "Support vector machines can efficiently perform non-linear classification using the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces."

### SVR in 2D Space

In a 2D space, SVR predicts a continuous value for each data point. The gradient between points indicates the direction and rate of change.

> [!TIP] Understanding SVR Output  
> The color gradient in SVR plots represents the predicted values, with different shades indicating varying levels of the target variable.

---

## Radial Basis Function (RBF) Kernel

The RBF kernel is a popular choice for non-linear SVM and SVR models. It maps input features into higher-dimensional spaces, allowing the model to capture complex patterns.

- **Definition**:  
    K(x, x') = exp(-γ ||x - x'||²)  
    where γ is a parameter that defines the influence of a single training example.
    

> [!CITE] Radial Basis Function Kernel - Wikipedia  
> "The RBF kernel on two samples x and x', represented as feature vectors in some input space, is defined as K(x, x') = exp(-γ||x - x'||²)."

> [!TIP] Choosing γ in RBF Kernel  
> A small γ value implies a larger similarity radius, leading to smoother decision boundaries. A large γ value focuses more on the exact match, potentially leading to overfitting.

---
