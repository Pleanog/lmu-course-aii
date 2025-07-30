## Online Machine Learning

$\to$ ***A Shift from Batch to Continuous Learning***

Online Machine Learning (OML) represents a fundamental change from the conventional batch learning paradigm. While batch learning involves training a model on a complete dataset with distinct training and testing phases, OML learns incrementally, updating the model continuously as new data arrives.

- **Incremental Updates:** Learning happens with every new data point.
- **Real-time Responsiveness:** Predictions update instantly.
- **Adaptability:** Able to handle changing data patterns (concept drift).
- **Efficiency:** No need to store large volumes of past data.

OML is very suitable for environments where speed, memory constraints, and adaptiveness are critical.

---

## Batch vs. Online Learning: Core Differences

![[Pasted image 20250717110536.png]]


| Aspect                    | Batch/Offline Learning         | Online Learning                          |
| ------------------------- | ------------------------------ | ---------------------------------------- |
| Training                  | Before using the model         | Continuously while using the model       |
| Data Access               | Multiple times during training | One-time only                            |
| Target Function           | Assumed stationary             | Dynamic                                  |
| Data Flow                 | Entire dataset at once         | Continuous stream, one example at a time |
| Training & Testing Phases | Separate                       | Blurred or non-existent                  |

---

## The Gradient Descent Analogy

Online vs. batch learning can be compared to:
**Stochastic Gradient Descent (SGD)** vs. **Gradient Descent (GD)**:

- **Gradient Descent (GD):** Uses the entire dataset to compute each update. It is slow but stable.
- **Stochastic Gradient Descent (SGD):** Uses one sample or mini-batches for frequent updates. It is noisy but adaptive.

![[Pasted image 20250717161427.png]]

> [!TLDR] SGD and Online Learning:  
> Just as SGD updates weights frequently based on single examples, online learning updates models in real time from a stream of data.


---

## Real-World Applications of Online Learning

### When to Use
- **Large-scale data that doesn’t fit in memory**
- **Constantly changing data streams**

### Use Cases
- **Real-time Recommendations:**  
    Platforms like Netflix dynamically adapt user profiles to recent behavior (views, ratings).
- **Anomaly Detection:**  
    Used in fraud/spam detection and cybersecurity to recognize suspicious behavior as it occurs.
- **Adaptive Control Systems:**  
    In domains like autonomous driving, IoT, and robotics where environments constantly change.
- **Robotics:**  
    Robots incrementally learn new object classes or actions in real-time.

---

## Major Challenges in Online Machine Learning

### **Concept Drift**

Concept drift refers to changes in the underlying data distribution, which can reduce prediction accuracy over time.

> [!CITE] Concept Drift Example  
> "A spam model trained on words like 'free' may lose performance as spammers change their vocabulary."

**Adaptation Strategies:**

- Continuously monitor model performance
- Retrain using updated datasets
- Incorporate maintenance policies and updating mechanisms

![[Pasted image 20250717162317.png]]

### **Catastrophic Forgetting**

As new data arrives, the model may forget older learned information.

> [!TIP] Avoiding Forgetting:  
> Mix old and new data during retraining or use ensemble methods to preserve older knowledge.

**Strategies:**
- Evaluate with older datasets
- Maintain hybrid memory-based systems
- Use ensemble models trained on various data segments

### **Latency vs. Performance Trade-off**

| Factor      | Description                                     |
| ----------- | ----------------------------------------------- |
| Latency     | Time taken to respond to new data               |
| Performance | Accuracy and resource-efficiency of predictions |

Real-time systems like fraud detection require **low latency**, but also **high accuracy**, which are often in tension.

> [!TIP] Optimizing Trade-off:  
> Reduce data dimensionality or use faster models. Feature selection helps reduce complexity while maintaining accuracy.

> [!TIP] Ensemble Methods:
> Use multiple models to handle different parts of the data distribution

---

## Ethical and Privacy Concerns

Online learning systems constantly collect and process data, leading to several risks:

- **Data Privacy:** Risk of leaks from continuous collection.
- **Informed Consent:** Users may not be aware of ongoing data usage.
- **Bias Amplification:** Biased input data leads to unfair predictions.
- **Security Threats:** Continuous exposure increases vulnerability.

### Mitigation Strategies

- **Data Anonymization**: Strip PII before processing
- **Secure Transmission**: Encrypt data in transit
- **Federated Learning**: Train models on-device, send only model updates

> 🔍 **Federated Learning:**  
> A privacy-preserving approach where training occurs locally and only model gradients are shared. It avoids raw data transmission, making it ideal for mobile and IoT devices.

---

## Detecting New or Unknown Classes

In dynamic environments, OML must identify when a new class appears that wasn’t part of the training data.

![[Pasted image 20250717195456.png]]

The graphs show the collection of (sound) data over time. As more data is collected the model can form clusters. It can detect user specific new clusters as it in learning from the direct enviroment (e.g. users kitchen sounds)

- Utilize encoder-decoder networks for dimensionality reduction.
	- Similar approach to large language models (LLMs) 
	- Train a domain-specific encoder-decoder network using relevant training data

### Use Case: Kitchen Action Recognition

*see graph above*
1. Features (e.g., from sound) are extracted and mapped into a high-dimensional vector space.
2. Clustering groups similar data (e.g., actions like “cutting” or “stirring”).
3. Unknown actions are detected and labeled through user input.
4. New clusters are formed and models updated.


![[Pasted image 20250717110456.png]]


> [!TIP] Active Learner:  
> The system prompts the user with open-ended or confirmatory questions once it detects a new potential cluster.


![[Pasted image 20250717110440.png]]

### Challenges in Detecting New Classes

- **Sample Size:**  
    Use **data augmentation** or **few-shot learning** to reduce the need for large training sets.

> [!TIP] Few-shot learning enables accurate predictions from very few labeled examples by leveraging pretrained knowledge and similarity metrics.

- **High Dimensionality:**  
    Use **PCA**, **t-SNE**, or **encoder-decoder networks** to reduce data dimensions.   

> [!INFO] t-SNE and PCA work well when test data is similar to training data. Encoder-decoder networks are better for generalizing to unseen inputs.

- **Fixed Clusters:**  
    Prevent misclassification of new samples by training many **one-class classifiers**, each for one known category.

> 🔍 **One-Class Classifiers:**  
> These models only recognize their own class and reject anything that doesn't match. It is useful for novelty detection.


---

## Combining Offline and Online Learning for Human Pose Detection

Human pose data can be used to detect actions. The process typically includes:



![[Pasted image 20250717162130.png]]

### Offline Phase:
- Train encoder-decoder networks on known actions (e.g., cutting, stirring).
- Create latent representations and corresponding classifiers.

### Online Phase:
- Use one-class classifiers to check if a new sample matches a known class.
- If not, label it as new and retrain a new classifier.

> [!CITE] Human Pose Evaluation  
> "Pose detection errors average ~20mm and are more pronounced for extremities like hands or the nose."

---

## Evaluating Online Machine Learning Models

### Metrics

- **Accuracy, Precision, Recall, F1 Score**: standard performance indicators
- **Learning Curves**: shows performance over time/data volume
- **Minimized Regret**: difference between actual model performance and a hypothetical optimal model

> [!TLDR] Minimized Regret:  
> Indicates how much performance is lost due to not having access to future data upfront.

### Guidelines

- Simulate streaming environments during testing
- Evaluate after each data chunk
- Use adaptive learning rates to prevent overfitting or forgetting
- Test on real-world data whenever possible

> [!question] _What is the difference between online learning and batch learning?_  
> Online learning updates the model continuously with each new data point, whereas batch learning trains the model once using the entire dataset. This is analogous to the difference between Stochastic Gradient Descent and Gradient Descent.

---