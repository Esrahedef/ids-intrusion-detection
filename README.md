#  Intrusion Detection System (IDS) using Machine Learning



Machine learning-based intrusion detection system using the **CIC-IDS2017 dataset** with realistic evaluation and optimized **recall–false alarm trade-off**.



---



##  Dataset



* Source: CIC-IDS2017

* Includes:



  * 🟢 Benign traffic

  * 🔴 DDoS attacks

  * 🔴 Brute force attacks

  * 🔴 Web attacks



---



##  Key Challenges Solved



* ⚠️ Severe class imbalance (up to 80:1)

* ⚠️ Distribution shift between attack types

* ⚠️ Data leakage from flow-based features

* ⚠️ Unrealistic evaluation from random splitting



---



##  Approach



### 1. Data Splitting (Critical)

* Used **day-based split**

* Ensured both attack types (DDoS + brute force) appear in training

* Prevented unrealistic evaluation



### 2. Preprocessing

* Removed leakage-prone features

* Handled missing/infinite values

* Applied median imputation



### 3. Imbalance Handling

* SMOTE (oversampling minority class)

* Random undersampling (majority class)

### 4. Model

* Random Forest Classifier

* Tuned for stability and generalization


### 5. Threshold Optimization

* Optimized decision threshold using F1-score

* Focused on **recall vs false alarm trade-off**


##  Final Results

| Metric             | Value |

| ------------------ | ----- |

| ROC-AUC            | 0.938 |

| Accuracy           | 0.79  |

| Attack Recall      | 0.63  |

| Precision (Attack) | 0.99  |

| False Alarm Rate   | 1.0%  |

---

## Trade-off Insight

* High precision → very low false alarms

* Moderate recall → some attacks missed

 This reflects **real-world IDS behavior**

---
##  Key Insight

> Random splitting led to misleading results.

> Proper evaluation required **time-based splitting and attack-type coverage**.

--

## Future Work

* Multi-class attack classification

* Real-time detection system

* Streamlit dashboard

* SIEM integration

---

##  Installation

```bash

pip install -r requirements.txt

---
##  Run

```bash
python src/train.py

---
## Author
Developed as a machine learning security project

Focused on realistic evaluation and deployment scenarios
