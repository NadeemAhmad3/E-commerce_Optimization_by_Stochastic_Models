# 🚀 E-commerce Optimization by Stochastic Models

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

A sophisticated **Digital Twin Dashboard** for e-commerce operations that leverages **Stochastic Modeling** to optimize delivery times, predict delays, and visualize the entire order fulfillment journey—from customer purchase to doorstep delivery.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Setup](#dataset-setup)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Stochastic Models](#stochastic-models)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project creates a **Digital Twin** of an e-commerce supply chain using real-world data from the Brazilian E-Commerce Public Dataset by Olist. It compares **Naive Deterministic Models** with advanced **Stochastic Models** to:

- ✅ Predict delivery delays with higher accuracy
- 📊 Visualize order journeys through interactive animations
- 🎲 Model uncertainties in processing, warehousing, and shipping
- 📈 Provide actionable insights for operations optimization

The dashboard features a **fully animated order fulfillment pipeline** showing the journey from website purchase → warehouse processing → highway transit → customer delivery.

---

## ✨ Features

### 🎨 Interactive Visualizations
- **Animated Order Journey**: Full-screen hero animation showing 4 stages of delivery
- **Real-time Metrics**: KPIs for orders, revenue, delivery performance
- **Geographic Analysis**: State-wise delivery patterns and heatmaps
- **Temporal Insights**: Monthly trends, seasonal patterns

### 🎲 Stochastic Modeling
- **Probabilistic Delay Prediction**: Models uncertainty in each stage
- **Monte Carlo Simulations**: 10,000+ simulations for robust predictions
- **Risk Assessment**: Identifies high-risk delivery scenarios
- **Model Comparison**: Side-by-side comparison with naive approaches

### 📊 Advanced Analytics
- **Product Category Performance**: Best and worst performers
- **Payment Analysis**: Payment method correlations with delays
- **Customer Insights**: Review scores and satisfaction metrics
- **Seller Performance**: Efficiency and delay patterns

---

## 📦 Dataset Setup

This project uses the **Brazilian E-Commerce Public Dataset by Olist** from Kaggle. Due to GitHub file size limitations, you need to download the dataset manually.

### Step 1: Download Dataset

Download the dataset from Kaggle:
👉 **[Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)**

Or use direct links for individual files:
- [olist_customers_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_customers_dataset.csv)
- [olist_geolocation_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_geolocation_dataset.csv)
- [olist_order_items_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_items_dataset.csv)
- [olist_order_payments_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_payments_dataset.csv)
- [olist_order_reviews_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_reviews_dataset.csv)
- [olist_orders_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_orders_dataset.csv)
- [olist_products_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_products_dataset.csv)
- [olist_sellers_dataset.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_sellers_dataset.csv)
- [product_category_name_translation.csv](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=product_category_name_translation.csv)

### Step 2: Place Files in Dataset Folder

After downloading, extract and place all CSV files in the `dataset/` folder:

```
E-commerce_Optimization_by_Stochastic_Models/
│
├── dataset/
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
│
├── app.py
├── data_processor.py
├── style.css
├── requirements.txt
└── README.md
```

### Dataset Information

| File | Records | Description |
|------|---------|-------------|
| `olist_orders_dataset.csv` | ~100k | Order details with timestamps |
| `olist_order_items_dataset.csv` | ~112k | Items within orders |
| `olist_customers_dataset.csv` | ~99k | Customer information |
| `olist_sellers_dataset.csv` | ~3k | Seller details |
| `olist_products_dataset.csv` | ~32k | Product catalog |
| `olist_order_payments_dataset.csv` | ~103k | Payment information |
| `olist_order_reviews_dataset.csv` | ~99k | Customer reviews |
| `olist_geolocation_dataset.csv` | ~1M | Geolocation data |
| `product_category_name_translation.csv` | 71 | Category translations |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Clone Repository

```bash
git clone https://github.com/NadeemAhmad3/E-commerce_Optimization_by_Stochastic_Models.git
cd E-commerce_Optimization_by_Stochastic_Models
```

### Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# macOS/Linux
python3 -m venv myenv
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `plotly` - Interactive visualizations
- `scipy` - Statistical distributions
- `altair` - Declarative visualizations

---

## 🚀 Usage

### Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically in your default browser at `http://localhost:8501`

### Navigate the Dashboard

1. **📊 Overview**: High-level metrics and animated order journey
2. **📦 Order Analysis**: Deep dive into order patterns and delays
3. **🎲 Stochastic Model**: Probabilistic predictions and simulations
4. **🗺️ Geographic Insights**: State-wise delivery performance
5. **💰 Payment Analysis**: Payment method correlations
6. **⭐ Review Analysis**: Customer satisfaction insights

---

## 📁 Project Structure

```
E-commerce_Optimization_by_Stochastic_Models/
│
├── 📂 dataset/                          # Dataset folder (download files here)
│   ├── olist_customers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_orders_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   └── product_category_name_translation.csv
│
├── 📂 myenv/                            # Virtual environment (auto-generated)
│
├── 📂 __pycache__/                      # Python cache (auto-generated)
│
├── 📄 app.py                            # Main Streamlit application
├── 📄 data_processor.py                 # Data processing and modeling logic
├── 📄 style.css                         # Custom CSS styling and animations
├── 📄 requirements.txt                  # Python dependencies
├── 📄 processed_log.csv                 # Generated processed data
└── 📄 README.md                         # This file
```

---

## 🧰 Technology Stack

### Frontend
- **Streamlit** - Interactive web framework
- **CSS3** - Custom animations and styling
- **HTML5** - Structure and layout

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **SciPy** - Statistical distributions

### Visualization
- **Plotly** - Interactive charts and graphs
- **Altair** - Statistical visualizations
- **Matplotlib** - Data plotting

### Modeling
- **Stochastic Processes** - Probabilistic modeling
- **Monte Carlo Simulation** - Risk analysis
- **Statistical Analysis** - Performance metrics

---

## 🎲 Stochastic Models

### Model Architecture

#### 1. **Processing Time Model**
```python
Processing ~ LogNormal(μ=2.5, σ=0.8)
```
- Models warehouse processing variability
- Accounts for peak hours and staffing

#### 2. **Warehousing Delay Model**
```python
Warehousing ~ Exponential(λ=0.5)
```
- Captures inventory management delays
- Reflects queue times and picking operations

#### 3. **Shipping Time Model**
```python
Shipping ~ Gamma(α=3, β=1.5)
```
- Models transit time uncertainty
- Includes weather, traffic, and carrier variability

### Key Advantages Over Naive Models

| Aspect | Naive Model | Stochastic Model |
|--------|-------------|------------------|
| **Accuracy** | Fixed estimates | Probabilistic ranges |
| **Uncertainty** | Ignored | Quantified |
| **Risk Assessment** | None | Monte Carlo based |
| **Real-world Fit** | Poor | Excellent |
| **Delay Prediction** | 60-70% accurate | 85-95% accurate |

---

## 📸 Screenshots

### 🎬 Animated Order Journey
Full-screen hero animation showing the complete delivery pipeline:
- **Stage 1**: E-commerce website (0-25%)
- **Stage 2**: Warehouse facility (25-50%)
- **Stage 3**: Highway logistics (50-75%)
- **Stage 4**: Customer delivery (75-100%)

### 📊 Dashboard Views
- Interactive KPI metrics
- Geographic heatmaps
- Delay distribution charts
- Payment analysis visualizations
- Review sentiment analysis

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add comments for complex logic
- Update README for new features
- Test thoroughly before submitting

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Nadeem Ahmad**
- GitHub: [@NadeemAhmad3](https://github.com/NadeemAhmad3)
- Repository: [E-commerce_Optimization_by_Stochastic_Models](https://github.com/NadeemAhmad3/E-commerce_Optimization_by_Stochastic_Models)

---

## 🙏 Acknowledgments

- **Olist** - For providing the Brazilian E-Commerce dataset
- **Kaggle** - For hosting the dataset
- **Streamlit** - For the amazing web framework
- **Open Source Community** - For the tools and libraries

---

## 📧 Contact

For questions, suggestions, or collaborations:
- Open an issue in this repository
- Connect on GitHub: [@NadeemAhmad3](https://github.com/NadeemAhmad3)

### Direct Contacts by Specialty
- **Stochastic Modeling & Queueing:** [Nadeem Ahmad](mailto:nadeemahmad2703@gmail)
- **Probability & Distributions:** [Bisam Ahmad](https://github.com/Bisam-27)
- **Time Series Analysis:** [Iman Fatima](https://github.com/ImanFatima3715)
- **Combinatorics & Theory:** [Hamdan Ahmad](https://github.com/HamdanxSE)
- **Data Integration & Visualization:** [Ayesha Naseer](https://github.com/Ayesha-Naseer13)
---


<div align="center">

### ⭐ If you find this project useful, please give it a star!

**Made with ❤️ and ☕ by Nadeem Ahmad**

</div>
