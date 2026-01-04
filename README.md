# Candlestick-Based Stock Selection Model

**Highlights**
- Constructed cross-sectional equity factors from candlestick shadow patterns
- Implemented quintile portfolio backtests on S&P 500 universe
- Achieved consistent benchmark outperformance using upper-shadow signals

This project develops **systematic stock selection factors** based on **candlestick (K-line) shadow patterns**
and evaluates their effectiveness through **cross-sectional portfolio backtesting**.

We focus on whether **upper and lower shadow lines**, which reflect market buying/selling pressure,
can be transformed into **alpha-generating stock selection signals**.

---

## 📌 Motivation

Candlestick charts are widely used in technical analysis to capture **market sentiment**.

- Long **upper shadows** → strong selling pressure  
- Long **lower shadows** → strong buying support  

While traditionally used for **market timing**, this project explores whether
shadow-line information can be used for **cross-sectional stock selection**.

---

## 📊 Data

- **Universe**: S&P 500 constituent stocks
- **Benchmark**: S&P 500 Index
- **Period**: Jan 2015 – Dec 2019  
  (pre-COVID period to avoid structural breaks)
- **Source**: Yahoo Finance (via `yfinance`)

---

## 🧮 Factor Construction

### 1️⃣ Candlestick Shadow Factors

For each stock and trading day:

- **Upper Shadow**: `High - max(Open, Close)`
- **Lower Shadow**: `min(Open, Close) - Low`

To normalize scale effects, we standardize shadow lengths using a **5-day rolling mean**.

Monthly factors are constructed using:
- Mean of standardized shadows
- Standard deviation of standardized shadows

---

### 2️⃣ Williams Shadow Factors

Inspired by the Williams %R indicator, we define:

- **Williams Upper Shadow**: `High − Close`
- **Williams Lower Shadow**: `Close − Low`

The same monthly aggregation and ranking procedure is applied.

---

## 🧠 Portfolio Construction

- Stocks are **ranked monthly** by factor value
- Split into **5 portfolios (quintiles)**
- **Equal-weighted** portfolios
- Monthly rebalancing
- **Long–short hedge**: Portfolio 5 − Portfolio 1

---

## 📈 Backtesting Metrics

Performance is evaluated using:

- Annual return
- Annual volatility
- Sharpe ratio
- Information ratio (IR)
- Maximum drawdown
- Alpha vs benchmark

---

## 🏆 Key Results

### Candlestick Shadow Factors
- **Upper Shadow Mean** performs best among candlestick-based factors
- Long–short portfolio delivers:
- Positive excess return
- Higher Sharpe ratio than benchmark
- Controlled drawdown despite market stress periods

### Williams-Based Factors
- Traditional lower shadow factor performs poorly
- **Williams Lower Shadow (Std)** significantly improves performance
- Acts as a **complementary signal** to candlestick shadows

---

## 🔍 Visualization

- Portfolio performance curves vs benchmark
- Excess return plots
- Candlestick charts of top-performing stocks during best-performing months
- Confirms consistency between factor signals and subsequent price trends

---

## 🧾 Conclusion

- Candlestick shadow information contains **useful cross-sectional signals**
- **Upper shadow–based factors** show stronger stock selection power
- Williams-based shadow factors help correct weaknesses in standard candlestick indicators
- While alpha is modest, results consistently outperform the benchmark

---

## 🚀 Future Improvements

- Test alternative rebalancing frequencies (daily vs monthly)
- Explore different normalization windows
- Incorporate volume-weighted portfolios
- Combine candlestick and Williams factors into composite signals

---

## 👥 Contributors

- Chuyi Wang  
- Yuxuan Zhou  
- **Linglan Xu**  
- Yiming Ding  
- Yitao Huang
