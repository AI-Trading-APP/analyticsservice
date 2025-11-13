# Analytics Service

Advanced analytics and reporting service for the AI Trading Platform.

## Features

### Performance Metrics
- **Total Return**: Overall portfolio return
- **Annualized Return**: Return normalized to yearly basis
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Trade Statistics**: Average win/loss, best/worst trades

### Risk Metrics
- **Value at Risk (VaR)**: 95% and 99% confidence levels
- **Conditional VaR (CVaR)**: Expected shortfall beyond VaR
- **Volatility**: Annualized standard deviation of returns
- **Downside Deviation**: Volatility of negative returns
- **Calmar Ratio**: Return to max drawdown ratio
- **Beta & Alpha**: Market-relative performance (optional)

## Setup

### Installation
```bash
cd analyticsservice
pip install -r requirements.txt
```

### Run Service
```bash
python main.py
```

Service runs on **http://localhost:8007**

## API Endpoints

### Health Check
```bash
GET /health
```

### Get Performance Analytics
```bash
GET /api/analytics/performance/{timeframe}
```

**Timeframes:**
- `1w` - One week
- `1m` - One month
- `3m` - Three months
- `6m` - Six months
- `1y` - One year
- `all` - All time

**Example:**
```bash
curl http://localhost:8007/api/analytics/performance/1m
```

**Response:**
```json
{
  "performance": {
    "total_return": 0.15,
    "annualized_return": 0.45,
    "sharpe_ratio": 1.8,
    "sortino_ratio": 2.3,
    "max_drawdown": 0.08,
    "max_drawdown_duration": 5,
    "win_rate": 0.65,
    "profit_factor": 2.1,
    "average_win": 150.50,
    "average_loss": 75.25,
    "total_trades": 20,
    "winning_trades": 13,
    "losing_trades": 7,
    "current_streak": 3,
    "best_trade": 500.00,
    "worst_trade": -200.00
  },
  "risk": {
    "value_at_risk_95": 0.02,
    "value_at_risk_99": 0.035,
    "conditional_var_95": 0.028,
    "volatility": 0.18,
    "downside_deviation": 0.12,
    "calmar_ratio": 5.6
  },
  "period": "1m",
  "start_date": "2025-10-13T00:00:00",
  "end_date": "2025-11-13T00:00:00",
  "portfolio_value": 105000.00
}
```

## API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8007/docs
- **ReDoc**: http://localhost:8007/redoc

## Dependencies

- **FastAPI**: Web framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Chart generation
- **ReportLab**: PDF report generation
- **Requests**: HTTP client for service communication

## Integration

The Analytics Service integrates with:
- **Portfolio Service** (Port 8004): Fetches portfolio data and transactions
- **Paper Trading Service** (Port 8005): Fetches paper trading history

## Metrics Explained

### Sharpe Ratio
Measures risk-adjusted return. Higher is better.
- < 1: Poor
- 1-2: Good
- 2-3: Very good
- \> 3: Excellent

### Sortino Ratio
Similar to Sharpe but only considers downside volatility. Higher is better.

### Maximum Drawdown
Largest peak-to-trough decline. Lower is better.
- < 10%: Excellent
- 10-20%: Good
- 20-30%: Acceptable
- \> 30%: High risk

### Calmar Ratio
Return divided by max drawdown. Higher is better.
- \> 3: Excellent
- 2-3: Good
- 1-2: Acceptable
- < 1: Poor

### Value at Risk (VaR)
Maximum expected loss at given confidence level.
- 95% VaR of 2% means 95% confidence losses won't exceed 2%

## Future Enhancements

- [ ] PDF report generation
- [ ] Email report delivery
- [ ] Benchmark comparison (S&P 500)
- [ ] Monte Carlo simulation
- [ ] Performance attribution
- [ ] Factor analysis
- [ ] Custom report templates

## Port

**8007**

