from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
from enum import Enum
import os

app = FastAPI(
    title="Analytics Service",
    description="Advanced analytics and reporting for trading platform",
    version="1.0.0"
)

# CORS configuration from environment
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs from environment
PORTFOLIO_SERVICE_URL = os.getenv("PORTFOLIO_SERVICE_URL", "http://localhost:8004")
PAPER_TRADING_SERVICE_URL = os.getenv("PAPER_TRADING_SERVICE_URL", "http://localhost:8005")

# Models
class TimeFrame(str, Enum):
    ONE_WEEK = "1w"
    ONE_MONTH = "1m"
    THREE_MONTHS = "3m"
    SIX_MONTHS = "6m"
    ONE_YEAR = "1y"
    ALL_TIME = "all"

class PerformanceMetrics(BaseModel):
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    current_streak: int
    best_trade: float
    worst_trade: float

class RiskMetrics(BaseModel):
    value_at_risk_95: float  # 95% VaR
    value_at_risk_99: float  # 99% VaR
    conditional_var_95: float  # CVaR/Expected Shortfall
    volatility: float  # Annualized
    beta: Optional[float] = None
    alpha: Optional[float] = None
    downside_deviation: float
    calmar_ratio: float

class AnalyticsResponse(BaseModel):
    performance: PerformanceMetrics
    risk: RiskMetrics
    period: str
    start_date: datetime
    end_date: datetime
    portfolio_value: float

# Helper functions
def sanitize_float(value: float, default: float = 0.0) -> float:
    """Replace NaN and Inf with default value for JSON serialization"""
    import math
    if math.isnan(value) or math.isinf(value):
        return default
    return value

def fetch_portfolio_data():
    """Fetch portfolio data from Portfolio Service"""
    try:
        response = requests.get(f"{PORTFOLIO_SERVICE_URL}/api/portfolio")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio data: {str(e)}")

def fetch_transactions():
    """Fetch transaction history from Portfolio Service"""
    try:
        response = requests.get(f"{PORTFOLIO_SERVICE_URL}/api/portfolio/transactions")
        response.raise_for_status()
        data = response.json()
        # Portfolio Service returns {"transactions": [...]}
        return data.get("transactions", [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch transactions: {str(e)}")

def calculate_returns(transactions: List[Dict]) -> pd.Series:
    """Calculate daily returns from transactions"""
    if not transactions:
        return pd.Series([0])
    
    df = pd.DataFrame(transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Calculate P&L for each transaction
    df['pnl'] = df.apply(lambda x: 
        (x['price'] - x.get('cost_basis', x['price'])) * x['quantity'] 
        if x['type'] == 'sell' else 0, axis=1)
    
    # Group by date and calculate daily returns
    daily_pnl = df.groupby(df['timestamp'].dt.date)['pnl'].sum()
    
    return daily_pnl

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe Ratio"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if excess_returns.std() == 0:
        return 0.0
    
    return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino Ratio (uses downside deviation)"""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / 252)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return np.sqrt(252) * (excess_returns.mean() / downside_std)

def calculate_max_drawdown(returns: pd.Series) -> tuple:
    """Calculate maximum drawdown and duration"""
    if len(returns) < 2:
        return 0.0, 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()

    # Calculate duration
    dd_duration = 0
    current_duration = 0
    for dd in drawdown:
        if dd < 0:
            current_duration += 1
            dd_duration = max(dd_duration, current_duration)
        else:
            current_duration = 0

    return abs(max_dd), dd_duration

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk"""
    if len(returns) < 2:
        return 0.0
    return abs(np.percentile(returns, (1 - confidence) * 100))

def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall)"""
    if len(returns) < 2:
        return 0.0
    var = calculate_var(returns, confidence)
    return abs(returns[returns <= -var].mean())

# API Endpoints
@app.get("/")
async def root():
    return {
        "service": "Analytics Service",
        "version": "1.0.0",
        "status": "running",
        "port": 9007
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/analytics/performance/{timeframe}", response_model=AnalyticsResponse)
async def get_performance_analytics(timeframe: TimeFrame):
    """Get comprehensive performance analytics for specified timeframe"""

    # Fetch data
    portfolio = fetch_portfolio_data()
    transactions = fetch_transactions()

    if not transactions:
        raise HTTPException(status_code=404, detail="No transaction data available")

    # Filter transactions by timeframe
    end_date = datetime.now()
    if timeframe == TimeFrame.ONE_WEEK:
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == TimeFrame.ONE_MONTH:
        start_date = end_date - timedelta(days=30)
    elif timeframe == TimeFrame.THREE_MONTHS:
        start_date = end_date - timedelta(days=90)
    elif timeframe == TimeFrame.SIX_MONTHS:
        start_date = end_date - timedelta(days=180)
    elif timeframe == TimeFrame.ONE_YEAR:
        start_date = end_date - timedelta(days=365)
    else:  # ALL_TIME
        start_date = min([datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
                         for t in transactions])

    # Filter transactions
    filtered_transactions = [
        t for t in transactions
        if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) >= start_date
    ]

    if not filtered_transactions:
        raise HTTPException(status_code=404, detail="No transactions in specified timeframe")

    # Calculate returns
    returns = calculate_returns(filtered_transactions)

    # Calculate performance metrics
    total_return = returns.sum()
    days = (end_date - start_date).days
    annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd, dd_duration = calculate_max_drawdown(returns)

    # Trade statistics
    winning_trades = [t for t in filtered_transactions if t.get('pnl', 0) > 0]
    losing_trades = [t for t in filtered_transactions if t.get('pnl', 0) < 0]

    total_trades = len(filtered_transactions)
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0

    avg_win = np.mean([t.get('pnl', 0) for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([abs(t.get('pnl', 0)) for t in losing_trades]) if losing_trades else 0
    profit_factor = (avg_win * win_count) / (avg_loss * loss_count) if loss_count > 0 and avg_loss > 0 else 0

    best_trade = max([t.get('pnl', 0) for t in filtered_transactions]) if filtered_transactions else 0
    worst_trade = min([t.get('pnl', 0) for t in filtered_transactions]) if filtered_transactions else 0

    # Calculate current streak
    current_streak = 0
    for t in reversed(filtered_transactions):
        pnl = t.get('pnl', 0)
        if pnl > 0:
            current_streak += 1
        elif pnl < 0:
            current_streak -= 1
        else:
            break

    # Calculate risk metrics
    var_95 = calculate_var(returns, 0.95)
    var_99 = calculate_var(returns, 0.99)
    cvar_95 = calculate_cvar(returns, 0.95)
    volatility = returns.std() * np.sqrt(252)  # Annualized

    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

    calmar_ratio = annualized_return / max_dd if max_dd > 0 else 0

    performance = PerformanceMetrics(
        total_return=sanitize_float(total_return),
        annualized_return=sanitize_float(annualized_return),
        sharpe_ratio=sanitize_float(sharpe),
        sortino_ratio=sanitize_float(sortino),
        max_drawdown=sanitize_float(max_dd),
        max_drawdown_duration=int(dd_duration),
        win_rate=sanitize_float(win_rate),
        profit_factor=sanitize_float(profit_factor),
        average_win=sanitize_float(avg_win),
        average_loss=sanitize_float(avg_loss),
        total_trades=total_trades,
        winning_trades=win_count,
        losing_trades=loss_count,
        current_streak=current_streak,
        best_trade=sanitize_float(best_trade),
        worst_trade=sanitize_float(worst_trade)
    )

    risk = RiskMetrics(
        value_at_risk_95=sanitize_float(var_95),
        value_at_risk_99=sanitize_float(var_99),
        conditional_var_95=sanitize_float(cvar_95),
        volatility=sanitize_float(volatility),
        downside_deviation=sanitize_float(downside_deviation),
        calmar_ratio=sanitize_float(calmar_ratio)
    )

    return AnalyticsResponse(
        performance=performance,
        risk=risk,
        period=timeframe.value,
        start_date=start_date,
        end_date=end_date,
        portfolio_value=portfolio.get('total_value', 0)
    )

@app.get("/api/analytics/returns/{timeframe}")
async def get_returns_distribution(timeframe: TimeFrame):
    """Get returns distribution data for visualization"""
    transactions = fetch_transactions()

    if not transactions:
        raise HTTPException(status_code=404, detail="No transaction data available")

    # Filter by timeframe
    end_date = datetime.now()
    if timeframe == TimeFrame.ONE_WEEK:
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == TimeFrame.ONE_MONTH:
        start_date = end_date - timedelta(days=30)
    elif timeframe == TimeFrame.THREE_MONTHS:
        start_date = end_date - timedelta(days=90)
    elif timeframe == TimeFrame.SIX_MONTHS:
        start_date = end_date - timedelta(days=180)
    elif timeframe == TimeFrame.ONE_YEAR:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = min([datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
                         for t in transactions])

    filtered_transactions = [
        t for t in transactions
        if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) >= start_date
    ]

    returns = calculate_returns(filtered_transactions)

    return {
        "returns": returns.tolist(),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "positive_count": int((returns > 0).sum()),
        "negative_count": int((returns < 0).sum())
    }

@app.get("/api/analytics/drawdown/{timeframe}")
async def get_drawdown_chart(timeframe: TimeFrame):
    """Get drawdown chart data"""
    transactions = fetch_transactions()

    if not transactions:
        raise HTTPException(status_code=404, detail="No transaction data available")

    # Filter by timeframe (same logic as above)
    end_date = datetime.now()
    if timeframe == TimeFrame.ONE_WEEK:
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == TimeFrame.ONE_MONTH:
        start_date = end_date - timedelta(days=30)
    elif timeframe == TimeFrame.THREE_MONTHS:
        start_date = end_date - timedelta(days=90)
    elif timeframe == TimeFrame.SIX_MONTHS:
        start_date = end_date - timedelta(days=180)
    elif timeframe == TimeFrame.ONE_YEAR:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = min([datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
                         for t in transactions])

    filtered_transactions = [
        t for t in transactions
        if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) >= start_date
    ]

    returns = calculate_returns(filtered_transactions)
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return {
        "dates": [str(d) for d in drawdown.index],
        "drawdown": drawdown.tolist(),
        "max_drawdown": float(drawdown.min())
    }

@app.get("/api/analytics/equity-curve/{timeframe}")
async def get_equity_curve(timeframe: TimeFrame):
    """Get equity curve data"""
    portfolio = fetch_portfolio_data()
    transactions = fetch_transactions()

    if not transactions:
        raise HTTPException(status_code=404, detail="No transaction data available")

    # Filter by timeframe
    end_date = datetime.now()
    if timeframe == TimeFrame.ONE_WEEK:
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == TimeFrame.ONE_MONTH:
        start_date = end_date - timedelta(days=30)
    elif timeframe == TimeFrame.THREE_MONTHS:
        start_date = end_date - timedelta(days=90)
    elif timeframe == TimeFrame.SIX_MONTHS:
        start_date = end_date - timedelta(days=180)
    elif timeframe == TimeFrame.ONE_YEAR:
        start_date = end_date - timedelta(days=365)
    else:
        start_date = min([datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00'))
                         for t in transactions])

    filtered_transactions = [
        t for t in transactions
        if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')) >= start_date
    ]

    # Calculate cumulative equity
    df = pd.DataFrame(filtered_transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    starting_capital = 100000  # Default starting capital
    df['equity'] = starting_capital

    cumulative_pnl = 0
    equity_values = []
    dates = []

    for _, row in df.iterrows():
        pnl = row.get('pnl', 0)
        cumulative_pnl += pnl
        equity = starting_capital + cumulative_pnl
        equity_values.append(equity)
        dates.append(row['timestamp'].isoformat())

    return {
        "dates": dates,
        "equity": equity_values,
        "starting_capital": starting_capital,
        "current_equity": equity_values[-1] if equity_values else starting_capital
    }

@app.get("/api/analytics/monthly-returns/{year}")
async def get_monthly_returns(year: int):
    """Get monthly returns for a specific year"""
    transactions = fetch_transactions()

    if not transactions:
        raise HTTPException(status_code=404, detail="No transaction data available")

    # Filter by year
    filtered_transactions = [
        t for t in transactions
        if datetime.fromisoformat(t['timestamp'].replace('Z', '+00:00')).year == year
    ]

    if not filtered_transactions:
        raise HTTPException(status_code=404, detail=f"No transactions in year {year}")

    df = pd.DataFrame(filtered_transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month

    monthly_returns = {}
    for month in range(1, 13):
        month_data = df[df['month'] == month]
        if not month_data.empty:
            monthly_return = month_data['pnl'].sum() if 'pnl' in month_data.columns else 0
            monthly_returns[month] = float(monthly_return)
        else:
            monthly_returns[month] = 0.0

    return {
        "year": year,
        "monthly_returns": monthly_returns,
        "total_return": sum(monthly_returns.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9007)

