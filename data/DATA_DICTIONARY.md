Feature Name,Definition,Data Source,Frequency,Point-in-Time Rule,Units,Missing Value Rule,Winsorization Rule
adj_close,Total-return adjusted closing price,Price Feed,Daily,Same-day market close,INR,Drop row if missing,Unwinsorized
forward_20d,Forward 20-trading-day return (Pt+20​/Pt​−1),Price Derived,Daily/Rebal,Look-ahead target,Ratio,Target drop if incomplete,Unwinsorized
mom_60d,60-trading-day cumulative price return,Price Derived,Daily,Historical close,Ratio,NaN if <60 trading days,None
sector_relative_mom_60d,Cross-sectional Z-score of sector-demeaned 60D return,Price Derived,Daily/Rebal,Historical close,Z-score,NaN if sector <2 stocks,Standardized
eps_growth_acceleration,1-quarter delta of YoY EPS growth rate,Filings / Financials,Quarterly,Usable date (Day after filing),Ratio,NaN until 6 quarters back,Standardized
roe_ttm,Trailing 12-Month Net Income divided by 1Y Average Equity,Filings / Financials,Quarterly,Usable date (Day after filing),Ratio,NaN for incomplete 4Q TTM,Standardized
fcf_margin_ttm,TTM Free Cash Flow (OCF−CapEx) / TTM Revenue,Filings / Financials,Quarterly,Usable date (Day after filing),Ratio,Masked to NaN for Financials,Standardized
BaseScore,Z(SectorRelMom60D​)+Z(EPSAccel),Factor Composite,Rebalance,PIT Aligned,Score,Requires both base factors,Equal-weighted