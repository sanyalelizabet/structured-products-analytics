"""

Reverse Convertible Module

==========================



This module implements a product-level analytics engine for reverse convertibles

and multi-asset reverse convertibles.



The main class, `ReverseConvertible`, takes a single portfolio row as input and

computes the payoff and return profile of the product based on its contractual

terms and underlying performance.



Expected input

\--------------

The class expects one row of a pandas DataFrame (typically from `portfolio.iterrows()`),

with the following fields:



Required fields

\---------------

\- product\_id : str

&#x20;   Unique product identifier ISIN.



\- product\_type : str

&#x20;   Product type, e.g. "BRC" or "MBRC".



\- type\_style : str

&#x20;   Barrier observation style, e.g. "European" or "American".



\- position\_units : int or float

&#x20;   Number of product units held.



\- notional : int or float

&#x20;   Notional amount per product unit.



\- cost\_price : float

&#x20;   Purchase price as fraction of nominal, e.g. 0.98 or 1.00.



\- coupon : float

&#x20;   Coupon rate as decimal, e.g. 0.08 for 8% p.a. .



\- barrier\_pct : float

&#x20;   Barrier level as fraction of initial fixing, e.g. 0.60 for 60%.



\- initial\_fixing\_date : str or datetime-like

&#x20;   Product start / fixing date.



\- maturity\_date : str or datetime-like

&#x20;   Product maturity date.



\- underlyings : list\[str]

&#x20;   List of underlying names.



\- underlying\_isins : list\[str]

&#x20;   List of ISINs corresponding to the underlyings.



\- initial\_levels : list\[float]

&#x20;   Initial fixing levels of the underlyings.



\- strike : list\[float]

&#x20;   Strike levels of the underlyings.



\- current\_spots : list\[float]

&#x20;   Current spot levels of the underlyings.



Optional fields

\----------------

* final\_levels : list\[float], optional

&#x20;   Scenario shocks in percent applied to current spot levels.

&#x20;   For example, -10 means the underlying is shocked down by 10%.

&#x20;   These shocks are converted internally into scenario prices.

Notes on dimensional consistency

\--------------------------------

For multi-asset products, the following lists must have the same length:

\- underlyings

\- underlying\_isins

\- initial\_levels

\- current\_spots

\- final\_levels (if provided)



For single-underlying products, these fields should still be provided as

one-element lists.



Example input row

\-----------------

{

&#x20;   "product\_id": "CH1483491150",

&#x20;   "product\_type": "BRC",

&#x20;   "type\_style": "European",

&#x20;   "position\_units": 10,

&#x20;   "notional": 1000,

&#x20;   "cost\_price": 0.98,

&#x20;   "coupon": 0.08,

&#x20;   "barrier\_pct": 0.60,

&#x20;   "initial\_fixing\_date": "2024-01-15",

&#x20;   "maturity\_date": "2025-01-15",

&#x20;   "underlyings": \["ALCON"],

&#x20;   "underlying\_isins": \["CH0432492467"],

&#x20;   "initial\_levels": \[59.72],

&#x20;   "strike": \[59.72],

&#x20;   "current\_spots": \[58.76]

}



Main analytics implemented

\--------------------------

The class calculates:

\- current underlying performances

\- final/scenario performances

\- worst-of underlying for multi-asset products

\- barrier breach status

\- payoff per unit

\- total payoff

\- total cost

\- profit and loss

\- return in percentage terms

\- annualized return

\- distance to barrier

\- break-even level





If `final\_levels` is not provided, the class assumes a 0% scenario shock for each

underlying. As a result, the internally calculated final levels are equal to the

current spot levels.





Design idea

\-----------

This class represents the product engine of the framework.



It is responsible for product-level payoff and valuation logic only.

Portfolio aggregation and reporting are handled separately by the

`PortfolioAnalytics` class.

"""

