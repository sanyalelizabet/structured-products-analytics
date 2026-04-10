from src.reverse_convertible import ReverseConvertible
import pandas as pd
import numpy as np
import hashlib

class ScenarioEngine:

    def __init__(self, portfolio, beta_map, vol_map, scenarios=None):
        self.portfolio = portfolio
        self.beta_map = beta_map
        self.scenarios = scenarios
        self.vol = vol_map
        
    def get_beta(self, isin):
        return self.beta_map.get(isin, 1.0)
    
    def get_vol(self, isin):
        return self.vol.get(isin, 0.15)


    def build_shock_paths(self, row, scenario, corr_matrix=None, scenario_seed=0):
        """
        Day-by-day correlated GBM simulation from today to maturity.

        Each business day, all underlyings are stepped jointly using a single
        Cholesky-correlated draw — so co-movement between assets is preserved
        throughout the path, not just at the terminal date.

        Each business day:
          1. Identify drift phase (pre-shock / between shocks / post-shock)
          2. Apply correlated GBM step across all assets simultaneously:
               S *= exp((drift - 0.5*vol²)*dt + vol*sqrt(dt)*Z_corr)
          3. Apply discrete shock on shock dates (market_shock, not beta-scaled)

        Parameters
        ----------
        scenario : dict
            market_shock        : float — % index move per shock (e.g. -20)
            n_shocks            : int   — number of shock events (default 1)
            shock_in_days       : int   — calendar days from today to first shock
            shock_spacing_days  : int   — calendar days between consecutive shocks
            pre_shock_drift_pa  : float — annualised drift before first shock
            post_shock_drift_pa : float — annualised drift after last shock to maturity
                                          0.0  → permanent shock (conservative)
                                          0.05 → ERP recovery
                                         -0.10 → bear market continuation

        corr_matrix : np.ndarray of shape (n_assets, n_assets), optional
            Correlation matrix for the underlyings. Must be positive-definite.
            Defaults to the identity matrix (independent assets).

        Volatility is sourced from self.vol (isin-keyed dict, annualised).
        Beta    is sourced from self.beta_map (isin-keyed dict).

        Returns
        -------
        shocks      : list[float]   % change current→terminal per underlying
        final_spots : list[float]   terminal spot per underlying at maturity
        T_remaining : float         years from today to maturity
        path_summary: dict          human-readable audit trail
        paths       : dict[isin → pd.DataFrame(date, price)]
        """
        today              = pd.Timestamp.today().normalize()
        portfolio_maturity = pd.to_datetime(self.portfolio["maturity_date"]).max()
        maturity           = pd.Timestamp(row["maturity_date"])

        T_remaining = (maturity - today).days / 360
        T_total     = (portfolio_maturity - today).days / 360

        # Scenario parameters
        market_shock       = scenario.get("market_shock", 0)
        n_shocks           = scenario.get("n_shocks", 1)
        shock_in_days      = scenario.get("shock_in_days", 0)
        shock_spacing_days = scenario.get("shock_spacing_days", 0)
        pre_shock_drift    = scenario.get("pre_shock_drift_pa", 0.05)
        post_shock_drift   = scenario.get("post_shock_drift_pa", 0.05)

        isins    = list(row["underlying_isins"])
        spots    = np.array([float(s) for s in row["current_spots"]])
        n_assets = len(isins)

        vols  = np.array([self.get_vol(isin)  for isin in isins])   # (n_assets,)
        betas = np.array([self.get_beta(isin) for isin in isins])   # (n_assets,)

        # ── Correlation & Cholesky ────────────────────────────────────────────
        if corr_matrix is None:
            corr_matrix = np.eye(n_assets)
            
        L = np.linalg.cholesky(corr_matrix)   # lower-triangular, (n_assets, n_assets)

        # ── Business-day grid ────────────────────────────────────────────────
        date_range = pd.bdate_range(start=today, end=portfolio_maturity)
        n_days     = len(date_range)

        # ── Shock offsets → nearest business day ─────────────────────────────
        shock_day_offsets = sorted(
            shock_in_days + i * shock_spacing_days
            for i in range(n_shocks)
            if (shock_in_days + i * shock_spacing_days) / 360 <= T_total
        )

        shock_dates = set()
        for d in shock_day_offsets:
            target      = today + pd.Timedelta(days=d)
            nearest_idx = np.argmin(np.abs(date_range - target))
            shock_dates.add(date_range[nearest_idx])

        T_first_shock = shock_day_offsets[0]  / 360 if shock_day_offsets else T_total
        T_last_shock  = shock_day_offsets[-1] / 360 if shock_day_offsets else 0.0
        T_post_shock  = max(T_total - T_last_shock, 0.0)

        rng_map = self._get_isin_rng_map(isins, scenario_seed=scenario_seed)
       

        Z_raw = np.column_stack([
                rng_map[isin].standard_normal(n_days)
                for isin in isins
            ])  # (n_days, n_assets)

            # apply correlation
        Z_corr = Z_raw @ L.T

        # ── Simulate all assets jointly day-by-day ───────────────────────────
        prices              = spots.copy()                        # (n_assets,)
        price_paths         = np.zeros((n_days, n_assets))
        product_final_prices = np.full(n_assets, np.nan)
        prev_date           = today

        for t_idx, bday in enumerate(date_range):
            dt      = (bday - prev_date).days / 360 if t_idx > 0 else 0.0
            t_years = (bday - today).days / 360

            # Drift phase (beta-scaled, vectorised over assets)
            if t_years <= T_first_shock:
                drift = pre_shock_drift  * betas   # (n_assets,)
            elif t_years > T_last_shock:
                drift = post_shock_drift * betas
            #else:
             #   drift = np.zeros(n_assets)

            # GBM step — all assets in one vectorised expression
            if dt > 0:
                prices *= np.exp(
                    (drift - 0.5 * vols ** 2) * dt
                    + vols * np.sqrt(dt) * Z_corr[t_idx]   # correlated draw
                )

            # Discrete shock applied uniformly across all underlyings
            if bday in shock_dates:
                prices *= (1 + market_shock / 100)

            # Capture each product's terminal price at its own maturity
            if bday >= maturity:
                mask = np.isnan(product_final_prices)
                product_final_prices = np.where(mask, prices, product_final_prices)

            price_paths[t_idx] = prices
            prev_date = bday

        # Safety: if maturity is beyond the grid, use last price
        #still_nan = np.isnan(product_final_prices)
       # product_final_prices = np.where(still_nan, price_paths[-1], product_final_prices)

        # ── Outputs ──────────────────────────────────────────────────────────
        shocks      = [(product_final_prices[i] / spots[i] - 1) * 100 for i in range(n_assets)]
        final_spots = [round(float(p), 4) for p in product_final_prices]

        paths = {
            isin: pd.DataFrame({"date": date_range, "price": price_paths[:, i]})
            for i, isin in enumerate(isins)
        }

        path_summary = {
            "maturity_date"       : row["maturity_date"],
            "T_remaining_years"   : round(T_remaining, 3),
            "T_first_shock_years" : round(T_first_shock, 3),
            "T_post_shock_years"  : round(T_post_shock, 3),
            "effective_n_shocks"  : len(shock_day_offsets),
            "market_shock_pct"    : market_shock,
            "pre_shock_drift_pa"  : pre_shock_drift,
            "post_shock_drift_pa" : post_shock_drift,
            "correlation_used"    : corr_matrix is not None,
        }

        return shocks, final_spots, T_remaining, path_summary, paths

    def run_product_path_scenario(self, row, scenario, corr_df=None, scenario_seed=0):

        corr_matrix = self.get_corr_subset(row, corr_df)

        shocks, final_spots, T_remaining, path_summary, paths = self.build_shock_paths(
            row, scenario, corr_matrix=corr_matrix, scenario_seed=scenario_seed
        )

        rc = ReverseConvertible(row, final_levels=shocks)
        s = rc.summary()

        idx = row["underlyings"].index(s["worst_underlying"])
        strike = row["strike"][idx]
        notional = row["notional"]

        total_shares = notional / strike
        price = final_spots[idx]

        is_physical = price < strike

        if is_physical:
            delivered_shares     = int(total_shares)
            fractional_shares    = total_shares - delivered_shares
            fractional_cash      = fractional_shares * price
            delivered_underlying = s["worst_underlying"]
            cash_redemption      = fractional_cash
            settlement_type      = "physical"
        else:
            delivered_shares     = 0
            fractional_shares    = 0
            fractional_cash      = 0
            delivered_underlying = None
            cash_redemption      = s["total_payoff"]
            settlement_type      = "cash"

        return {
            # Product identifiers
            "product_id"          : row["product_id"],
            "product_type"        : row["product_type"],
            "currency"            : row["currency"],
            "position_units"      : row["position_units"],
            "notional"            : row["notional"],
            "maturity_date"       : row["maturity_date"],

            # Path
            "T_remaining_years"   : round(T_remaining, 2),
            "final_spots"         : final_spots,

            # Scenario inputs
            "market_shock"        : scenario.get("market_shock", 0),
            "n_shocks"            : scenario.get("n_shocks", 1),
            "shock_in_days"       : scenario.get("shock_in_days", 0),
            "pre_shock_drift_pa"  : scenario.get("pre_shock_drift_pa", 0.0),
            "post_shock_drift_pa" : scenario.get("post_shock_drift_pa", 0.0),

            # Payoff
            "total_cost"          : s["total_cost"],
            "payoff_per_unit"     : s["payoff_per_unit"],
            "total_payoff"        : s["total_payoff"],
            "pnl"                 : s["pnl"],
            "return_pct"          : s["return_pct"],
            "barrier_breached"    : s["barrier_breached"],
            "worst_underlying"    : s["worst_underlying"],

            # Settlement
            "settlement_type"     : settlement_type,
            "delivered_underlying": delivered_underlying,
            "delivered_shares"    : delivered_shares,
            "strike"              : strike,
            "price"               : price,
            "fractional_shares"   : fractional_shares,
            "fractional_cash"     : fractional_cash,
            "cash_redemption"     : cash_redemption,

            # Full path audit
            "path_summary"        : path_summary,
            "paths"               : paths,
        }

    def run_path_scenario(self, scenario, corr_df=None):
        """
        Runs path-based scenario across full portfolio
        (equivalent to run(), but using correlated path logic).

        Parameters
        ----------
        corr_df : pd.DataFrame, optional
            Full correlation matrix with ISINs as both index and columns.
            Defaults to identity (independent assets) per product.

        A scenario_seed is derived from the scenario parameters so that
        different scenarios produce visually distinct path shapes, while
        all products within one run share the same seed (portfolio consistency).
        """
        # Derive a repeatable seed from the scenario so each unique scenario
        # produces different random draws while remaining deterministic.
        scenario_seed = int(hashlib.md5(str(sorted(scenario.items())).encode()).hexdigest(), 16) % (2 ** 31)

        results   = []
        all_paths = {}

        for _, row in self.portfolio.iterrows():
            res = self.run_product_path_scenario(row, scenario, corr_df=corr_df,
                                                  scenario_seed=scenario_seed)
            for isin, path_df in res.pop("paths", {}).items():
                all_paths[isin] = path_df
            results.append(res)

        product_df = pd.DataFrame(results)

        # =========================
        # Cash positions
        # =========================
        cash_positions = (
            product_df.groupby("currency", as_index=False)
            .agg(
                total_cash=("cash_redemption", "sum")
            )
        )

        # =========================
        # Delivered stocks
        # =========================
        delivered_df = product_df[
            product_df["delivered_underlying"].notna()
        ].copy()

        if not delivered_df.empty:
            delivered_stocks = (
                delivered_df.groupby(["currency", "delivered_underlying"], as_index=False)
                .agg(
                    total_shares=("delivered_shares", "sum"),
                    total_fractional_cash=("fractional_cash", "sum"),
                    strike_x_shares=("strike", lambda x: (x * delivered_df.loc[x.index, "delivered_shares"]).sum()),
                    price_x_shares=("price", lambda x: (x * delivered_df.loc[x.index, "delivered_shares"]).sum()),
                )
            )

            delivered_stocks["strike"] = delivered_stocks["strike_x_shares"] / delivered_stocks["total_shares"]
            delivered_stocks["price"]  = delivered_stocks["price_x_shares"]  / delivered_stocks["total_shares"]

            delivered_stocks["market_value"]          = delivered_stocks["total_shares"] * delivered_stocks["price"]
            delivered_stocks["cost"]                  = delivered_stocks["total_shares"] * delivered_stocks["strike"]
            delivered_stocks["total_value_incl_cash"] = delivered_stocks["market_value"] + delivered_stocks["total_fractional_cash"]
            delivered_stocks["pnl"]                   = delivered_stocks["total_value_incl_cash"] - delivered_stocks["cost"]
            delivered_stocks["return_pct"]            = delivered_stocks["pnl"] / delivered_stocks["cost"]

            delivered_stocks = delivered_stocks[[
                "delivered_underlying", "total_shares", "strike", "price", "currency",
                "market_value", "total_fractional_cash", "total_value_incl_cash",
                "cost", "pnl", "return_pct"
            ]]
        else:
            delivered_stocks = pd.DataFrame()

        # =========================
        # Portfolio aggregation
        # =========================
        pf_scenario_per_ccy = (
            product_df.groupby("currency", as_index=False)
            .agg(
                n_products=("product_id", "count"),
                total_cost=("total_cost", "sum"),
                total_payoff=("total_payoff", "sum"),
                total_pnl=("pnl", "sum"),
                underlyings=("worst_underlying", lambda x: sorted(set(x)))
            )
        )

        pf_scenario_per_ccy["portfolio_return_pct"] = (
            pf_scenario_per_ccy["total_pnl"] /
            pf_scenario_per_ccy["total_cost"]
        )

        return {
            "product_df"        : product_df,
            "pf_scenario_per_ccy": pf_scenario_per_ccy,
            "cash_positions"    : cash_positions,
            "delivered_stocks"  : delivered_stocks,
            "paths"             : all_paths,
        }
    
    
    def get_corr_subset(self, row, corr_df):
        """
        Extract product-specific correlation submatrix in the exact order of
        row["underlying_isins"].
    
        Parameters
        ----------
        row : pd.Series
            Portfolio row containing "underlying_isins"
        corr_df : pd.DataFrame
            Full correlation matrix with ISINs as both index and columns
    
        Returns
        -------
        np.ndarray
            Correlation matrix subset with shape (n_assets, n_assets)
        """
        isins = list(row["underlying_isins"])
    
        if corr_df is None:
            return np.eye(len(isins))
    
        missing = [isin for isin in isins if isin not in corr_df.index or isin not in corr_df.columns]
        if missing:
            raise KeyError(f"Missing ISIN(s) in correlation matrix: {missing}")
    
        corr_subset = corr_df.loc[isins, isins]
    
        return corr_subset.to_numpy(dtype=float)
    
    def _get_isin_rng_map(self, isins, scenario_seed=0):
        """
        Deterministic RNG per ISIN — same underlying always gets the same path
        within one portfolio run (portfolio consistency).

        scenario_seed : int
            XOR'd into every ISIN seed so different scenarios produce different
            path shapes, not just a scaled version of the same path.
            Pass the same value for every product in one run_path_scenario call.
        """
        return {
            isin: np.random.default_rng(
                (int(hashlib.md5(str(isin).encode()).hexdigest(), 16) % (2 ** 31))
                ^ scenario_seed
            )
            for isin in isins
        }
