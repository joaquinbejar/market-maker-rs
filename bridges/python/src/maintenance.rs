//! The maintenance state of one linear position under `margin-rs`, step by step: the reading
//! of #105's margin-stress holdout.
//!
//! [`maintenance_path`] takes, for each step, the delivered position, the mark it is valued
//! at and the account's net result, and returns the account state `margin-rs` assigns. The
//! account holds one linear perpetual position of `position` contracts of multiplier 1 at
//! `mark`, margined by the rule `linear-im/1` (initial 10 %, maintenance 5 % of the
//! notional), and its cash, collateral and equity are all `capital + net`: a linear account
//! whose only asset is the settlement currency. Each step is assessed by
//! `margin_rs::maintenance::assess` from the previous step's state, the first from
//! `Healthy`. Nothing is triggered: a state is recorded, never acted on.

use margin_rs::Decimal;
use margin_rs::initial_margin::{
    ExposurePosition, ExposureView, InitialMarginModel, MarginRequirement, ProductKind,
};
use margin_rs::maintenance::{AccountState, CollateralView, assess};
use margin_rs::rules::TieredLinearIm;
use margin_rs::{AccountId, Asset, ContractSize, InstrumentId, VenueId};

/// One step of an account, as read: its state, the maintenance margin and the shortfall.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaintenanceStep {
    /// The account state after the step.
    pub state: AccountState,
    /// The maintenance margin of the position.
    pub maintenance: Decimal,
    /// `max(0, maintenance - collateral)`.
    pub shortfall: Decimal,
}

/// A step's inputs: the delivered position in contracts, its mark and the net result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInput {
    /// Signed position, in contracts of multiplier 1.
    pub position: Decimal,
    /// Mark, in the settlement currency.
    pub mark: Decimal,
    /// Net result of the account so far.
    pub net: Decimal,
}

fn refused(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn requirement(
    rule: &TieredLinearIm,
    ids: &(AccountId, VenueId, InstrumentId, Asset),
    step: StepInput,
) -> Result<MarginRequirement, String> {
    let (account, venue, instrument, currency) = ids;
    let positions = if step.position.is_zero() {
        Vec::new()
    } else {
        vec![ExposurePosition {
            instrument: instrument.clone(),
            product: ProductKind::LinearPerpetual,
            settlement: currency.clone(),
            contracts: step.position,
            size: ContractSize::Multiplier(Decimal::ONE),
            mark: step.mark,
        }]
    };
    rule.requirement(&ExposureView {
        account: account.clone(),
        venue: venue.clone(),
        positions,
    })
    .map_err(refused)
}

/// Each step's account state under `linear-im/1`, from `capital` and the steps in order.
/// `scale` is the settlement currency's scale, to which the rule rounds its margins up.
///
/// # Errors
///
/// A message when `margin-rs` refuses an input (a scale it cannot round to, or a position
/// or mark it cannot value) or when the capital plus a net result does not fit a decimal.
pub fn maintenance_path(
    capital: Decimal,
    scale: u32,
    steps: &[StepInput],
) -> Result<Vec<MaintenanceStep>, String> {
    let currency = Asset::new("USDC").map_err(refused)?;
    let rule = TieredLinearIm::linear_v1(currency.clone(), scale).map_err(refused)?;
    let ids = (
        AccountId::new("account").map_err(refused)?,
        VenueId::new("venue").map_err(refused)?,
        InstrumentId::new("PERP-USDC").map_err(refused)?,
        currency.clone(),
    );
    let mut previous = AccountState::Healthy;
    let mut out = Vec::with_capacity(steps.len());
    for step in steps {
        let required = requirement(&rule, &ids, *step)?;
        let equity = capital
            .checked_add(step.net)
            .ok_or_else(|| format!("capital {capital} plus net {} overflows", step.net))?;
        let view = CollateralView {
            currency: currency.clone(),
            cash: equity,
            collateral: Some(equity),
            equity: Some(equity),
        };
        let decision = assess(previous, &view, &required).map_err(refused)?;
        previous = decision.state;
        out.push(MaintenanceStep {
            state: decision.state,
            maintenance: decision.maintenance,
            // A shortfall of zero is reported as 0, not as a negative zero.
            shortfall: decision
                .maintenance_shortfall
                .filter(|shortfall| !shortfall.is_zero())
                .unwrap_or(Decimal::ZERO),
        });
    }
    Ok(out)
}

/// The name a state is reported by.
#[must_use]
pub fn state_name(state: AccountState) -> &'static str {
    match state {
        AccountState::Healthy => "healthy",
        AccountState::ReduceOnly => "reduce_only",
        AccountState::Default => "default",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(text: &str) -> Decimal {
        Decimal::from_str(text).unwrap()
    }

    fn step(position: &str, mark: &str, net: &str) -> StepInput {
        StepInput {
            position: d(position),
            mark: d(mark),
            net: d(net),
        }
    }

    #[test]
    fn test_the_maintenance_is_five_percent_of_the_notional_and_flat_is_healthy() {
        let path = maintenance_path(
            d("125"),
            6,
            &[step("0", "100", "0"), step("-3", "100", "0")],
        )
        .unwrap();
        assert_eq!(path[0].state, AccountState::Healthy);
        assert_eq!(path[0].maintenance, Decimal::ZERO);
        // 0.05 * 3 * 100 = 15 against 125: healthy, with no shortfall.
        assert_eq!(path[1].maintenance, d("15"));
        assert_eq!(
            (path[1].state, path[1].shortfall),
            (AccountState::Healthy, Decimal::ZERO)
        );
        // No shortfall is 0, never a negative zero.
        assert!(!path[1].shortfall.is_sign_negative());
        let at = maintenance_path(d("125"), 2, &[step("20", "100", "-25")]).unwrap();
        assert!(at[0].shortfall.is_zero() && !at[0].shortfall.is_sign_negative());
    }

    #[test]
    fn test_below_maintenance_is_reduce_only_at_equality_healthy_and_negative_default() {
        // 20 lots at 100: maintenance 100. Equity 125 - 25 = 100 is at it: healthy; 99.99
        // is below: reduce only, short by 0.01; below zero: default.
        let path = maintenance_path(
            d("125"),
            2,
            &[
                step("20", "100", "-25"),
                step("20", "100", "-25.01"),
                step("20", "100", "-126"),
            ],
        )
        .unwrap();
        let states: Vec<_> = path.iter().map(|s| s.state).collect();
        assert_eq!(
            states,
            [
                AccountState::Healthy,
                AccountState::ReduceOnly,
                AccountState::Default
            ]
        );
        assert_eq!(path[1].shortfall, d("0.01"));
        assert_eq!(path[2].shortfall, d("101"));
    }

    #[test]
    fn test_a_default_is_carried_until_cash_and_collateral_clear_it() {
        // From default, collateral back above maintenance with positive cash clears it; a
        // position still below maintenance does not, even with equity positive.
        let path = maintenance_path(
            d("125"),
            2,
            &[
                step("1", "100", "-130"),
                step("20", "100", "-30"),
                step("1", "100", "-30"),
            ],
        )
        .unwrap();
        let states: Vec<_> = path.iter().map(|s| s.state).collect();
        assert_eq!(
            states,
            [
                AccountState::Default,
                AccountState::Default,
                AccountState::Healthy
            ]
        );
        assert_eq!(state_name(AccountState::ReduceOnly), "reduce_only");
    }

    #[test]
    fn test_an_equity_that_overflows_is_an_error_not_a_panic() {
        let error = maintenance_path(Decimal::MAX, 6, &[step("0", "1", "1")]).unwrap_err();
        assert!(error.contains("overflows"), "{error}");
        assert!(maintenance_path(Decimal::MAX, 6, &[step("0", "1", "0")]).is_ok());
    }
}
