use super::attribute::FYSampler;
use crate::attribute::{DfAttribute, DfPivot, SplittingIterator};
use xrf::{Mask, RfInput, RfRng};

mod da;
mod impurity;
mod votes;
pub use votes::Votes;

pub struct DataFrame {
    features: Vec<DfAttribute>,
    decision: Vec<f64>,
    m: usize,
    n: usize,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = f64;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::DaAggregator;
    type FeatureSampler = FYSampler<Self>;
    fn observation_count(&self) -> usize {
        self.n
    }
    fn feature_count(&self) -> usize {
        self.m
    }
    fn feature_sampler(&self) -> Self::FeatureSampler {
        super::attribute::FYSampler::new(self)
    }
    fn decision_slice(&self, mask: &Mask) -> Self::DecisionSlice {
        DecisionSlice::new(mask, &self.decision)
    }
    fn new_split(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        y: &Self::DecisionSlice,
        rng: &mut RfRng,
    ) -> Option<(Self::Pivot, f64)> {
        use DfAttribute::*;
        let feature = &self.features[using as usize];
        match *feature {
            Numeric(x) => impurity::scan_f64(x, y, on),
            Integer(x) => impurity::scan_i32(x, y, on),
            Logical(x) => impurity::scan_bin(x, y, on),
            Factor(xc, x) => impurity::scan_factor(x, xc, y, on, rng),
        }
    }
    fn split_iter(
        &self,
        on: &Mask,
        using: Self::FeatureId,
        by: &Self::Pivot,
    ) -> impl Iterator<Item = bool> {
        let feature = &self.features[using as usize];
        SplittingIterator::new(feature, by, on.iter())
    }
}

pub struct DecisionSlice {
    values: Vec<f64>,
    summary: VarAggregator,
}

impl DecisionSlice {
    fn new(mask: &Mask, values: &[f64]) -> Self {
        let mut summary = VarAggregator::new();
        let values = mask
            .iter()
            .map(|&e| values[e])
            .inspect(|&e| summary.ingest(e))
            .collect();
        DecisionSlice { values, summary }
    }
}

impl xrf::DecisionSlice<f64> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.values.len() < 5
    }
    fn condense(&self, _rng: &mut RfRng) -> f64 {
        self.summary.ave()
    }
}

//Boring stuff

impl DataFrame {
    //TOOD: Better order of arguments, maybe?
    pub fn new(features: Vec<DfAttribute>, decision: Vec<f64>, m: usize, n: usize) -> Self {
        Self {
            features,
            decision,
            m,
            n,
        }
    }
}

#[derive(Clone)]
struct VarAggregator {
    sum: f64,
    sum_sq: f64,
    n: usize,
}

impl VarAggregator {
    fn new() -> Self {
        Self {
            sum: 0.,
            sum_sq: 0.,
            n: 0,
        }
    }
    fn ingest(&mut self, x: f64) {
        self.sum += x;
        self.sum_sq += x * x;
        self.n += 1;
    }
    fn degest(&mut self, x: f64) {
        self.sum -= x;
        self.sum_sq -= x * x;
        self.n -= 1;
    }
    fn ave(&self) -> f64 {
        self.sum / (self.n as f64)
    }
    fn var_n(&self) -> f64 {
        self.sum_sq - self.sum * self.sum / (self.n as f64)
    }
    fn merge(&mut self, other: &Self) {
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.n += other.n;
    }
}
