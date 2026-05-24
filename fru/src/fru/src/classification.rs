use super::attribute::FYSampler;
use crate::attribute::{DfAttribute, DfPivot, SplittingIterator};
use xrf::{Mask, RfInput, RfRng, VoteAggregator};

mod da;
mod impurity;
mod votes;
pub use votes::Votes;

pub struct DataFrame {
    features: Vec<DfAttribute>,
    decision: Vec<u32>,
    ncat: u32,
    m: usize,
    n: usize,
}

impl RfInput for DataFrame {
    type FeatureId = u32;
    type DecisionSlice = DecisionSlice;
    type Pivot = DfPivot;
    type Vote = u32;
    type VoteAggregator = Votes;
    type AccuracyDecreaseAggregator = da::ClsDaAggregator;
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
        DecisionSlice::new(mask, &self.decision, self.ncat)
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
    values: Vec<u32>,
    ncat: u32,
    summary: Votes,
}
impl DecisionSlice {
    fn new(mask: &Mask, values: &[u32], ncat: u32) -> Self {
        let mut summary = Votes::new(ncat);
        let values = mask
            .iter()
            .map(|&e| values[e])
            .inspect(|&x| summary.ingest_vote(x))
            .collect();
        DecisionSlice {
            values,
            ncat,
            summary,
        }
    }
}

impl xrf::DecisionSlice<u32> for DecisionSlice {
    fn is_pure(&self) -> bool {
        self.summary.is_pure()
    }
    fn condense(&self, rng: &mut RfRng) -> u32 {
        //Leaf always need a single class, so select
        // random one if there is no data
        self.summary.collapse_empty_random(rng)
    }
}

//Boring stuff

impl DataFrame {
    //TOOD: Better order of arguments, maybe?
    pub fn new(
        features: Vec<DfAttribute>,
        decision: Vec<u32>,
        ncat: u32,
        m: usize,
        n: usize,
    ) -> Self {
        Self {
            features,
            decision,
            ncat,
            m,
            n,
        }
    }
}
