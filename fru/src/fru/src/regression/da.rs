use super::DataFrame;
use std::collections::HashMap;
use xrf::{AccuracyDecreaseAggregator, Mask, RfInput};

pub struct DaAggregator {
    direct: Vec<Option<f64>>,
    drops: HashMap<u32, f64>,
    n: usize,
    true_decision: Vec<f64>,
}
impl AccuracyDecreaseAggregator<DataFrame> for DaAggregator {
    fn new(input: &DataFrame, on: &Mask, n: usize) -> Self {
        Self {
            direct: vec![None; n],
            drops: HashMap::new(),
            n: on.len(),
            //TODO: Reference here
            true_decision: input.decision.clone(),
        }
    }
    fn ingest(&mut self, permutted: Option<u32>, mask: &Mask, vote: &f64) {
        if let Some(permutted) = permutted {
            let diff: f64 = mask
                .iter()
                .map(|&e| {
                    let oob_vote = self.direct.get(e).unwrap().unwrap();
                    let truth = self.true_decision[e];
                    (truth - vote) * (truth - vote) - (truth - oob_vote) * (truth - oob_vote)
                })
                .sum();
            *self.drops.entry(permutted).or_insert(0.) += diff;
        } else {
            for &e in mask.iter() {
                self.direct[e] = Some(*vote);
            }
        }
    }
    fn get_direct_vote(&self, e: usize) -> f64 {
        self.direct.get(e).unwrap().unwrap()
    }
    fn mda_iter(&self) -> impl Iterator<Item = (<DataFrame as RfInput>::FeatureId, f64)> {
        self.drops.iter().map(|(a, b)| (*a, (*b) / (self.n as f64)))
    }
}
