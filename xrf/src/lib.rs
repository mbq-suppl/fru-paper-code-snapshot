mod mask;
pub use mask::{Mask, MaskCache};

mod rng;
pub use rng::RfRng;

mod fair_best;
pub use fair_best::FairBest;

mod rfinput;
pub use rfinput::{
    AccuracyDecreaseAggregator, DecisionSlice, FeatureSampler, RfInput, VoteAggregator,
};

mod forest;
pub use forest::{Forest, ImportanceAggregator, Prediction};

mod walk;
pub use walk::Walk;

mod error;
pub use error::XrfError;

#[cfg(test)]
mod mockups;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iris() {
        let tt = 2;
        let iris = mockups::generate_iris();
        let fo = forest::Forest::new_parallel(&iris, 500, 2, true, true, true, 1, tt);

        let mut imp = vec![0., 0., 0., 0.];
        fo.importance().for_each(|(e, v)| imp[e] = v);
        // eprintln!("Imp {imp:?}");
        assert!(imp[0] < imp[2]);
        assert!(imp[1] < imp[3]);

        let mut imp_n = vec![0., 0., 0., 0.];
        fo.importance_normalised().for_each(|(e, v)| imp_n[e] = v);
        // eprintln!("Imp normalised {imp_n:?}");
        assert!(imp_n[0] < imp_n[2]);
        assert!(imp_n[1] < imp_n[3]);

        let mut rng = RfRng::from_seed(1, 1);
        let oob_acc = fo
            .oob()
            .map(|(e, v)| (iris.y()[e], v.collapse(&mut rng)))
            .map(|(p, t)| if p == t { 1 } else { 0 })
            .sum::<usize>() as f64
            / iris.observation_count() as f64;
        assert!(oob_acc > 0.90);
        // eprintln!("OOB accuracy is {oob_acc}");

        let pred_acc = fo
            .predict_parallel(&iris, tt)
            .predictions()
            .map(|(e, v)| (iris.y()[e], v.collapse(&mut rng)))
            .map(|(p, t)| if p == t { 1 } else { 0 })
            .sum::<usize>() as f64
            / iris.observation_count() as f64;
        assert!(pred_acc > 0.95);
        // eprintln!("Prediction accuracy is {pred_acc}");
    }
}
