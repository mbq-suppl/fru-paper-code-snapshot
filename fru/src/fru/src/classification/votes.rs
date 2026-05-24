use super::DataFrame;
use xrf::VoteAggregator;
use xrf::{FairBest, RfRng};

#[derive(Clone)]
pub struct Votes(pub Vec<u32>); //TODO: Fix impurity to make it private

impl Votes {
    pub fn is_pure(&self) -> bool {
        self.0.iter().filter(|&&x| x > 0).count() <= 1
    }
    pub fn new(ncat: u32) -> Self {
        Self(std::iter::repeat_n(0, ncat as usize).collect())
    }
    pub fn ncat(&self) -> usize {
        self.0.len()
    }
    pub unsafe fn from_raw(vals: *const u8, ncat: u32) -> Self {
        Self(
            (0..(ncat as usize))
                .map(|e| {
                    u32::from_le_bytes(unsafe {
                        [
                            *vals.add(e * 4),
                            *vals.add(e * 4 + 1),
                            *vals.add(e * 4 + 2),
                            *vals.add(e * 4 + 3),
                        ]
                    })
                })
                .collect(),
        )
    }
    pub unsafe fn into_raw(&self, vals: *mut u8) {
        self.0
            .iter()
            .flat_map(|x| x.to_le_bytes().to_vec().into_iter())
            .enumerate()
            .for_each(|(e, v)| unsafe { *vals.add(e) = v })
    }
    pub fn collapse_empty_na(&self, rng: &mut RfRng) -> u32 {
        self.0
            .iter()
            .enumerate()
            .fold(FairBest::new(), |mut best, (cls, count)| {
                best.ingest(count, cls, rng);
                best
            })
            .consume()
            .filter(|(score, _class)| **score > 0)
            .map(|(_score, class)| class as u32)
            .unwrap_or(u32::MAX)
    }
    pub fn collapse_empty_random(&self, rng: &mut RfRng) -> u32 {
        self.0
            .iter()
            .enumerate()
            .fold(FairBest::new(), |mut best, (cls, count)| {
                best.ingest(count, cls, rng);
                best
            })
            .consume()
            .map(|(_score, class)| class as u32)
            .unwrap()
    }
}

impl VoteAggregator<DataFrame> for Votes {
    fn new(input: &DataFrame) -> Self {
        Votes::new(input.ncat)
    }
    fn ingest_vote(&mut self, v: u32) {
        self.0[v as usize] += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.0
            .iter_mut()
            .zip(other.0.iter())
            .for_each(|(t, s)| *t += s);
    }
}
