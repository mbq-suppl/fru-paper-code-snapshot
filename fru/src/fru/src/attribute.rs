use std::marker::PhantomData;
use xrf::{FeatureSampler, RfInput, RfRng};

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum DfPivot {
    Logical,
    Real(f64),
    Integer(i32),
    Subset(u64),
}

impl DfPivot {
    pub fn encode(&self) -> (i32, f64) {
        match self {
            Self::Logical => (3, f64::NAN),
            Self::Real(x) => (4, *x),
            Self::Integer(x) => (5, *x as f64),
            Self::Subset(x) => (6, *x as f64),
        }
    }
    pub fn decode(val: (i32, f64)) -> Result<Self, ()> {
        match val.0 {
            3 => Ok(Self::Logical),
            4 => Ok(Self::Real(val.1)),
            5 => Ok(Self::Integer(val.1 as i32)),
            6 => Ok(Self::Subset(val.1 as u64)),
            _ => Err(()),
        }
    }
}

pub struct FYSampler<I> {
    mixed: Vec<u32>,
    left: usize,
    marker: PhantomData<I>,
}

impl<I: RfInput<FeatureId = u32>> FYSampler<I> {
    pub fn new(input: &I) -> Self {
        Self {
            mixed: (0..input.feature_count()).map(|x| x as u32).collect(),
            left: input.feature_count(),
            marker: PhantomData,
        }
    }
}

impl<I: RfInput<FeatureId = u32>> FeatureSampler<I> for FYSampler<I> {
    fn random_feature(&mut self, rng: &mut RfRng) -> I::FeatureId {
        let sel = rng.up_to(self.left);
        let ans = self.mixed[sel];
        self.left = self.left.checked_sub(1).unwrap();
        self.mixed.swap(sel, self.left);
        ans
    }
    fn reload(&mut self) {
        self.left = self.mixed.len();
    }
    fn reset(&mut self) {
        self.mixed = (0..self.mixed.len()).map(|x| x as u32).collect();
        self.left = self.mixed.len();
    }
}

//On pointer one can add() to move it around & * to (usafely) deref
pub enum DfAttribute {
    Logical(*const u32),
    Numeric(*const f64),
    Integer(*const i32),
    Factor(u32, *const i32),
}

unsafe impl Send for DfAttribute {}
unsafe impl Sync for DfAttribute {}

pub struct SplittingIterator<M> {
    pair: DfSplittingPair,
    mask_iter: M,
}
enum DfSplittingPair {
    Logical(*const u32),
    Numeric(*const f64, f64),
    Integer(*const i32, i32),
    Factor((*const i32, u64)),
}
impl<M> SplittingIterator<M> {
    pub fn new(x: &DfAttribute, pivot: &DfPivot, mask_iter: M) -> Self {
        use DfAttribute::*;
        Self {
            pair: match (x, pivot) {
                (&Logical(x), &DfPivot::Logical) => DfSplittingPair::Logical(x),
                (&Numeric(x), &DfPivot::Real(xt)) => DfSplittingPair::Numeric(x, xt),
                (&Integer(x), &DfPivot::Integer(xt)) => DfSplittingPair::Integer(x, xt),
                (&Factor(_ncat, x), &DfPivot::Integer(xt)) => DfSplittingPair::Integer(x, xt),
                (&Factor(_ncat, x), &DfPivot::Subset(sub)) => DfSplittingPair::Factor((x, sub)),
                _ => panic!("Mismatched value & pivot!"),
            },
            mask_iter,
        }
    }
}
impl<'a, M> Iterator for SplittingIterator<M>
where
    M: Iterator<Item = &'a usize>,
{
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if let Some(e) = self.mask_iter.next() {
            let ans = match self.pair {
                DfSplittingPair::Logical(x) => (unsafe { *x.add(*e) } != 0),
                DfSplittingPair::Numeric(x, xt) => {
                    let val = unsafe { *x.add(*e) };
                    //TODO: Check polarity
                    val > xt
                }
                DfSplittingPair::Integer(x, xt) => {
                    let val = unsafe { *x.add(*e) };
                    //TODO: Check polarity
                    val > xt
                }
                DfSplittingPair::Factor((x, split)) => {
                    let val = unsafe { *x.add(*e) - 1 };
                    (split as u32) & (1 << val as u32) != 0
                }
            };
            Some(ans)
        } else {
            None
        }
    }
}
