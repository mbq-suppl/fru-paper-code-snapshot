use xrf::{Forest, ImportanceAggregator, RfRng, Walk};

use crate::attribute::DfPivot;
use crate::{fill_double_vec, fill_int_vec, fill_raw_vec};

use super::attribute::DfAttribute;

use super::classification::DataFrame as DataFrameClassification;

use super::regression::DataFrame as DataFrameRegression;

pub enum PackedForest {
    Classification(Forest<DataFrameClassification>),
    Regression(Forest<DataFrameRegression>),
}

impl PackedForest {
    pub fn into_raw(self) -> *mut PackedForest {
        Box::into_raw(Box::new(self))
    }
    pub fn from_raw(ptr: *mut PackedForest) -> Self {
        *unsafe { Box::from_raw(ptr) }
    }

    pub fn parameters(&self) -> (usize, usize, bool, bool, bool) {
        match self {
            Self::Classification(x) => {
                (x.trees(), 1, x.has_trees(), x.has_oob(), x.has_importance())
            }
            Self::Regression(x) => (x.trees(), 0, x.has_trees(), x.has_oob(), x.has_importance()),
        }
    }
    pub fn importance(&self, imp: *mut f64) {
        let f = |(e, v): (u32, f64)| {
            unsafe { *imp.add(e as usize) = v };
        };
        match self {
            Self::Classification(x) => x.importance().for_each(f),
            Self::Regression(x) => x.importance().for_each(f),
        }
    }
    pub fn importance_normalised(&self, imp: *mut f64) {
        let f = |(e, v): (u32, f64)| {
            unsafe { *imp.add(e as usize) = v };
        };
        match self {
            Self::Classification(x) => x.importance_normalised().for_each(f),
            Self::Regression(x) => x.importance_normalised().for_each(f),
        }
    }
    pub fn serialise_importance_raw(&self, ans: *mut u8, idx: i32) {
        use super::fill_raw_vec;
        match self {
            Self::Classification(x) => {
                let records = x.raw_importance().count();
                let size = records * (4 + 24);
                let mut p = unsafe { fill_raw_vec(ans, idx, size as i32) };
                let mut acts = 0_usize;
                for (val, agg) in x.raw_importance() {
                    val.to_le_bytes()
                        .iter()
                        .chain(agg.into_raw().iter())
                        .for_each(|&x| {
                            acts += 1;
                            unsafe {
                                *p = x;
                                p = p.add(1);
                            }
                        });
                }
            }
            Self::Regression(x) => {
                let records = x.raw_importance().count();
                let size = records * (4 + 24);
                let mut p = unsafe { fill_raw_vec(ans, idx, size as i32) };
                let mut acts = 0_usize;
                for (val, agg) in x.raw_importance() {
                    val.to_le_bytes()
                        .iter()
                        .chain(agg.into_raw().iter())
                        .for_each(|&x| {
                            acts += 1;
                            unsafe {
                                *p = x;
                                p = p.add(1);
                            }
                        });
                }
            }
        }
    }
    pub fn deserialise_importance_raw(&mut self, from: *const u8, len: usize) {
        let records = len / (4 + 24);
        let mut fidb = [0u8; 4];
        let mut aggb = [0u8; 24];
        let ii = (0..records).map(|e| {
            let p = unsafe { from.add(e * (4 + 24)) };
            fidb.iter_mut()
                .enumerate()
                .for_each(|(e, t)| *t = unsafe { *p.add(e) });
            let fid = u32::from_le_bytes(fidb);

            let p = unsafe { from.add(e * (4 + 24) + 4) };
            aggb.iter_mut()
                .enumerate()
                .for_each(|(e, t)| *t = unsafe { *p.add(e) });
            let agg = ImportanceAggregator::from_raw(&aggb);
            (fid, agg)
        });
        match self {
            Self::Classification(x) => x.replace_importance(ii),
            Self::Regression(x) => x.replace_importance(ii),
        };
    }
    pub fn deserialise_oob_raw(&mut self, from: *const u8, len: usize) {
        match self {
            Self::Classification(x) => {
                use super::classification::Votes;
                let ncat = u32::from_le_bytes(unsafe {
                    [*from, *from.add(1), *from.add(2), *from.add(3)]
                });
                let p = unsafe { from.add(4) };
                let records = (len - 4) / (ncat as usize) / 4;
                let ii = (0..records).map(|e| {
                    let p = unsafe { p.add(e * (ncat as usize) * 4) };
                    unsafe { Votes::from_raw(p, ncat) }
                });
                x.replace_oob(ii);
            }
            Self::Regression(x) => {
                use super::regression::Votes;
                let records = (len) / 16;
                let ii = (0..records).map(|e| unsafe { Votes::from_raw(from.add(e * 8)) });
                x.replace_oob(ii);
            }
        };
    }
    pub fn serialise_oob_raw(&self, ans: *mut u8, idx: i32) {
        match self {
            Self::Classification(x) => {
                let n = x.oob().count();
                let mut ii = x.oob().peekable();
                let ncat: u32 = if let Some(v) = ii.peek() {
                    v.1.ncat() as u32
                } else {
                    return;
                };
                let len = n * (ncat as usize) * 4 + 4;
                let p: *mut u8 = unsafe { fill_raw_vec(ans, idx, len as i32) };
                ncat.to_le_bytes()
                    .iter()
                    .enumerate()
                    .for_each(|(e, &v)| unsafe { *p.add(e) = v });
                let p = unsafe { p.add(4) };

                ii.for_each(|(e, x)| unsafe { x.into_raw(p.add(e * 4 * (ncat as usize))) });
            }
            Self::Regression(x) => {
                let n = x.oob().count();
                let ii = x.oob();
                let len = n * 16;
                let p: *mut u8 = unsafe { fill_raw_vec(ans, idx, len as i32) };
                ii.for_each(|(e, x)| unsafe { x.into_raw(p.add(e * 16)) });
            }
        };
    }
    pub fn oob(&self, oob_cat: *mut i32, oob_num: *mut f64) {
        // OOB uses this hand-picked seed for consistency
        let mut rng = RfRng::from_seed(1, 1);
        match self {
            Self::Classification(x) => x.oob().for_each(|(e, v)| unsafe {
                // Mark objects lacking votes with NAs
                let v = v.collapse_empty_na(&mut rng);
                *oob_cat.add(e) = v.checked_add(1).unwrap_or(i32::MIN as u32) as i32
            }),
            Self::Regression(x) => x.oob().for_each(|(e, v)| unsafe {
                let v = v.collapse();
                *oob_num.add(e) = v
            }),
        }
    }
    pub fn oob_votes(&self, oob_cat: *mut i32, n: usize) {
        // OOB uses this hand-picked seed for consistency
        match self {
            Self::Classification(x) => x.oob().for_each(|(e, v)| {
                v.0.iter().enumerate().for_each(|(ee, vv)| unsafe {
                    *oob_cat.add(ee * n + e) = *vv as i32;
                })
            }),
            Self::Regression(_) => unreachable!("Votes for regression"),
        }
    }
    pub fn predict(
        &self,
        att: Vec<DfAttribute>,
        ncat: u32,
        n: usize,
        m: usize,
        seed: u64,
        threads: usize,
        pred_cat: *mut i32,
        pred_num: *mut f64,
    ) {
        match self {
            Self::Classification(x) => {
                let mut rng = RfRng::from_seed(seed, 1);
                let df = DataFrameClassification::new(att, Vec::new(), ncat, m, n);
                let pred = if threads == 1 {
                    x.predict(&df)
                } else {
                    x.predict_parallel(&df, threads)
                };
                pred.predictions().for_each(|(e, v)| {
                    // Always produce some prediction
                    let v = v.collapse_empty_random(&mut rng);
                    unsafe { *pred_cat.add(e) = v.checked_add(1).unwrap_or(i32::MIN as u32) as i32 }
                });
            }
            Self::Regression(x) => {
                let df = DataFrameRegression::new(att, Vec::new(), m, n);
                let pred = if threads == 1 {
                    x.predict(&df)
                } else {
                    x.predict_parallel(&df, threads)
                };
                pred.predictions().for_each(|(e, v)| unsafe {
                    let v = v.collapse();
                    *pred_num.add(e) = v
                });
            }
        }
    }
    pub fn predict_votes(
        &self,
        att: Vec<DfAttribute>,
        ncat: u32,
        n: usize,
        m: usize,
        threads: usize,
        pred_cat: *mut i32,
    ) {
        match self {
            Self::Classification(x) => {
                let df = DataFrameClassification::new(att, Vec::new(), ncat, m, n);
                let pred = if threads == 1 {
                    x.predict(&df)
                } else {
                    x.predict_parallel(&df, threads)
                };

                pred.predictions().for_each(|(e, v)| {
                    v.0.iter().enumerate().for_each(|(ee, vv)| unsafe {
                        *pred_cat.add(ee * n + e) = *vv as i32;
                    })
                });
            }
            Self::Regression(_) => unimplemented!("Vote prediction for regression"),
        }
    }
    pub fn walk_steps(&self) -> usize {
        match self {
            Self::Classification(x) => x.walk().count(),
            Self::Regression(x) => x.walk().count(),
        }
    }
    pub fn flatten(&self, ans: *mut u8, idxs: (i32, i32, i32)) {
        let len = self.walk_steps();
        let flags: *mut i32 = unsafe { fill_int_vec(ans, idxs.0, len as i32) };
        let features: *mut i32 = unsafe { fill_int_vec(ans, idxs.1, len as i32) };
        let values: *mut f64 = unsafe { fill_double_vec(ans, idxs.2, len as i32) };
        let mut e: usize = 0;
        match self {
            Self::Classification(x) => {
                for step in x.walk() {
                    match step {
                        Walk::VisitLeaf(l) => {
                            unsafe {
                                *flags.add(e) = 1;
                                *features.add(e) = i32::MIN; // NA_integer
                                *values.add(e) = l as f64;
                            }
                        }
                        Walk::VisitBranch(fid, piv) => {
                            let (flag, value) = piv.encode();
                            unsafe {
                                *flags.add(e) = flag;
                                *features.add(e) = fid as i32;
                                *values.add(e) = value;
                            }
                        }
                    }
                    e += 1;
                }
            }
            Self::Regression(x) => {
                for step in x.walk() {
                    match step {
                        Walk::VisitLeaf(l) => {
                            unsafe {
                                *flags.add(e) = 2;
                                *features.add(e) = i32::MIN; // NA_integer
                                *values.add(e) = l;
                            }
                        }
                        Walk::VisitBranch(fid, piv) => {
                            let (flag, value) = piv.encode();
                            unsafe {
                                *flags.add(e) = flag;
                                *features.add(e) = fid as i32;
                                *values.add(e) = value;
                            }
                        }
                    }
                    e += 1;
                }
            }
        }
    }
    pub fn from_flattened(
        len: usize,
        ftype: i32,
        flags: *const i32,
        features: *const i32,
        values: *const f64,
    ) -> Self {
        if len == 0 {
            Self::from_nothing(0, ftype);
        }
        let iter = (0..len).map(|e| {
            let flag = unsafe { *flags.add(e) };
            let feature = unsafe { *features.add(e) as u32 };
            let value = unsafe { *values.add(e) };
            (flag, feature, value)
        });
        match ftype {
            1 => {
                //Classification vote
                Forest::from_walk(iter.map(|(flag, feature, value)| match flag {
                    1 => Walk::VisitLeaf(value as u32),
                    3..=6 => Walk::VisitBranch(feature, DfPivot::decode((flag, value)).unwrap()),
                    _ => panic!("Deserialisation failure"),
                }))
                .map(PackedForest::Classification)
                .unwrap()
            }
            2 => {
                //Regression walk
                Forest::from_walk(iter.map(|(flag, feature, value)| match flag {
                    2 => Walk::VisitLeaf(value),
                    3..=6 => Walk::VisitBranch(feature, DfPivot::decode((flag, value)).unwrap()),
                    _ => panic!("Deserialisation failure"),
                }))
                .map(PackedForest::Regression)
                .unwrap()
            }
            _ => panic!("Deserialisation failure"),
        }
    }
    pub fn from_nothing(ntree: i32, ftype: i32) -> Self {
        if ftype == 0 {
            PackedForest::Regression(Forest::new_with_num_trees(ntree as usize))
        } else {
            PackedForest::Classification(Forest::new_with_num_trees(ntree as usize))
        }
    }
}

pub trait Packable {
    fn pack(self) -> PackedForest;
}

impl Packable for Forest<DataFrameClassification> {
    fn pack(self) -> PackedForest {
        PackedForest::Classification(self)
    }
}
impl Packable for Forest<DataFrameRegression> {
    fn pack(self) -> PackedForest {
        PackedForest::Regression(self)
    }
}
