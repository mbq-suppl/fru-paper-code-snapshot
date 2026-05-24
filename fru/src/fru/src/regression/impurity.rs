use super::DecisionSlice;
use super::VarAggregator;
use crate::attribute::DfPivot;
use crate::tools::midpoint;
use xrf::{Mask, RfRng};

pub fn scan_bin(x: *const u32, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut left = VarAggregator::new();
    let mut right = VarAggregator::new();
    mask.iter()
        .map(|&e| unsafe { *x.add(e) } != 0u32)
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            if x {
                left.ingest(y);
            } else {
                right.ingest(y);
            }
        });
    let score: f64 = -(left.var_n() + right.var_n());
    //Big variance is bad, we measure var(Y)-var*(left)-var*(right), so - (var*left+var*right)
    Some((DfPivot::Logical, score))
}

pub fn scan_factor(
    x: *const i32,
    xc: u32,
    ys: &DecisionSlice,
    mask: &Mask,
    _rng: &mut RfRng,
) -> Option<(DfPivot, f64)> {
    if xc > crate::MAX_FACTOR_LEVELS {
        //When there is too many combinations, just treat it as ordered
        return scan_i32(x, ys, mask);
    }
    if xc < 2 {
        return None;
    }
    let mut va: Vec<VarAggregator> = std::iter::repeat_with(VarAggregator::new)
        .take(xc as usize)
        .collect();
    mask.iter()
        .map(|&e| unsafe { *x.add(e) - 1 })
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            va.get_mut(x as usize)
                .expect("Invalid value or NA in a factor feature")
                .ingest(y)
        });
    let sub_max: u64 = (1 << (xc - 1)) - 1;

    (0..sub_max)
        .map(|bitmask_id| bitmask_id + (1 << (xc - 1)))
        .fold(None, |acc: Option<(u64, f64)>, bitmask| {
            let left = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) != 0)
                .fold(VarAggregator::new(), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let right = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) == 0)
                .fold(VarAggregator::new(), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let score: f64 = -(left.var_n() + right.var_n());
            if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                return Some((bitmask, score));
            }
            acc
        })
        .map(|(bitmask, score)| (DfPivot::Subset(bitmask), score))
}

pub fn scan_f64(x: *const f64, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(f64, f64)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| {
            let x = unsafe { *x.add(xe) };
            (x, y)
        })
        .collect();
    bound.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    scan_ordered(bound.into_iter(), ys, |a, b| (a + b) / 2.)
        .map(|(thresh, score)| (DfPivot::Real(thresh), score))
}

pub fn scan_i32(x: *const i32, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(i32, f64)> = mask
        .iter()
        .zip(ys.values.iter())
        .map(|(&xe, &y)| {
            let x = unsafe { *x.add(xe) };
            (x, y)
        })
        .collect();
    bound.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    scan_ordered(bound.into_iter(), ys, midpoint)
        .map(|(thresh, score)| (DfPivot::Integer(thresh), score))
}

#[inline(always)]
fn scan_ordered<T: Copy + PartialOrd + Default, I: Iterator<Item = (T, f64)>, M: Fn(T, T) -> T>(
    iter: I,
    ys: &DecisionSlice,
    midpoint: M,
) -> Option<(T, f64)> {
    let mut left = VarAggregator::new();
    let mut right = ys.summary.clone();
    iter.scan((T::default(), 0.), |last, cur| {
        let ans = (last.0, cur.0, last.1);
        last.0 = cur.0;
        last.1 = cur.1;
        Some(ans)
    })
    .skip(1)
    .fold(None, |acc: Option<(T, f64)>, (x, next_x, y)| {
        left.ingest(y);
        right.degest(y);
        if x.partial_cmp(&next_x).unwrap().is_ne() {
            let score: f64 = -(left.var_n() + right.var_n());
            if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                return Some((midpoint(x, next_x), score));
            }
        }
        acc
    })
}
