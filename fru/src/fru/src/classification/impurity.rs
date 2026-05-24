use super::{DecisionSlice, Votes};
use crate::attribute::DfPivot;
use crate::tools::midpoint;
use xrf::{Mask, RfRng, VoteAggregator};

pub fn scan_bin(x: *const u32, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut left = Votes::new(ys.ncat);
    let mut xt = 0_usize;
    let n = mask.len();
    mask.iter()
        .map(|&e| unsafe { *x.add(e) } != 0u32)
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            if x {
                left.ingest_vote(y);
                xt += 1;
            }
        });
    let score: f64 = ys
        .summary
        .0
        .iter()
        .zip(left.0.iter())
        .map(|(total, &left)| {
            let for_false = (n - xt) as f64;
            let for_true = xt as f64;
            let n = n as f64;
            let right = (total - left) as f64;
            let left = left as f64;
            (left / for_true) * (left / n) + (right / for_false) * (right / n)
        })
        .sum();
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
    let n = mask.len();
    let mut va: Vec<Votes> = std::iter::repeat_with(|| Votes::new(ys.ncat))
        .take(xc as usize)
        .collect();
    mask.iter()
        .map(|&e| unsafe { *x.add(e) - 1 })
        .zip(ys.values.iter())
        .for_each(|(x, &y)| {
            va.get_mut(x as usize)
                .expect("Invalid value or NA in a factor feature")
                .ingest_vote(y)
        });
    let sub_max: u64 = (1 << (xc - 1)) - 1;

    (0..sub_max)
        .map(|bitmask_id| bitmask_id + (1 << (xc - 1)))
        .fold(None, |acc: Option<(u64, f64)>, bitmask| {
            let left = va
                .iter()
                .enumerate()
                .filter(|(e, _)| bitmask & (1 << e) != 0)
                .fold(Votes::new(ys.ncat), |mut acc, (_, v)| {
                    acc.merge(v);
                    acc
                });
            let in_left: u32 = left.0.iter().sum();

            let score: f64 = ys
                .summary
                .0
                .iter()
                .zip(left.0.iter())
                .map(|(&all, &left)| {
                    let ahead = (n - in_left as usize) as f64;
                    let scanned = in_left as f64;
                    let n = n as f64;
                    let right = (all - left) as f64;
                    let left = left as f64;
                    (left / scanned) * (left / n) + (right / ahead) * (right / n)
                })
                .sum();
            if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                return Some((bitmask, score));
            }
            acc
        })
        .map(|(bitmask, score)| (DfPivot::Subset(bitmask), score))
}

pub fn scan_f64(x: *const f64, ys: &DecisionSlice, mask: &Mask) -> Option<(DfPivot, f64)> {
    let mut bound: Vec<(f64, u32)> = mask
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
    let mut bound: Vec<(i32, u32)> = mask
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
fn scan_ordered<T: Copy + PartialOrd + Default, I: Iterator<Item = (T, u32)>, M: Fn(T, T) -> T>(
    iter: I,
    ys: &DecisionSlice,
    midpoint: M,
) -> Option<(T, f64)> {
    let n = ys.values.len();
    let mut left = Votes::new(ys.ncat);
    let mut scanned = 0_usize;
    iter.scan((T::default(), 0), |last, cur| {
        let ans = (last.0, cur.0, last.1);
        last.0 = cur.0;
        last.1 = cur.1;
        Some(ans)
    })
    .skip(1)
    .fold(None, |acc: Option<(T, f64)>, (x, next_x, y)| {
        scanned += 1;
        left.ingest_vote(y);
        if x.partial_cmp(&next_x).unwrap().is_ne() {
            let score: f64 = ys
                .summary
                .0
                .iter()
                .zip(left.0.iter())
                .map(|(&all, &left)| {
                    let ahead = (n - scanned) as f64;
                    let scanned = scanned as f64;
                    let n = n as f64;
                    let right = (all - left) as f64;
                    let left = left as f64;
                    (left / scanned) * (left / n) + (right / ahead) * (right / n)
                })
                .sum();
            if score > acc.map(|x| x.1).unwrap_or(f64::NEG_INFINITY) {
                return Some((midpoint(x, next_x), score));
            }
        }
        acc
    })
}
