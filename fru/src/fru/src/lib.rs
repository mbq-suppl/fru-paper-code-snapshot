use std::panic::{RefUnwindSafe, catch_unwind};

use xrf::{Forest, RfInput};

mod attribute;
use attribute::DfAttribute;

mod classification;
use classification::DataFrame as DataFrameClassification;

mod regression;
use regression::DataFrame as DataFrameRegression;

mod packed;
use packed::{Packable, PackedForest};

pub mod tools;

pub const MAX_FACTOR_LEVELS: u32 = 5;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xrf_cls(
    n: i32,
    m: i32,
    trees: i32,
    tries: i32,
    att: *const u8,
    y: *const i32,
    yc: i32,
    threads: i32,
    forest: *mut *const PackedForest,
    todo: u8,
    seed: u64,
) {
    let decision: Vec<u32> = (0..n)
        .map(|e| unsafe { *y.add(e as usize) as u32 } - 1)
        .collect();
    let features = catch_unwind(|| ingest_features(m, n, att));
    if features.is_err() {
        unsafe { *forest = std::ptr::null() };
        return;
    }

    unsafe {
        xrf_hub(
            DataFrameClassification::new(
                features.unwrap(),
                decision,
                yc as u32,
                m as usize,
                n as usize,
            ),
            trees,
            tries,
            threads,
            forest,
            todo,
            seed,
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn xrf_reg(
    n: i32,
    m: i32,
    trees: i32,
    tries: i32,
    att: *const u8,
    y: *const f64,
    threads: i32,
    forest: *mut *const PackedForest,
    todo: u8,
    seed: u64,
) {
    let decision: Vec<f64> = (0..n).map(|e| unsafe { *y.add(e as usize) }).collect();
    let features = catch_unwind(|| ingest_features(m, n, att));
    if features.is_err() {
        unsafe { *forest = std::ptr::null() };
        return;
    }

    unsafe {
        xrf_hub(
            DataFrameRegression::new(features.unwrap(), decision, m as usize, n as usize),
            trees,
            tries,
            threads,
            forest,
            todo,
            seed,
        );
    }
}

unsafe extern "C" {
    fn pull_feature(
        atts: *const u8,
        which: u32,
        n: *const i32,
        ti: *const i32,
        data: *const *const u8,
    );
}

fn ingest_features(m: i32, n: i32, att: *const u8) -> Vec<DfAttribute> {
    let mut features: Vec<DfAttribute> = Vec::with_capacity(m as usize);
    for e in 0..m {
        let mut an: i32 = 17;
        let mut ti: i32 = 19;
        let mut v: *const u8 = std::ptr::null();
        unsafe {
            pull_feature(
                att,
                e as u32,
                &mut an as *mut i32,
                &mut ti as *mut i32,
                &mut v as *mut *const u8,
            )
        };
        if an != n {
            panic!("Wrong feature size");
        }
        let x = match ti {
            i32::MIN..=-3 => panic!("Unsupported data type in input"),
            -2 => DfAttribute::Logical(v as *const u32),
            -1 => DfAttribute::Numeric(v as *const f64),
            0 => DfAttribute::Integer(v as *const i32),
            x => DfAttribute::Factor(x as u32, v as *const i32),
        };
        features.push(x);
    }
    features
}

unsafe fn xrf_hub<R>(
    df: R,
    trees: i32,
    tries: i32,
    threads: i32,
    forest: *mut *const PackedForest,
    todo: u8,
    seed: u64,
) where
    R: RfInput<FeatureId = u32>,
    R: RefUnwindSafe,
    Forest<R>: Packable,
    R: Sync + Send,
    R::Pivot: Send + Sync,
    R::Vote: Send + Sync,
    R::FeatureId: Send + Sync,
    R::VoteAggregator: Send + Sync,
{
    // let mut ctrl = RfControl::new_basic_rf(ntree as usize, mtry as usize, seed);
    let mut threads = threads as usize;
    if threads == 0 {
        threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1);
    }
    // ctrl.max_depth = 512; //Default 32
    let make_importance = (todo & 8) != 0;
    let make_oob = (todo & 4) != 0;
    let save_forest = (todo & 2) != 0;

    let fo = catch_unwind(|| {
        if threads == 1 {
            Forest::new(
                &df,
                trees as usize,
                tries as usize,
                save_forest,
                make_importance,
                make_oob,
                seed,
            )
        } else {
            Forest::new_parallel(
                &df,
                trees as usize,
                tries as usize,
                save_forest,
                make_importance,
                make_oob,
                seed,
                threads,
            )
        }
    });
    if fo.is_err() {
        unsafe { *forest = std::ptr::null() };
        return;
    }
    let fo = fo.unwrap();
    unsafe {
        *forest = fo.pack().into_raw();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn finalise_xrf(forest: *mut PackedForest) {
    drop(PackedForest::from_raw(forest));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn importance_xrf(
    forest: *mut PackedForest,
    importance: *mut f64,
    normalise: bool,
    noimp: *mut i32,
) -> *mut PackedForest {
    let fo = PackedForest::from_raw(forest);
    if fo.parameters().4 {
        if normalise {
            fo.importance_normalised(importance);
        } else {
            fo.importance(importance);
        }
        unsafe { *noimp = 0 }
    } else {
        unsafe { *noimp = 1 }
    }
    fo.into_raw()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn predict_xrf(
    forest: *mut PackedForest,
    att: *const u8,
    ncat: i32,
    n: i32,
    m: i32,
    get_votes: bool,
    seed: u64,
    threads: i32,
    cat: *mut i32,
    num: *mut f64,
    fail: *mut bool,
) -> *mut PackedForest {
    let fo = PackedForest::from_raw(forest);
    if att.is_null() {
        if !get_votes {
            fo.oob(cat, num);
        } else {
            fo.oob_votes(cat, n as usize);
        }
    } else {
        let mut threads = threads as usize;
        if threads == 0 {
            threads = std::thread::available_parallelism()
                .map(|x| x.get())
                .unwrap_or(1);
        }
        let err = catch_unwind(|| {
            let att = ingest_features(m, n, att);
            if !get_votes {
                fo.predict(
                    att,
                    ncat as u32,
                    n as usize,
                    m as usize,
                    seed,
                    threads as usize,
                    cat,
                    num,
                );
            } else {
                fo.predict_votes(
                    att,
                    ncat as u32,
                    n as usize,
                    m as usize,
                    threads as usize,
                    cat,
                );
            }
        })
        .is_err();
        unsafe { *fail = err };
    }
    fo.into_raw()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn info_xrf(forest: *mut PackedForest, info: *mut i32) -> *mut PackedForest {
    let fo = PackedForest::from_raw(forest);
    let params = fo.parameters();
    unsafe {
        *info.add(0) = params.0 as i32;
        *info.add(1) = if params.2 { 1 } else { 0 };
        *info.add(2) = if params.3 { 1 } else { 0 };
        *info.add(3) = if params.4 { 1 } else { 0 };
    }
    fo.into_raw()
}

unsafe extern "C" {
    fn fill_raw_vec(within: *const u8, idx: i32, len: i32) -> *mut u8;
    fn fill_double_vec(within: *const u8, idx: i32, len: i32) -> *mut f64;
    fn fill_int_vec(within: *const u8, idx: i32, len: i32) -> *mut i32;
    fn decode_raw_vec(within: *const u8, idx: i32) -> *mut u8;
    fn decode_int_vec(within: *const u8, idx: i32) -> *mut i32;
    fn decode_double_vec(within: *const u8, idx: i32) -> *mut f64;
    fn elem_length(within: *const u8, idx: i32) -> i32;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn codec_xrf(forest: *mut PackedForest, ans: *mut u8) -> *mut PackedForest {
    if forest.is_null() {
        let (ntree, ftype, has_forest, has_oob, has_importance) = unsafe {
            let m_len = elem_length(ans, 0);
            assert_eq!(m_len, 5);
            let manifest = decode_int_vec(ans, 0);
            (
                *manifest,
                *manifest.add(1),
                *manifest.add(2) != 0,
                *manifest.add(3) != 0,
                *manifest.add(4) != 0,
            )
        };
        let mut fo = if has_forest {
            unsafe {
                let fo_len = elem_length(ans, 1);
                if fo_len != elem_length(ans, 2) || fo_len != elem_length(ans, 3) {
                    panic!("Serialised data corrupted");
                }
                let fo_flags = decode_int_vec(ans, 1);
                let fo_features = decode_int_vec(ans, 2);
                let fo_values = decode_double_vec(ans, 3);
                PackedForest::from_flattened(
                    fo_len as usize,
                    ftype,
                    fo_flags,
                    fo_features,
                    fo_values,
                )
            }
        } else {
            PackedForest::from_nothing(ntree, ftype)
        };
        if has_oob {
            unsafe {
                let oob_len = elem_length(ans, 4);
                let oob_str = decode_raw_vec(ans, 4);
                fo.deserialise_oob_raw(oob_str, oob_len as usize);
            }
        }
        if has_importance {
            unsafe {
                let imp_len = elem_length(ans, 5);
                let imp_str = decode_raw_vec(ans, 5);
                fo.deserialise_importance_raw(imp_str, imp_len as usize);
            }
        }
        fo.into_raw()
    } else {
        let fo = PackedForest::from_raw(forest);
        let (ntree, ftype, has_forest, has_oob, has_importance) = fo.parameters();
        unsafe {
            let manifest = fill_int_vec(ans, 0, 5);
            *manifest = ntree as i32;
            *manifest.add(1) = ftype as i32;
            *manifest.add(2) = if has_forest { 1 } else { 0 };
            *manifest.add(3) = if has_oob { 1 } else { 0 };
            *manifest.add(4) = if has_importance { 1 } else { 0 };
        }
        if has_forest {
            fo.flatten(ans, (1, 2, 3));
        }
        if has_oob {
            fo.serialise_oob_raw(ans, 4);
        }
        if has_importance {
            fo.serialise_importance_raw(ans, 5);
        }
        fo.into_raw()
    }
}
