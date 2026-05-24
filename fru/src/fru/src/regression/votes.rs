use super::DataFrame;
use xrf::VoteAggregator;

#[derive(Clone)]
pub struct Votes {
    sum: f64,
    n: usize,
}

impl Votes {
    pub fn new() -> Self {
        Votes { sum: 0., n: 0 }
    }
    pub unsafe fn from_raw(vals: *const u8) -> Self {
        Votes {
            sum: f64::from_le_bytes(unsafe {
                [
                    *vals,
                    *vals.add(1),
                    *vals.add(2),
                    *vals.add(3),
                    *vals.add(4),
                    *vals.add(5),
                    *vals.add(6),
                    *vals.add(7),
                ]
            }),
            n: u64::from_le_bytes(unsafe {
                [
                    *vals.add(8),
                    *vals.add(8 + 1),
                    *vals.add(8 + 2),
                    *vals.add(8 + 3),
                    *vals.add(8 + 4),
                    *vals.add(8 + 5),
                    *vals.add(8 + 6),
                    *vals.add(8 + 7),
                ]
            })
            .try_into()
            .unwrap(),
        }
    }
    pub unsafe fn into_raw(&self, vals: *mut u8) {
        self.sum
            .to_le_bytes()
            .iter()
            .chain((self.n as u64).to_le_bytes().iter())
            .enumerate()
            .for_each(|(e, &v)| {
                unsafe { *vals.add(e) = v };
            })
    }
    pub fn collapse(&self) -> f64 {
        if self.n > 0 {
            self.sum / (self.n as f64)
        } else {
            f64::NAN
        }
    }
}

impl VoteAggregator<DataFrame> for Votes {
    fn new(_input: &DataFrame) -> Self {
        Votes::new()
    }
    fn ingest_vote(&mut self, v: f64) {
        self.sum += v;
        self.n += 1;
    }
    fn merge(&mut self, other: &Self) {
        self.sum += other.sum;
        self.n += other.n;
    }
}
