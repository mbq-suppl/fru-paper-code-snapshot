pub fn midpoint(a: i32, b: i32) -> i32 {
    //Hacker's delight approach
    // creates floor((a+b)/2), which is what is needed
    // for negative inputs due to how integer pivot is executed
    ((a ^ b) >> 1) + (a & b)
}

pub fn ordering_vector<T: PartialOrd + Copy>(x: *const T, n: usize) -> Vec<u32> {
    let mut ans: Vec<u32> = (0..(n as u32)).collect();
    ans.sort_unstable_by(|a, b| {
        ((unsafe { *x.add(*a as usize) }).partial_cmp(&(unsafe { *x.add(*b as usize) }))).unwrap()
    });
    ans
}
