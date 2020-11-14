#![allow(non_snake_case)]

#[inline(always)]
unsafe fn get<T: Copy>(p: *const T, offset: usize) -> T {
    p.offset(offset as isize).read()
}

#[inline(always)]
unsafe fn get_mut<T: Copy>(p: *mut T, offset: usize) -> *mut T {
    p.offset(offset as isize)
}

pub unsafe fn sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
    let (M, K, N) = (m, k, n);
    let A = |y: usize, x: usize| get(a, y * K + x);
    let B = |y: usize, x: usize| get(b, y * N + x);
    let C = |y: usize, x: usize| get_mut(c, y * N + x);
    for i in 0..M {
        for k in 0..K {
            for j in 0..N {
                *C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn mat<F>(rows: usize, cols: usize, mut f: F) -> Vec<f32>
    where
        F: FnMut(usize, usize) -> usize,
    {
        let mut m = vec![];
        for i in 0..rows {
            for j in 0..cols {
                m.push(f(i, j) as f32)
            }
        }
        m
    }

    unsafe fn mm_sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
        let alpha = 1.0;
        let beta = 1.0;
        let rsa = k as isize;
        let csa = 1;
        let rsb = n as isize;
        let csb = 1;
        let rsc = n as isize;
        let csc = 1;
        #[rustfmt::skip]
        matrixmultiply::sgemm(m, k, n, alpha,
                              a, rsa, csa,
                              b, rsb, csb,
                              beta, c, rsc, csc);
    }

    fn var<T: std::str::FromStr>(name: &str, default: T) -> T {
        match std::env::var(name) {
            Ok(s) => s
                .parse()
                .unwrap_or_else(|_| panic!("Failed to parse {}", s)),
            Err(_) => default,
        }
    }

    fn bench_one(iters: usize, name: &str, mut f: impl FnMut()) {
        let mut ts = vec![];
        for _ in 0..iters {
            let t = Instant::now();
            f();
            ts.push(t.elapsed())
        }
        let sum: Duration = ts.iter().sum();
        let avg = sum / iters as u32;
        let min = ts.iter().min().unwrap();
        println!(
            "{} iters of {} in {:?}, avg: {:?}, min: {:?}",
            iters, name, sum, avg, min
        );
    }

    #[test]
    fn bench() {
        let n: usize = var("N", 128);
        let iters: usize = var("ITERS", 1);

        println!("Benching {} iters of {}x{}", iters, n, n);

        let (m, k, n) = (n, n, n);
        let a = mat(m, k, |i, j| i + j);
        let b = mat(k, n, |i, j| i + j);
        let mut c1 = mat(m, n, |_, _| 0);
        let mut c2 = mat(m, n, |_, _| 0);

        bench_one(iters, "sgemm", || unsafe {
            sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr());
        });
        bench_one(iters, "mm_sgemm", || unsafe {
            mm_sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr());
        });
    }

    #[test]
    fn it_works() {
        let (m, k, n) = (3, 4, 5);
        let a = mat(m, k, |i, j| i + j);
        let b = mat(k, n, |i, j| i + j);
        let mut c1 = mat(m, n, |_, _| 0);
        let mut c2 = mat(m, n, |_, _| 0);

        unsafe {
            sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr());
            mm_sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr());
        }
        assert_eq!(c1, c2);
    }
}
