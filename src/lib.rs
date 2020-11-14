#[derive(Clone, Copy)]
struct Matrix {
    ptr: *const f32,
    cols: usize,
}

impl Matrix {
    #[inline(always)]
    unsafe fn get(self, y: usize, x: usize) -> f32 {
        let offset = y * self.cols + x;
        self.ptr.offset(offset as isize).read()
    }
}

#[derive(Clone, Copy)]
struct MutMatrix {
    ptr: *mut f32,
    cols: usize,
}

impl MutMatrix {
    #[inline(always)]
    unsafe fn get_mut(self, y: usize, x: usize) -> *mut f32 {
        let offset = y * self.cols + x;
        self.ptr.offset(offset as isize)
    }
}

unsafe impl Send for Matrix {}
unsafe impl Send for MutMatrix {}
unsafe impl Sync for Matrix {}
unsafe impl Sync for MutMatrix {}

pub unsafe fn sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
    let a = Matrix { ptr: a, cols: k };
    let b = Matrix { ptr: b, cols: n };
    let c = MutMatrix { ptr: c, cols: n };

    use rayon::prelude::*;

    (0..m).into_par_iter().for_each(move |i| {
        for k in 0..k {
            for j in 0..n {
                *c.get_mut(i, j) += a.get(i, k) * b.get(k, j);
            }
        }
    });

    // let block_size = 16;
    // let block = |block_n: usize| {
    //     let start = block_n * block_size;
    //     start..(start + block_size)
    // };

    // (0..m).into_par_iter().for_each(move |ii| {
    //         for bj in 0..(n / block_size) {
    //     for bk in 0..(k / block_size) {
    //             for kk in block(bk) {
    //                 for jj in block(bj) {
    //                     *c.get_mut(ii, jj) += a.get(ii, kk) * b.get(kk, jj);
    //                 }
    //             }
    //         }
    //     }
    // });
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

    unsafe fn mm_mt_sgemm(m: usize, k: usize, n: usize, a: *const f32, b: *const f32, c: *mut f32) {
        let alpha = 1.0;
        let beta = 1.0;
        let rsa = k as isize;
        let csa = 1;
        let rsb = n as isize;
        let csb = 1;
        let rsc = n as isize;
        let csc = 1;
        #[rustfmt::skip]
        matrixmultiply_mt::sgemm(m, k, n, alpha,
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

    fn bench_one(iters: usize, flops: f64, name: &str, mut f: impl FnMut()) {
        let mut ts = vec![];
        for _ in 0..iters {
            let t = Instant::now();
            f();
            ts.push(t.elapsed())
        }
        let sum: Duration = ts.iter().sum();
        let avg = sum / iters as u32;
        let min = ts.iter().min().unwrap();
        let avg_gflops = (flops / 1e9) / avg.as_secs_f64();
        println!(
            "{} iters of {} in {:?}, avg: {:?}, min: {:?}, avg GFLOPS: {}",
            iters, name, sum, avg, min, avg_gflops
        );
    }

    #[test]
    fn bench() {
        let n: usize = var("N", 128);
        let iters: usize = var("ITERS", 1);

        println!("Benching {} iters of {}x{}", iters, n, n);

        let (m, k, n) = (n, n, n);
        let flops = (2 * m * n * k) as f64;
        let a = mat(m, k, |i, j| i + j);
        let b = mat(k, n, |i, j| i + j);
        let mut c1 = mat(m, n, |_, _| 0);
        let mut c2 = mat(m, n, |_, _| 0);

        bench_one(iters, flops, "sgemm", || unsafe {
            sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c1.as_mut_ptr());
        });
        bench_one(iters, flops, "mm_sgemm", || unsafe {
            mm_sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr());
        });
        bench_one(iters, flops, "mm_mt_sgemm", || unsafe {
            mm_mt_sgemm(m, k, n, a.as_ptr(), b.as_ptr(), c2.as_mut_ptr());
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
