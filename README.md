# mm

Compile like this:

``` sh
RAYON_NUM_THREADS= ITERS=128 N=512 RUSTFLAGS="-C target-cpu=native" cargo test --release bench -- --nocapture
```
