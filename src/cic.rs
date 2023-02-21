use std::intrinsics::black_box;
use std::simd::{i32x8, Mask, ToBitMask};
use test::Bencher;

#[derive(Debug, Copy, Clone)]
struct Integrator {
    yn: i32,
    ynm: i32,
}

impl Integrator {
    fn new() -> Integrator {
        Integrator { yn: 0, ynm: 0 }
    }

    fn update(&mut self, inp: i32) -> i32 {
        self.ynm = self.yn;
        self.yn = self.ynm + inp;
        self.yn
    }
}

#[derive(Debug, Copy, Clone)]
struct Comb {
    xn: i32,
    xnm: i32,
}

impl Comb {
    fn new() -> Comb {
        Comb { xn: 0, xnm: 0 }
    }

    fn update(&mut self, inp: i32) -> i32 {
        self.xnm = self.xn;
        self.xn = inp;
        self.xn - self.xnm
    }
}

#[derive(Debug, Clone)]
struct CIC {
    integrators: Vec<Integrator>,
    combs: Vec<Comb>,
    decimation: usize,
    count: usize,
}

impl CIC {
    fn new(decimation: usize, stages: usize) -> CIC {
        let mut integrators = Vec::new();
        let mut combs = Vec::new();
        for _ in 0..stages {
            integrators.push(Integrator::new());
            combs.push(Comb::new());
        }
        CIC { integrators, combs, decimation, count: 0 }
    }

    fn update(&mut self, inp: i32) -> Option<i32> {
        let mut out = inp;
        for i in 0..self.integrators.len() {
            out = self.integrators[i].update(out);
        }

        if self.count == self.decimation - 1 {
            self.count = 0;
            for i in 0..self.combs.len() {
                out = self.combs[i].update(out);
            }
            Some(out)
        } else {
            self.count += 1;
            None
        }
    }
}

pub struct SimdCIC<const STAGES: usize, const WIDTH: usize> {
    integrator_state: [[i32x8; WIDTH]; STAGES],
    comb_state: [[i32x8; WIDTH]; STAGES],

    decimation: usize,
    count: usize,
}


impl <const STAGES: usize, const WIDTH: usize> SimdCIC<STAGES, WIDTH> {
    pub(crate) fn new(decimation: usize) -> Self {
        SimdCIC {
            decimation,
            count: 0,
            comb_state: [[i32x8::splat(0); WIDTH]; STAGES],
            integrator_state: [[i32x8::splat(0); WIDTH]; STAGES],
        }
    }

    pub(crate) fn update(&mut self, sample: [u8; WIDTH]) -> Option<[i32x8; WIDTH]> {

        let ones = i32x8::splat(1);
        let neg_one = i32x8::splat(-1);

        let mut x = [i32x8::splat(0); WIDTH];

        for i in 0..WIDTH {
            let mask = Mask::from_bitmask(sample[i]);
            x[i] = mask.select(ones, neg_one);
        }

        for i in 0..STAGES {
            for j in 0..WIDTH {
                self.integrator_state[i][j] += x[j];
            }
            x = self.integrator_state[i].clone();
            // println!("fast integrator state: {} {:?}", i, self.integrator_state[i]);
        }

        if self.count < self.decimation - 1 {
            self.count += 1;
            None
        } else {
            self.count = 0;
            for i in 0..STAGES {
                for j in 0..WIDTH {
                    let comb = self.comb_state[i][j].clone();
                    self.comb_state[i][j] = x[j];

                    x[j] -= comb;
                }
                // println!("fast comb state: {} {:?}", i, self.comb_state[i]);
            }
            Some(x)
        }
    }
}


pub struct FastCIC<const STAGES: usize, const WIDTH: usize> {
    pub(crate) decimation: usize,
    count: usize,

    integrator_state: [[i32; WIDTH]; STAGES],
    comb_state: [[i32; WIDTH]; STAGES],
}

impl <const STAGES: usize, const WIDTH: usize> FastCIC<STAGES, WIDTH> {
    pub(crate) fn new(decimation: usize) -> Self {
        FastCIC {
            decimation,
            count: 0,
            comb_state: [[0; WIDTH]; STAGES],
            integrator_state: [[0; WIDTH]; STAGES],
        }
    }

    pub(crate) fn update(&mut self, sample: [i32; WIDTH]) -> Option<[i32; WIDTH]> {
        let mut x = sample;
        for i in 0..STAGES {
            for j in 0..WIDTH {
                self.integrator_state[i][j] = self.integrator_state[i][j].wrapping_add(x[j]);
            }
            x = self.integrator_state[i];
            // println!("fast integrator state: {} {:?}", i, self.integrator_state[i]);
        }

        if self.count < self.decimation - 1 {
            self.count += 1;
            None
        } else {
            self.count = 0;
            for i in 0..STAGES {
                for j in 0..WIDTH {
                    let comb = self.comb_state[i][j];
                    self.comb_state[i][j] = x[j];

                    x[j] = x[j].wrapping_sub(comb);
                }
                // println!("fast comb state: {} {:?}", i, self.comb_state[i]);
            }
            Some(x)
        }
    }
}

#[test]
fn test_cic() {
    let mut cic = CIC::new(4, 2);
    let mut fast_cic = FastCIC::<2, 1>::new(4);
    for i in 0..100 {
        let out = cic.update(i / 3);
        let fast_out = fast_cic.update([i / 3]);
        if let Some(out) = out {
            assert_eq!(out, fast_out.unwrap()[0], "{}", i);
        }
    }
}




#[bench]
fn bench_cic(b: &mut Bencher) {
    let mut cics = Vec::new();

    for i in 0..192 {
        cics.push(CIC::new(4, 4))
    }

    b.bytes = 192 / 8;

    b.iter(|| {
        let inputs = black_box([0; 6]);

        let mut buf = [0; 192];
        for channel_idx in 0..192 {
            let u32_idx = channel_idx / 32;
            let bit_idx = channel_idx % 32;
            let bit = ((inputs[u32_idx] >> bit_idx) & 1) == 1;
            buf[channel_idx] = if bit {
                1
            } else {
                -1
            }
        }
        for i in 0..192 {
            black_box(cics[i].update(buf[i]));
        }
    })
}

#[bench]
fn bench_fast_cic(b: &mut Bencher) {
    let mut fast_cic = FastCIC::<4, 192>::new(4);

    b.bytes = 192 / 8;

    b.iter(|| {
        let inputs = black_box([0; 6]);

        let mut buf = [0; 192];
        for channel_idx in 0..192 {
            let u32_idx = channel_idx / 32;
            let bit_idx = channel_idx % 32;
            let bit = ((inputs[u32_idx] >> bit_idx) & 1) == 1;
            buf[channel_idx] = if bit {
                1
            } else {
                -1
            }
        }
        black_box(fast_cic.update(buf))
    })
}

#[bench]
fn bench_fast_cic_no_bittwiddle(b: &mut Bencher) {
    let mut fast_cic = FastCIC::<4, 192>::new(4);

    b.bytes = 192 / 8;

    b.iter(|| {
        let inputs = black_box([0; 192]);
        black_box(fast_cic.update(inputs))
    })
}

#[bench]
fn bench_simd_cic(b: &mut Bencher) {
    let mut simd_cic = SimdCIC::<4, 24>::new(4);
    let inputs = [0; 24];

    b.bytes = 192 / 8;

    b.iter(|| {
        black_box(simd_cic.update(black_box(inputs)))
    })
}