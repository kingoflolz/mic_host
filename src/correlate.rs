use std::intrinsics::black_box;
use std::sync::Arc;
use test::Bencher;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use realfft::num_complex::Complex;

use itertools::Itertools;

use rustfft;
use rayon::iter::ParallelBridge;
use numpy::ndarray::{Array2, parallel::prelude::*};

use pyo3::prelude::*;

#[pyclass]
pub(crate) struct Correlator {
    fft_impl: Arc<dyn rustfft::Fft<f32>>,
    ifft_impl: Arc<dyn rustfft::Fft<f32>>,

    rfft_impl: Arc<dyn RealToComplex<f32>>,
    rifft_impl: Arc<dyn ComplexToReal<f32>>,
    length: usize,
}

#[pymethods]
impl Correlator {
    #[new]
    fn new(length: usize) -> Correlator {
        let mut planner = rustfft::FftPlanner::new();
        let fft_impl = planner.plan_fft_forward(length);
        let ifft_impl = planner.plan_fft_inverse(length);

        let mut planner = RealFftPlanner::new();
        let rfft_impl = planner.plan_fft_forward(length - 1);
        let rifft_impl = planner.plan_fft_inverse(length - 1);

        let _ = rayon::ThreadPoolBuilder::new().num_threads(15).build_global();

        Correlator { fft_impl, ifft_impl, rfft_impl, rifft_impl, length }
    }

    #[pyo3(name = "correlate")]
    fn correlate_py<'py>(
        &'py mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, i32>,
        max_offset: PyReadonlyArray2<'py, i32>,
    ) -> (&'py PyArray2<i32>, &'py PyArray2<f32>) {
        let input = input.as_array();
        let max_offset = max_offset.as_array();
        let (offset, corr) = py.allow_threads(move || self.correlate(input, max_offset));
        (offset.into_pyarray(py), corr.into_pyarray(py))
    }

    #[pyo3(name = "rfft")]
    fn rfft_py<'py>(
        &'py mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<'py, i32>,
    ) -> (&'py PyArray2<Complex<f32>>) {
        let input = input.as_array();
        py.allow_threads(move || self.rfft(input)).into_pyarray(py)
    }
}

impl Correlator {
    fn correlate(&self, input: ArrayView2<i32>, max_offset: ArrayView2<i32>) -> (Array2<i32>, Array2<f32>) {
        assert_eq!(input.shape()[0], 192);
        assert_eq!(input.shape()[1], self.length);

        assert_eq!(max_offset.shape()[0], 192);
        assert_eq!(max_offset.shape()[1], 192);

        let mut input_complex = input.as_standard_layout().mapv(|x| Complex{ re: x as f32, im: 0.0f32});

        for i in 0..192 {
            self.fft_impl.process(input_complex.row_mut(i).as_slice_mut().unwrap());
        }

        let mut out_offset = Array2::zeros((192, 192));
        let mut out_corr = Array2::zeros((192, 192));

        input_complex.rows().into_iter().enumerate()
            .cartesian_product(input_complex.rows().into_iter().enumerate()).into_iter()
            .zip(
                out_offset.as_slice_mut().unwrap().iter_mut()
                .zip(out_corr.as_slice_mut().unwrap().iter_mut())).par_bridge()
            .for_each(|(((aa_idx, aa), (bb_idx, bb)), (o, c))| {
                if aa_idx > bb_idx {
                    return;
                }
                let mut out = Array1::zeros(self.length);
                out.assign(&(&bb.mapv(|x| x.conj()) * &aa));
                out[[0]] = Complex{ re: 0.0, im: 0.0 };

                self.ifft_impl.process(out.as_slice_mut().unwrap());

                let out = out.mapv(|x| x.re);

                // let mut out = self.rifft_impl.make_output_vec();
                //
                // self.rifft_impl.process(
                //     freqs.as_slice_mut().unwrap(),
                //     out.as_mut_slice()
                // ).unwrap();

                // TODO faster argmax
                let (argmax, max) = (0..max_offset[[aa_idx, bb_idx]] as usize)
                    .chain((self.length - max_offset[[aa_idx, bb_idx]] as usize)..self.length)
                    .map(|x| (x, out[x]))
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();

                *o = if argmax < self.length / 2 {
                    argmax as i32
                } else {
                    (argmax - self.length) as i32
                };
                *c = max;
            });

        for i in 0..192 {
            for j in 0..i {
                out_offset[[i, j]] = -out_offset[[j, i]];
                out_corr[[i, j]] = out_corr[[j, i]];
            }
        }

        (out_offset, out_corr)
    }

    fn rfft(&self, input: ArrayView2<i32>) -> Array2<Complex<f32>> {
        assert_eq!(input.shape()[0], 192);
        assert_eq!(input.shape()[1], self.length);

        let mut input = input.as_standard_layout().mapv(|x| x as f32);
        let mut rffts = Array2::zeros((192, self.length / 2));

        for i in 0..192 {
            self.rfft_impl.process(&mut input.row_mut(i).as_slice_mut().unwrap()[0..self.length - 1],
                                   rffts.row_mut(i).as_slice_mut().unwrap()).unwrap();
            rffts[[i, 0]] = Complex{ re: 0.0f32, im: 0.0f32 };
        }

        rffts
    }
}



use ndarray::{Array1, ArrayView2};
use numpy::{Complex32, IntoPyArray, PyArray2, PyReadonlyArray2};

// #[bench]
// fn bench_correlator(b: &mut Bencher) {
//     let corr = Correlator::new(32 * 1024);
//
//     let mut input = Vec::new();
//
//     for _ in 0..192 {
//         input.push(vec![0.0f32; 32 * 1024]);
//     }
//
//     b.iter(|| {
//         let mut input = black_box(input.clone());
//         corr.correlate(black_box(input));
//     })
// }
//
// #[test]
// fn test_correlator() {
//     let corr = Correlator::new(64 * 1024);
//
//     let mut input = Vec::new();
//
//     for _ in 0..192 {
//         input.push(vec![0.0f32; 64 * 1024]);
//     }
//
//     corr.correlate(input);
// }

// #[bench]
// fn bench_correlator(b: &mut Bencher) {
//
//     let corr = Correlator::new(32 * 1024);
//
//     let mut input = Array2::zeros((192, 32 * 1024));
//
//     b.iter(|| {
//         let mut input = black_box(input.clone());
//         corr.correlate(black_box(input.view()));
//     })
// }
//
// #[test]
// fn test_correlator() {
//     let corr = Correlator::new(32 * 1024);
//
//     let mut input = Array2::zeros((192, 32 * 1024));
//
//     corr.correlate(input.view());
// }

#[bench]
fn bench_real_fft(b: &mut Bencher) {
    let mut planner = RealFftPlanner::new();
    let plan = planner.plan_fft_forward(32 * 1024);

    let mut inputs = [0.0f32; 32 * 1024];

    b.iter(|| {
        let mut outputs = plan.make_output_vec();
        black_box(plan.process(&mut inputs, &mut outputs).unwrap());
        black_box(outputs);
    })
}

#[bench]
fn bench_avx_fft(b: &mut Bencher) {
    use rustfft::{FftPlannerAvx, num_complex::Complex};

    if let Ok(mut planner) = FftPlannerAvx::new() {
        let fft = planner.plan_fft_forward(32 * 1024);

        let mut buffer = vec![Complex{ re: 0.0f32, im: 0.0f32 }; 32 * 1024];
        b.iter(|| {
            black_box(fft.process(&mut buffer));
        });
    }
}
