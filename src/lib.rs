#![feature(test)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]

extern crate test;
extern crate npy;
extern crate ndarray;
extern crate crossbeam;
extern crate realfft;
extern crate rustfft;
extern crate itertools;

mod cic;
mod correlate;

use cic::*;

use pyo3::prelude::*;




use bitvec::prelude::*;
use std::io::Cursor;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use socket2::{Socket, Domain, Type};
use std::io::Read;
use std::io::Result;
use std::net::SocketAddr;
use std::sync::Arc;

use std::time::Duration;
use std::thread;

use numpy::ndarray::{Array, Ix2, ShapeBuilder};
use numpy::{PyArray2, IntoPyArray};

use crossbeam::queue::ArrayQueue;

fn bool_to_i32(b: bool) -> i32 {
    if b {
        1
    } else {
        -1
    }
}

fn bool_to_i8(b: bool) -> i8 {
    if b {
        1
    } else {
        -1
    }
}

#[derive(Debug, Clone, Copy)]
struct Sample {
    pub seq: u32,
    pub data: [u32; 6]
}

impl Sample {
    fn parse(rdr: &mut Cursor<&[u8]>) -> Result<Sample> {
        let seq = rdr.read_u32::<LittleEndian>()?;
        let data = [
            rdr.read_u32::<LittleEndian>()?,
            rdr.read_u32::<LittleEndian>()?,
            rdr.read_u32::<LittleEndian>()?,
            rdr.read_u32::<LittleEndian>()?,
            rdr.read_u32::<LittleEndian>()?,
            rdr.read_u32::<LittleEndian>()?,
        ];
        Ok(Sample { seq, data })
    }
}

#[pyclass]
struct RawListener {
    recv: Arc<ArrayQueue<[Sample; 48]>>,
    remaining: Vec<Sample>,
    cic: FastCIC<4, 192>,
}

#[pymethods]
impl RawListener {
    #[new]
    fn new(addr: String, decimation: usize) -> RawListener {
        let queue: Arc<ArrayQueue<[Sample; 48]>> = Arc::new(ArrayQueue::new(32_768));
        let queue_2 = queue.clone();

        thread::spawn(move || {
            let mut buf = [0; 2048];
            let mut socket = Socket::new(Domain::IPV4, Type::DGRAM, None).unwrap();
            socket.set_recv_buffer_size(128 * 1024 * 1024).unwrap();

            let address: SocketAddr = addr.parse().unwrap();
            socket.bind(&address.into()).unwrap();

            let mut seq = 0;

            let mut last_drop_state = false;

            'outer: loop {
                let number_of_bytes = socket.read(&mut buf).expect("no data received");
                assert_eq!(number_of_bytes, 1344);

                let mut rdr = Cursor::new(&buf[..number_of_bytes]);

                let mut packet_idx = 0;

                let mut packet_buf = Vec::new();

                while let Ok(data) = Sample::parse(&mut rdr) {
                    let new_seq = data.seq;

                    packet_buf.push(data);

                    if (new_seq - 1) != seq {
                        println!("seq error: {:#x} {:#x} {} {}", seq, new_seq, packet_idx, new_seq - seq);
                    }

                    seq = new_seq;
                    packet_idx += 1;
                }

                match queue.force_push(packet_buf.try_into().unwrap()) {
                    Some(dropped) => {
                        if last_drop_state == false {
                            println!("dropping packet {}", dropped[0].seq);
                            last_drop_state = true;
                        }
                    }
                    _ => {
                        if last_drop_state == true {
                            println!("recovered");
                            last_drop_state = false;
                        }
                    }
                }
            }
        });

        RawListener {
            recv: queue_2,
            remaining: Vec::new(),
            cic: FastCIC::new(decimation),
        }
    }

    fn chunk<'py>(
        &'py mut self,
        py: Python<'py>,
        size: usize) -> &'py PyArray2<i32> {
        py.allow_threads(move || self.get_chunk_array(size)).into_pyarray(py)
    }
}

impl RawListener {
    fn get_chunk(&mut self, size: usize) -> Vec<Sample> {
        let mut chunk = self.remaining.drain(..).collect::<Vec<Sample>>();
        while chunk.len() < size {
            if let Some(sample) = self.recv.pop() {
                chunk.extend_from_slice(&sample);
            } else {
                thread::sleep(Duration::from_millis(1));
                // println!("waiting...");
            }
        };
        chunk.drain(size..).for_each(|s| self.remaining.push(s));
        chunk
    }

    fn get_chunk_array(&mut self, size: usize) -> Array<i32, Ix2> {
        let decimation = self.cic.decimation;
        assert_eq!(size % decimation, 0);

        let chunk = self.get_chunk(size);

        assert_eq!(chunk.len() % decimation, 0);

        let mut a = Array::<i32, _>::zeros((192usize, chunk.len() / decimation).f());
        let mut output_sample_idx = 0;

        for (_sample_idx, i) in chunk.iter().enumerate() {
            let mut wtr = Vec::new();
            for d in i.data {
                wtr.write_u32::<LittleEndian>(d).unwrap();
            }

            let mut buf = [0; 192];
            for channel_idx in 0..192 {
                let u32_idx = channel_idx / 32;
                let bit_idx = channel_idx % 32;
                let bit = ((i.data[u32_idx] >> bit_idx) & 1) == 1;
                buf[channel_idx] = if bit {
                    1
                } else {
                    -1
                }
            }

            if let Some(x) = self.cic.update(buf) {
                for channel_idx in 0..192 {
                    let shuffled_idx = match channel_idx / 96 {
                        0 => 1 + channel_idx * 2,
                        1 => (channel_idx - 96) * 2,
                        _ => unreachable!(),
                    };

                    a[[shuffled_idx, output_sample_idx]] = x[channel_idx];
                }
                output_sample_idx += 1;
            }
        }
        a
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn mic_host(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RawListener>()?;
    m.add_class::<correlate::Correlator>()?;
    Ok(())
}