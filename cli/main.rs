use std::{io::{self, Write}, path::PathBuf, cmp};

use anyhow::Context;
use bao_tree::{BaoTree, Blake3Hasher, BlockSize, Hash, Hasher};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use fr32::Fr32Reader;

#[derive(Parser, Debug, Clone)]
#[clap(version)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Command,
    #[clap(
        short,
        long,
        default_value = "0",
        help = "Bao block size, the actual block size in bytes is 1024 << block_size"
    )]
    pub block_size: u8,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    Outboard {
        path: PathBuf,
        #[clap(long)]
        out: Option<PathBuf>,
    },
}

/// The hasher implementation for using BLAKE3.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CommpHasher;

impl Hasher for CommpHasher {
    const CHUNK_SIZE: usize = 64;

    fn hash_chunk(_start_chunk: u64, data: &[u8], _is_root: bool) -> Hash {
        let mut hashed = Sha256::digest(data);
        println!("vmx: leaf data: {:X?}", data);
        println!("vmx: leaf hashed: {:X?}", hashed);
        // CommP uses a 254-bit SHA-256 hash, hence zero the last two bits.
        hashed[31] &= 0b0011_1111;
        //hashed[31] &= 0b1111_1100;
        //hashed.into_array().into()
        //let foo: [u8; 32] = hashed.into();
        <[u8; 32]>::from(hashed).into()
    }
    fn hash_inner(left_child: &Hash, right_child: &Hash, _is_root: bool) -> Hash {
        let data: Vec<_> = [&left_child.as_bytes()[..], &right_child.as_bytes()[..]].concat();
        let mut hashed = Sha256::digest(&data);
        println!("vmx: inner data: {:X?}", data);
        println!("vmx: inner hashed: {:X?}", hashed);
        // CommP uses a 254-bit SHA-256 hash, hence zero the last two bits.
        hashed[31] &= 0b0011_1111;
        //hashed[31] &= 0b1111_1100;
        <[u8; 32]>::from(hashed).into()
    }
}

// From https://github.com/rvagg/rust-fil-commp-generate

// use a file as an io::Reader but pad out extra length at the end with zeros up
// to the padded size
struct Base2PadReader<R: io::Read> {
    size: usize,
    padsize: usize,
    pos: usize,
    inp: R,
}

impl<R: io::Read> io::Read for Base2PadReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        println!("vmx: base2padreader: pos, buf len: {:?} {:?}", self.pos, buf.len());
        let cs = if self.pos >= self.size {
            println!("vmx: do padding");
            for i in 0..buf.len() {
                buf[i] = 0;
            }
            cmp::min(self.padsize - self.pos, buf.len())
        } else {
            println!("vmx: base2padreader: fill buffer");
            self.inp.read(buf)?
        };
        println!("vmx: base2padreader: cs: {:?}", cs);
        self.pos = self.pos + cs;
        Ok(cs)
    }
}

fn piece_size(size: u64, next: bool) -> u64 {
    1u64 << (64 - size.leading_zeros() + if next { 1 } else { 0 })
}

// logic partly copied from Lotus' Fr32Reader which is also in go-fil-markets
// figure out how big this piece will be when padded
fn padded_size(size: u64) -> u64 {
    let bound = piece_size(size, false);
    if size <= bound {
        bound
    } else {
       piece_size(size, true)
    }
}


fn base2_padded<R: Sized + io::Read>(inp: &mut R, size: u64) -> Base2PadReader<&mut R> {
    let padded_size = padded_size(size);
    println!("vmx: padded_size: {:?}", padded_size);

    let base2_pad_reader = Base2PadReader {
        size: usize::try_from(size).unwrap(),
        padsize: usize::try_from(padded_size).unwrap(),
        pos: 0,
        inp: inp,
    };

    base2_pad_reader
}


/// A reader that appends zeros to the end of the data from the underlying reader.
pub struct ZeroPaddingReader<R> {
    inner: R,
    zeros_left: usize,
    eof_reached: bool,
}

impl<R: io::Read> ZeroPaddingReader<R> {
    /// Creates a new `ZeroPaddingReader` with the given underlying reader and number of zeros to append.
    pub fn new(reader: R, zeros: usize) -> Self {
        ZeroPaddingReader {
            inner: reader,
            zeros_left: zeros,
            eof_reached: false,
        }
    }
}

impl<R: io::Read> io::Read for ZeroPaddingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        println!("vmx: ZeroPaddingReader: zeros left: {:?}", self.zeros_left);
        if self.eof_reached {
            // If we've already reached EOF and have zeros left, fill the buffer with zeros.
            let len = std::cmp::min(buf.len(), self.zeros_left);
            buf[..len].fill(0);
            self.zeros_left -= len;
            Ok(len)
        } else {
            // Read from the inner reader.
            let n = self.inner.read(buf)?;
            if n == 0 {
                self.eof_reached = true;
                // If we've reached EOF and have zeros left, fill the rest of the buffer with zeros.
                let zeros_len = std::cmp::min(buf.len() - n, self.zeros_left);
                buf[n..n + zeros_len].fill(0);
                self.zeros_left -= zeros_len;
                Ok(n + zeros_len)
            } else {
                Ok(n)
            }
        }
    }
}


/// A reader that wraps another reader and appends a given number of zero bytes at the end.
pub struct ReadWithTrailingZeros<R: io::Read> {
    inner: R,
    zeros_to_append: usize,
    zeros_appended: usize,
}

impl<R: io::Read> ReadWithTrailingZeros<R> {
    /// Create a new `ReadWithTrailingZeros` from an existing reader and the number of trailing zeros.
    pub fn new(inner: R, zeros_to_append: usize) -> Self {
        Self {
            inner,
            zeros_to_append,
            zeros_appended: 0,
        }
    }
}

impl<R: io::Read> io::Read for ReadWithTrailingZeros<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        println!("vmx: readwithtrailingzeros: zeros appended: {:?}", self.zeros_appended);
        // If we still have data to read from the underlying reader
        if self.zeros_appended == 0 {
            let n = self.inner.read(buf)?;

            // If the underlying reader is exhausted
            if n == 0 {
                // Fall through to append zeros in this call
                // mark that underlying reader is done
                self.zeros_appended = 0; // technically already 0
            } else {
                return Ok(n);
            }
        }

        // Append zeros now
        if self.zeros_appended < self.zeros_to_append {
            let remaining = self.zeros_to_append - self.zeros_appended;
            let to_write = remaining.min(buf.len());
            for b in &mut buf[..to_write] {
                *b = 0;
            }
            self.zeros_appended += to_write;
            return Ok(to_write);
        }

        // No more data
        Ok(0)
    }
}

fn main2() {
    use std::io::Read;

    let data = vec![1, 2, 3];
    let mut cursor = io::Cursor::new(data);
    //let mut reader = ReadWithTrailingZeros::new(cursor, 5);
    let mut reader = base2_padded(&mut cursor, 3);

    let mut buf = [0u8; 3];
    let n = reader.read(&mut buf).unwrap();
    println!("Read {} bytes: {:?}", n, &buf[..n]);

    let n = reader.read(&mut buf).unwrap();
    println!("Read {} bytes: {:?}", n, &buf[..n]);

    let n = reader.read(&mut buf).unwrap();
    println!("Read {} bytes: {:?}", n, &buf[..n]);
}

fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    let bs = BlockSize::from_chunk_log(args.block_size);
    if args.block_size != 0 {
        println!("Using block size: {}", bs.bytes());
    }
    match args.command {
        Command::Outboard { path, out } => {
            let meta = std::fs::metadata(&path)?;
            anyhow::ensure!(meta.is_file(), "Path must be a file");
            let size = meta.len();
            let size = padded_size(size);
            let out = if let Some(out) = out {
                out
            } else {
                let name = path.file_name().context("context")?;
                let extension = "obao";
                std::env::current_dir()?.join(format!("{}.{}", name.to_string_lossy(), extension))
            };
            let mut source = std::fs::File::open(&path)?;
            let source_size = source.metadata().unwrap().len();
            println!("vmx: source size: {:?}", source_size);
            let target = std::fs::File::create(out)?;
            //let source = std::io::BufReader::with_capacity(1024 * 1024 * 16, source);
            let base2_padded = base2_padded(&mut source, source_size);
            //let base2_padded = ZeroPaddingReader::new(source, source_size as usize);
            //let base2_padded = ReadWithTrailingZeros::new(source, source_size as usize);
            let fr32_padded = Fr32Reader::new(base2_padded);
            //let fr32_padded = Fr32Reader::new(source);
            let mut target = std::io::BufWriter::with_capacity(1024 * 1024 * 16, target);
            let t0 = std::time::Instant::now();
            let tree = BaoTree::new(size, bs);
            let hash =
                //bao_tree::io::sync::outboard_post_order::<Blake3Hasher>(source, tree, &mut target)?;
                //bao_tree::io::sync::outboard_post_order::<CommpHasher>(source, tree, &mut target)?;
                bao_tree::io::sync::outboard_post_order::<CommpHasher>(fr32_padded, tree, &mut target)?;
            target.write_all(size.to_le_bytes().as_ref())?;
            let dt = t0.elapsed();
            let rate = size as f64 / dt.as_secs_f64();
            println!("{}", hash);
            println!(
                "{} bytes in {} seconds, {} bytes/sec",
                size,
                dt.as_secs_f64(),
                rate
            );
        }
    }
    Ok(())
}
