//! Decoder for the compressed reverb impulse responses embedded in
//! `TA_REVIR` presets.
//!
//! Layout (little endian):
//!
//! ```text
//! u32 channels (1 or 2), u32 frames, f32 peak, then a sequence of blocks
//! ```
//!
//! Each block is word aligned and codes 512 samples of one channel as an
//! LSB-first bitstream:
//!
//! * 9 bits: block length in 32-bit words minus one
//! * 6 bits: `a` (third predictor coefficient is `a - 64`)
//! * 7 bits: `b` (second coefficient is `b - 63`; the first is `47 - a' - b`)
//! * 3 x 24-bit signed warm-up samples
//! * 16 groups of 32 residuals: a 5-bit shift `k`, then per residual a prefix
//!   code for a quotient in -8..=7 followed by `k` low bits
//!
//! Samples are reconstructed with a 3rd order predictor in 1/16 units and
//! scaled by 2^-20. Stereo files interleave a mid and a side block per 512
//! frames (`L = (m + 2s) >> 1`, `R = (m - 2s) >> 1`).

const BLOCK: usize = 512;
const SCALE: f32 = 1.0 / (1 << 20) as f32;

struct Bits<'a> {
    words: &'a [u32],
    pos: usize,
    buf: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(words: &'a [u32], pos: usize) -> Self {
        Self { words, pos, buf: 0, n: 0 }
    }

    #[inline]
    fn fill(&mut self, need: u32) -> Option<()> {
        while self.n < need {
            let w = *self.words.get(self.pos)? as u64;
            self.buf |= w << self.n;
            self.pos += 1;
            self.n += 32;
        }
        Some(())
    }

    #[inline]
    fn read(&mut self, k: u32) -> Option<u32> {
        if k == 0 {
            return Some(0);
        }
        self.fill(k)?;
        let v = (self.buf & ((1u64 << k) - 1)) as u32;
        self.buf >>= k;
        self.n -= k;
        Some(v)
    }

    /// Decode the quotient prefix: a run of identical bits terminated by the
    /// opposite bit (at most eight bits in total).
    #[inline]
    fn quotient(&mut self) -> Option<i32> {
        // Eight bits are enough for the longest code; tolerate a short tail
        // at the very end of the stream.
        if self.fill(8).is_none() {
            self.fill(1)?;
        }
        let first = self.buf & 1;
        let mut run = 1;
        while run < 8 && (self.buf >> run) & 1 == first {
            run += 1;
        }
        let (q, len) = match (first == 1, run) {
            (true, 8) => (7, 8),
            (true, r) => (r as i32 - 1, r + 1),
            (false, 8) => (-8, 8),
            (false, r) => (-(r as i32), r + 1),
        };
        self.buf >>= len;
        self.n = self.n.saturating_sub(len);
        Some(q)
    }
}

#[inline]
fn sign24(v: u32) -> i32 {
    ((v << 8) as i32) >> 8
}

/// Decode one block into `out`; returns the number of words it occupied.
fn block(words: &[u32], start: usize, out: &mut [i32; BLOCK]) -> Option<usize> {
    let len_words = (*words.get(start)? & 0x1ff) as usize + 1;
    let mut br = Bits::new(words, start);
    br.read(9)?;
    let a = br.read(6)? as i32;
    let b = br.read(7)? as i32;
    let c3 = a - 64;
    let c2 = b - 63;
    let c1 = 47 - (b + c3);
    out[0] = sign24(br.read(24)?);
    out[1] = sign24(br.read(24)?);
    out[2] = sign24(br.read(24)?);
    let mut res = [0i32; BLOCK];
    for group in res.chunks_mut(32) {
        let k = br.read(5)?;
        for r in group.iter_mut() {
            let q = br.quotient()?;
            let low = br.read(k)? as i32;
            *r = q.wrapping_shl(k).wrapping_add(low);
        }
    }
    for n in 3..BLOCK {
        let p = c1.wrapping_mul(out[n - 3]).wrapping_add(c2.wrapping_mul(out[n - 2])).wrapping_add(c3.wrapping_mul(out[n - 1]));
        out[n] = res[n].wrapping_sub(p >> 4);
    }
    Some(len_words)
}

/// Returns one sample vector per channel at 44.1 kHz.
pub fn decode(data: &[u8]) -> Option<Vec<Vec<f32>>> {
    if data.len() < 16 {
        return None;
    }
    let words: Vec<u32> = data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let channels = words[0] as usize;
    let frames = words[1] as usize;
    if !(1..=2).contains(&channels) || frames == 0 || frames > 48_000 * 60 {
        return None;
    }
    let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(frames + BLOCK); channels];
    let mut pos = 3;
    let mut a = [0i32; BLOCK];
    let mut b = [0i32; BLOCK];
    while out[0].len() < frames {
        if channels == 1 {
            pos += block(&words, pos, &mut a)?;
            out[0].extend(a.iter().map(|&v| v as f32 * SCALE));
        } else {
            pos += block(&words, pos, &mut a)?;
            pos += block(&words, pos, &mut b)?;
            for i in 0..BLOCK {
                out[0].push(((a[i].wrapping_add(b[i].wrapping_mul(2))) >> 1) as f32 * SCALE);
                out[1].push(((a[i].wrapping_sub(b[i].wrapping_mul(2))) >> 1) as f32 * SCALE);
            }
        }
    }
    for ch in &mut out {
        ch.truncate(frames);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The catalog's default IR: one 512-frame mono block.
    const DEFAULT_IR: &str = "AQAAAAACAAAAAIA/dVIjAMgDAAAACAAACYIA5RyFnn9AIuRBEAQHKHlAOFeC4ACF/jmkPCAHKPRzJQgCwrlC+BxSfoCSB0EQBEEQBEEQHKCMOXygIOYQAsTPVID4mQofKIg5QPx8AmoOIYQwQvgA1Bwgfj4BNUP8TIUQQgjhA1Bz+ADUHD4ANZOAcK4EQUA4n4DyY0jnKPkBCj0/UILgAOUclB+TCPkI4QNQcwghhBAgfgbqZyqEEALEz1QIH4CaQxgQP1MhhA8UxBxCCCEEiJ+pD0DN4QNQcwghQIyfqRACxM8noH4Gag4hQPx8AmoOAeJnKoQQIH6mSBAEByh5QDhH+RxSHgQB4Rwl/wBIj5UgCAjnygEKoZ8rwQFKHgQB4Vw5UAjnoDxWgiA4QMkDwrkSBCQICOdKEAQB4XNIeRAQzpUDlJzwOaRzlDwIDkB5TFDOlYBwrhyA8niCQj+fgPJYOUCh5x8AaXwwUQJCfkBQ8iAIDlDyIAgCwjnK55DyA5RzFPrnkPIgCA5AGZMPINE/h5QHByh5EAQHKHlwgJIHByh5EASEcyUISBAQPoeUB8EBSh4EwQFKHgQHKHkQBEEQBEEQjBBCCAHi5xMFMYfwAaj5A1BzCBA/UwFi/kCFDwAAAA==";

    #[test]
    fn decodes_default_ir() {
        let data = ezmix_content::engine::decode_b64(DEFAULT_IR);
        let chans = decode(&data).expect("decodes");
        assert_eq!(chans.len(), 1);
        assert_eq!(chans[0].len(), 512);
        assert!(chans[0].iter().all(|v| v.is_finite() && v.abs() <= 1.5));
        assert!(chans[0].iter().any(|v| v.abs() > 0.01));
    }
}
