//! Minimal RIFF/WAVE reader for impulse responses: integer PCM (8–32 bit)
//! and IEEE float, plain or `WAVE_FORMAT_EXTENSIBLE`.

pub struct Wav {
    pub sample_rate: f32,
    /// One vector per channel (at most two are kept).
    pub channels: Vec<Vec<f32>>,
}

/// Longest IR kept, in frames (about 5 s at 48 kHz).
pub const MAX_FRAMES: usize = 1 << 18;

pub fn parse(bytes: &[u8]) -> Result<Wav, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
    let mut fmt = None;
    let mut data = None;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32_at(bytes, pos + 4) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                let mut tag = u16_at(body, 0);
                // WAVE_FORMAT_EXTENSIBLE: the real format leads the sub-format GUID.
                if tag == 0xfffe && body.len() >= 26 {
                    tag = u16_at(body, 24);
                }
                fmt = Some((tag, u16_at(body, 2) as usize, u32_at(body, 4) as f32, u16_at(body, 14) as usize));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    let (tag, n_ch, sample_rate, bits) = fmt.ok_or("missing format chunk")?;
    let data = data.ok_or("missing data chunk")?;
    let width = bits.div_ceil(8);
    if n_ch == 0 || width == 0 || sample_rate <= 0.0 {
        return Err("invalid format".into());
    }
    let decode: fn(&[u8]) -> f32 = match (tag, width) {
        (1, 1) => |b| (b[0] as f32 - 128.0) / 128.0,
        (1, 2) => |b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
        (1, 3) => |b| i32::from_le_bytes([0, b[0], b[1], b[2]]) as f32 / 2_147_483_648.0,
        (1, 4) => |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
        (3, 4) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (3, 8) => |b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        _ => return Err(format!("unsupported WAV encoding (format {tag}, {bits} bit)")),
    };
    let keep = n_ch.min(2);
    let mut channels = vec![Vec::new(); keep];
    for frame in data.chunks_exact(width * n_ch).take(MAX_FRAMES) {
        for (c, out) in channels.iter_mut().enumerate() {
            let v = decode(&frame[c * width..(c + 1) * width]);
            out.push(if v.is_finite() { v } else { 0.0 });
        }
    }
    if channels[0].is_empty() {
        return Err("empty WAV file".into());
    }
    Ok(Wav { sample_rate, channels })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(tag: u16, channels: u16, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend(16u32.to_le_bytes());
        b.extend(tag.to_le_bytes());
        b.extend(channels.to_le_bytes());
        b.extend(44100u32.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        b.extend((channels * bits / 8).to_le_bytes());
        b.extend(bits.to_le_bytes());
        b.extend(b"data");
        b.extend((data.len() as u32).to_le_bytes());
        b.extend(data);
        b
    }

    #[test]
    fn decodes_pcm_and_float() {
        // 24-bit stereo: one frame of (+0.5, -0.5).
        let w = parse(&wav(1, 2, 24, &[0, 0, 0x40, 0, 0, 0xc0])).unwrap();
        assert_eq!(w.sample_rate, 44100.0);
        assert_eq!(w.channels, vec![vec![0.5], vec![-0.5]]);
        let w = parse(&wav(3, 1, 32, &0.25f32.to_le_bytes())).unwrap();
        assert_eq!(w.channels, vec![vec![0.25]]);
        assert!(parse(b"RIFF").is_err());
    }
}
